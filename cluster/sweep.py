"""Sweeps of SFOLS/ experiments on SLURM: one YAML spec -> a grid of runs -> one directory per run -> a job array.

    python cluster/sweep.py submit cluster/sweeps/smoke.yaml [--dry-run] [--live] [--name NAME]
    python cluster/sweep.py status [SWEEP_DIR]          # no argument: every sweep under SFOLS/runs/
    python cluster/sweep.py resubmit SWEEP_DIR [--all-unfinished]
    python cluster/sweep.py collect SWEEP_DIR           # writes SWEEP_DIR/index.csv
    python cluster/sweep.py merge SWEEP_DIR --into DIR  # gather outputs (list pickles of the same name concatenated)
    python cluster/sweep.py local SWEEP_DIR RUN_INDEX   # run one run here, without SLURM (debugging)

Layout of a sweep (SFOLS/runs/<name>-<timestamp>/):
    spec.yaml, git.txt (commit + status), git.diff (uncommitted changes, if any)
    code/       snapshot of SFOLS/*.py (+ symlinks code/RM -> RM, code/SFOLS/results -> SFOLS/results), so queued
                jobs are not affected by later edits; --live runs from the working tree instead
    runs.json   the expanded runs: index, id, params, env, command
    tasks*.txt  line i = run indices executed by array task i (several when slurm.pack > 1)
    job.sbatch, slurm/  the job script and SLURM's own stdout/stderr
    r/<id>/     per run: log.txt (stdout + stderr), status.json, out/ (SF_OUT: every output the script writes)

In loader code: `from sweep import load_sweep` (with cluster/ on sys.path) gives (params, status, out_dir) per run.
"""
import argparse
import csv
import datetime as dt
import glob
import itertools
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SFOLS = os.path.join(REPO, "SFOLS")
RUNS = os.path.join(SFOLS, "runs")
PYTHON = os.environ.get("SF_PYTHON", os.path.expanduser("~/miniconda3/envs/sffsa/bin/python"))
SLURM_DEFAULTS = dict(partition="amd", cpus=1, mem="4G", time="12:00:00", max_parallel=8, pack=1)


# ---------------------------------------------------------------------------------------------------------- expand

def expand(spec):
    """Runs of a spec: the cartesian product of `grid`, plus every entry of `runs` (explicit combinations).
    UPPERCASE keys are exported as environment variables; lowercase keys fill {placeholders} in `command`
    ({cpus} = slurm.cpus is always available); a `name` key, if given, is the run's directory name."""
    grid = spec.get("grid") or {}
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(_as_list(grid[k]) for k in keys))] if keys else []
    combos += [dict(r) for r in spec.get("runs") or []]
    if not combos:
        combos = [{}]
    cpus = int(spec["slurm"]["cpus"])
    runs = []
    for i, p in enumerate(combos):
        env = {k: str(v) for k, v in (spec.get("env") or {}).items()}
        env.update({k: str(v) for k, v in p.items() if k.isupper()})
        fill = {k: v for k, v in p.items() if not k.isupper()}
        fill.setdefault("cpus", cpus)
        try:
            cmd = spec["command"].format(**fill)
            for _ in range(3):                    # a placeholder's value may itself hold placeholders ({cmd} -> {cpus})
                cmd, prev = cmd.format(**fill), cmd
                if cmd == prev:
                    break
        except KeyError as e:
            sys.exit(f"command placeholder {e} is not a lowercase key of grid/runs")
        rid = f"{i:03d}_{_slug(p['name'])}" if "name" in p else \
            f"{i:03d}" + "".join(f"_{_slug(k)}-{_slug(v)}" for k, v in p.items())
        runs.append(dict(index=i, id=rid[:120], params=p, env=env, command=cmd))
    return runs


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _slug(x):
    return re.sub(r"[^A-Za-z0-9.+-]+", "", str(x).replace(",", "+"))


# ---------------------------------------------------------------------------------------------------------- submit

def cmd_submit(a):
    spec = yaml.safe_load(open(a.spec))
    spec["slurm"] = {**SLURM_DEFAULTS, **(spec.get("slurm") or {})}
    name = a.name or spec.get("name") or os.path.splitext(os.path.basename(a.spec))[0]
    runs = expand(spec)
    sdir = os.path.join(RUNS, f"{name}-{dt.datetime.now():%Y%m%d-%H%M%S}")
    print(f"{name}: {len(runs)} runs -> {os.path.relpath(sdir, REPO)}")
    for r in runs[:5]:
        print(f"  [{r['index']}] {' '.join(f'{k}={v}' for k, v in r['env'].items())} {r['command']}")
    if len(runs) > 5:
        print(f"  ... ({len(runs) - 5} more)")
    os.makedirs(os.path.join(sdir, "slurm"))
    shutil.copy(a.spec, os.path.join(sdir, "spec.yaml"))
    _provenance(sdir)
    code = os.path.join(sdir, "code", "SFOLS") if not a.live else SFOLS
    if not a.live:
        _snapshot(code)
    meta = dict(name=name, spec=spec, code=code, live=a.live, runs=runs, created=time.time())
    json.dump(meta, open(os.path.join(sdir, "runs.json"), "w"), indent=1)
    for r in runs:
        os.makedirs(os.path.join(sdir, "r", r["id"], "out"))
    _write_job(sdir, spec["slurm"])
    pack = int(spec["slurm"]["pack"])
    tasks = [list(range(i, min(i + pack, len(runs)))) for i in range(0, len(runs), pack)]
    _submit(sdir, tasks, "tasks.txt", spec["slurm"], dry=a.dry_run)


def _provenance(sdir):
    def git(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout
    with open(os.path.join(sdir, "git.txt"), "w") as fh:
        fh.write(f"commit {git('rev-parse', 'HEAD').strip()}\n{git('status', '--short', '--', 'SFOLS', 'cluster')}")
    diff = git("diff", "HEAD", "--", "SFOLS", "cluster")
    if diff:
        open(os.path.join(sdir, "git.diff"), "w").write(diff)
        print("  note: uncommitted changes in SFOLS/ or cluster/ (saved to git.diff)")


def _snapshot(code):
    """Copy every .py under SFOLS/ (not runs/, results/), and link the inputs, so relative paths resolve as in SFOLS/."""
    for src in glob.glob(os.path.join(SFOLS, "**", "*.py"), recursive=True):
        rel = os.path.relpath(src, SFOLS)
        if rel.split(os.sep)[0] in ("runs", "results"):
            continue
        dst = os.path.join(code, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    os.symlink(os.path.join(SFOLS, "results"), os.path.join(code, "results"))
    os.symlink(os.path.join(REPO, "RM"), os.path.join(os.path.dirname(code), "RM"))


def _write_job(sdir, sl):
    pack = int(sl["pack"])
    extra = "".join(f"#SBATCH {x}\n" for x in _as_list(sl.get("extra") or []))
    job = f"""#!/bin/bash
#SBATCH --job-name={os.path.basename(sdir)[:40]}
#SBATCH --partition={sl['partition']}
#SBATCH --cpus-per-task={int(sl['cpus']) * pack}
#SBATCH --mem={_mem_times(sl['mem'], pack)}
#SBATCH --time={sl['time']}
#SBATCH --output={sdir}/slurm/%A_%a.out
{extra}# usage: sbatch --array=0-N%K job.sbatch TASKS_FILE   (written by cluster/sweep.py)
exec {PYTHON} {os.path.abspath(__file__)} exec {sdir} "$1" "$SLURM_ARRAY_TASK_ID"
"""
    open(os.path.join(sdir, "job.sbatch"), "w").write(job)


def _mem_times(mem, k):
    m = re.fullmatch(r"(\d+)([KMGT]?)", str(mem).strip().upper())
    if not m:
        sys.exit(f"slurm.mem must look like 4G or 500M, got {mem!r}")
    return f"{int(m.group(1)) * k}{m.group(2)}"


def _submit(sdir, tasks, tasks_name, sl, dry=False):
    with open(os.path.join(sdir, tasks_name), "w") as fh:
        fh.write("".join(" ".join(map(str, t)) + "\n" for t in tasks))
    cmd = ["sbatch", "--parsable", f"--array=0-{len(tasks) - 1}%{int(sl['max_parallel'])}",
           os.path.join(sdir, "job.sbatch"), os.path.join(sdir, tasks_name)]
    if dry:
        print(f"  dry run, not submitted: {' '.join(cmd)}")
        return
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    jid = re.findall(r"^(\d+)(?:;\S+)?$", res.stdout, re.M)   # this cluster's sbatch also prints a banner on stdout
    if not jid:
        sys.exit(f"could not parse the job id from sbatch:\n{res.stdout}\n{res.stderr}")
    jid = jid[-1]
    with open(os.path.join(sdir, "jobs.txt"), "a") as fh:
        fh.write(f"{jid} {tasks_name}\n")
    print(f"  submitted job array {jid} ({len(tasks)} tasks, at most {sl['max_parallel']} at once)")


# ------------------------------------------------------------------------------------------------------------ exec

def cmd_exec(a):
    """Body of one array task: run its runs (concurrently when packed) and record their status."""
    meta = json.load(open(os.path.join(a.sweep, "runs.json")))
    line = open(a.tasks_file).read().splitlines()[int(a.task)]
    idx = [int(x) for x in line.split()]
    procs = [_start(a.sweep, meta, meta["runs"][i]) for i in idx]
    rc = 0
    for p, fh, st, path in procs:
        code = p.wait()
        fh.close()
        st.update(state="done" if code == 0 else "failed", returncode=code, end=time.time())
        st["elapsed_s"] = round(st["end"] - st["start"], 1)
        json.dump(st, open(path, "w"), indent=1)
        rc = rc or code
    sys.exit(rc)


def _start(sdir, meta, r, foreground=False):
    rdir = os.path.join(sdir, "r", r["id"])
    cpus = int(meta["spec"]["slurm"]["cpus"])
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS=str(cpus),
               PYTHONUNBUFFERED="1", PYTHONPATH=meta["code"], SF_OUT=os.path.join(rdir, "out"),
               PATH=os.path.dirname(PYTHON) + os.pathsep + env.get("PATH", ""), MPLBACKEND="Agg")
    env.update(r["env"])
    st = dict(state="running", host=socket.gethostname(), start=time.time(), command=r["command"], env=r["env"],
              job=f"{os.environ.get('SLURM_ARRAY_JOB_ID', '-')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '-')}")
    path = os.path.join(rdir, "status.json")
    json.dump(st, open(path, "w"), indent=1)
    fh = None if foreground else open(os.path.join(rdir, "log.txt"), "w")
    if fh:
        fh.write(f"# {dt.datetime.now():%Y-%m-%d %H:%M:%S} on {st['host']} ({st['job']}), cwd {meta['code']}\n"
                 f"# {' '.join(f'{k}={v}' for k, v in r['env'].items())} {r['command']}\n")
        fh.flush()
    p = subprocess.Popen(["bash", "-c", r["command"]], cwd=meta["code"], env=env, stdout=fh,
                         stderr=subprocess.STDOUT if fh else None)
    return p, fh or open(os.devnull, "w"), st, path


def cmd_local(a):
    meta = json.load(open(os.path.join(a.sweep, "runs.json")))
    p, fh, st, path = _start(a.sweep, meta, meta["runs"][int(a.index)], foreground=True)
    code = p.wait()
    st.update(state="done" if code == 0 else "failed", returncode=code, end=time.time(),
              elapsed_s=round(time.time() - st["start"], 1))
    json.dump(st, open(path, "w"), indent=1)
    sys.exit(code)


# ------------------------------------------------------------------------------------------------- status / resubmit

def _states(sdir):
    meta = json.load(open(os.path.join(sdir, "runs.json")))
    queue = _queue(sdir)
    out = []
    for r in meta["runs"]:
        f = os.path.join(sdir, "r", r["id"], "status.json")
        st = json.load(open(f)) if os.path.exists(f) else {"state": "queued" if queue else "not run"}
        if st["state"] == "running" and st.get("job") not in queue:
            st["state"] = "lost"                  # killed (time limit, node failure, scancel) before it could record
        out.append((r, st))
    return meta, out


def _queue(sdir):
    """SLURM ids (JOB_TASK, one per array task) of this sweep's jobs that are still pending or running."""
    f = os.path.join(sdir, "jobs.txt")
    if not os.path.exists(f):
        return set()
    ids = [l.split()[0] for l in open(f).read().splitlines() if l.strip()]
    q = subprocess.run(["squeue", "-h", "-r", "-o", "%i", "-j", ",".join(ids)], capture_output=True, text=True).stdout
    return set(q.split())


def cmd_status(a):
    if not a.sweep:
        for sdir in sorted(glob.glob(os.path.join(RUNS, "*", "runs.json"))):
            sdir = os.path.dirname(sdir)
            _, sts = _states(sdir)
            print(f"{os.path.basename(sdir):45s} {_counts(sts)}")
        return
    meta, sts = _states(a.sweep)
    print(f"{meta['name']}: {_counts(sts)}")
    for r, st in sts:
        el = st.get("elapsed_s") or (time.time() - st["start"] if "start" in st else None)
        print(f"  [{r['index']:3d}] {st['state']:8s} {_hms(el):>9s} {st.get('host', ''):6s} {r['id']}")
    for r, st in sts:
        if st["state"] in ("failed", "lost"):
            log = os.path.join(a.sweep, "r", r["id"], "log.txt")
            tail = open(log).read().splitlines()[-a.tail:] if os.path.exists(log) else []
            print(f"\n--- {r['id']} ({st['state']}), last lines of log.txt:\n" + "\n".join(tail))


def _counts(sts):
    c = {}
    for _, st in sts:
        c[st["state"]] = c.get(st["state"], 0) + 1
    return ", ".join(f"{v} {k}" for k, v in sorted(c.items()))


def _hms(s):
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def cmd_resubmit(a):
    meta, sts = _states(a.sweep)                  # failed/lost runs are finished, even if other tasks still run
    bad = ("failed", "lost") + (("not run",) if a.all_unfinished else ())
    idx = [r["index"] for r, st in sts if st["state"] in bad]
    if not idx:
        print("nothing to resubmit")
        return
    for i in idx:
        r = meta["runs"][i]
        f = os.path.join(a.sweep, "r", r["id"], "status.json")
        if os.path.exists(f):
            os.rename(f, f + f".{int(time.time())}.old")
    sl = meta["spec"]["slurm"]
    pack = int(sl["pack"])
    tasks = [idx[i:i + pack] for i in range(0, len(idx), pack)]
    n = len(glob.glob(os.path.join(a.sweep, "tasks*.txt")))
    print(f"resubmitting {len(idx)} runs: {idx}")
    _submit(a.sweep, tasks, f"tasks_{n}.txt", sl, dry=a.dry_run)


# ---------------------------------------------------------------------------------------------------------- collect

def load_sweep(sdir, unpickle=False):
    """[(params, status, out_dir or {file: object})] for every run of a sweep."""
    import pickle
    _, sts = _states(sdir)
    rows = []
    for r, st in sts:
        out = os.path.join(sdir, "r", r["id"], "out")
        if unpickle:
            out = {os.path.relpath(f, out): pickle.load(open(f, "rb"))
                   for f in glob.glob(os.path.join(out, "**", "*.pkl"), recursive=True)}
        rows.append((dict(r["params"]), st, out))
    return rows


def cmd_collect(a):
    meta, sts = _states(a.sweep)
    keys = sorted({k for r in meta["runs"] for k in r["params"]})
    path = os.path.join(a.sweep, "index.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "id", *keys, "state", "elapsed_s", "host", "outputs"])
        for r, st in sts:
            out = os.path.join(a.sweep, "r", r["id"], "out")
            files = sorted(os.path.relpath(f, out) for f in glob.glob(os.path.join(out, "**", "*"), recursive=True)
                           if os.path.isfile(f))
            w.writerow([r["index"], r["id"], *(r["params"].get(k, "") for k in keys), st["state"],
                        st.get("elapsed_s", ""), st.get("host", ""), " ".join(files)])
    print(f"{_counts(sts)} -> {os.path.relpath(path, REPO)}")


def cmd_merge(a):
    """Gather the outputs of finished runs into one directory: a file name produced by one run is copied; produced by
    several runs it is concatenated if every copy is a pickled list (e.g. q_ccs_stochastic's per-seed run lists)."""
    import pickle
    _, sts = _states(a.sweep)
    only = {int(x) for x in a.only.split(",")} if a.only else None
    files = {}
    for r, st in sts:
        if st["state"] != "done" or (only is not None and r["index"] not in only):
            continue
        out = os.path.join(a.sweep, "r", r["id"], "out")
        for f in sorted(glob.glob(os.path.join(out, "**", "*"), recursive=True)):
            if os.path.isfile(f):
                files.setdefault(os.path.relpath(f, out), []).append(f)
    plan = []
    for rel, srcs in sorted(files.items()):
        dst = os.path.join(a.into, rel)
        if os.path.exists(dst) and not a.force:
            sys.exit(f"{dst} exists; pass --force to overwrite (nothing was written)")
        if len(srcs) > 1:
            objs = [pickle.load(open(f, "rb")) for f in srcs] if rel.endswith(".pkl") else None
            if not objs or not all(isinstance(o, list) for o in objs):
                sys.exit(f"{rel} is written by {len(srcs)} runs and is not a list pickle; merge them by hand or "
                         f"select runs with --only (nothing was written)")
        plan.append((rel, dst, srcs))
    for rel, dst, srcs in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if len(srcs) == 1:
            shutil.copy2(srcs[0], dst)
        else:
            pickle.dump([x for f in srcs for x in pickle.load(open(f, "rb"))], open(dst, "wb"))
        print(f"  {rel}  <- {len(srcs)} run(s)")
    print(f"{len(plan)} files -> {a.into}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("submit")
    p.add_argument("spec")
    p.add_argument("--name")
    p.add_argument("--dry-run", action="store_true", help="create the sweep directory but do not call sbatch")
    p.add_argument("--live", action="store_true", help="run from the working tree instead of a code snapshot")
    p.set_defaults(f=cmd_submit)
    p = sp.add_parser("status")
    p.add_argument("sweep", nargs="?")
    p.add_argument("--tail", type=int, default=15)
    p.set_defaults(f=cmd_status)
    p = sp.add_parser("resubmit")
    p.add_argument("sweep")
    p.add_argument("--all-unfinished", action="store_true", help="also runs that never started")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(f=cmd_resubmit)
    p = sp.add_parser("collect")
    p.add_argument("sweep")
    p.set_defaults(f=cmd_collect)
    p = sp.add_parser("merge")
    p.add_argument("sweep")
    p.add_argument("--into", required=True, help="target directory, e.g. SFOLS/results to feed the plot scripts")
    p.add_argument("--only", help="comma-separated run indices (default: every finished run)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(f=cmd_merge)
    p = sp.add_parser("local")
    p.add_argument("sweep")
    p.add_argument("index")
    p.set_defaults(f=cmd_local)
    p = sp.add_parser("exec")                     # internal: called by job.sbatch
    p.add_argument("sweep")
    p.add_argument("tasks_file")
    p.add_argument("task")
    p.set_defaults(f=cmd_exec)
    a = ap.parse_args()
    if getattr(a, "sweep", None):
        a.sweep = os.path.abspath(a.sweep.rstrip("/"))
    a.f(a)


if __name__ == "__main__":
    main()
