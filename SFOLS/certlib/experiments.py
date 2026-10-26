"""Experiments for the certified-library note (results in SFOLS/results/certlib/).

usage: PYTHONPATH=SFOLS python -m certlib.experiments <command> [args]
  build <domain> <ladder> <eps> [eta]   one anytime run of the library construction (log + library)
  truth <domain>                        held-out weights and V* (ground truth, reporting only)
  curve <domain> <ladder> <eps>         true GPI loss of library prefixes along a saved run
  tasks <domain> <ladder> <eps>         zero-shot SF-FSA-VI regret of library prefixes on random tasks
  sampled <domain> <n> <eps> <seed>     the construction on an empirical model from n samples per (s, a)
  test <domain> <eps> <n_learned>       sufficiency test of the first n learned policies of the cl run
domains: ds (double slit, wind 3), office0 / office1 / office3 (Office, slip 0 / 0.1 / 0.3), windy
"""
import os
import pickle
import sys
import time

import numpy as np

from . import domains
from .cl import Builder
from .fsa import PAPER_TASKS, library_value, optimal_value, random_tasks, windy_tasks
from .library import Library
from .truth import gpi_loss, vstar

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "certlib")
SAVE_DIR = os.environ.get("SF_OUT") or OUT      # outputs (cluster/sweep.py sets SF_OUT); inputs are also looked up in OUT
os.makedirs(SAVE_DIR, exist_ok=True)


def get(domain):
    """(mdp, faces). Office objects are ordered C1 C2 O1 O2 M1 M2 (symbols C, O, M).
      office<p>      full simplex (every FSA)
      office<p>f2    FSAs whose states branch on at most 2 symbols: the three 4-exit faces
      windy          two-stage tasks: {A, B} and every pair of corridor exits (all 1-D)"""
    import itertools
    if domain == "ds":
        return domains.double_slit(3), [[0, 1]]
    if domain.startswith("office"):
        p = {"0": 0.0, "1": 0.1, "3": 0.3}[domain[6]]
        mdp = domains.office(p)
        if domain.endswith("f2"):
            return mdp, [[0, 1, 2, 3], [0, 1, 4, 5], [2, 3, 4, 5]]
        return mdp, [list(range(6))]
    if domain == "windy":
        mdp = domains.windy_two_stage()
        return mdp, [[0, 1]] + [list(c) for c in itertools.combinations([2, 3, 4, 5], 2)]
    raise ValueError(domain)


def fname(*parts):
    name = "_".join(str(p) for p in parts) + ".pkl"
    f = os.path.join(SAVE_DIR, name)
    return f if os.path.exists(f) else os.path.join(OUT, name)


def save(obj, *parts):
    with open(os.path.join(SAVE_DIR, "_".join(str(p) for p in parts) + ".pkl"), "wb") as fh:
        pickle.dump(obj, fh)


def load(*parts):
    with open(fname(*parts), "rb") as fh:
        return pickle.load(fh)


def library_from(mdp, run, upto=None):
    """Rebuild the library (SFs by evaluation) from a saved run's action tables."""
    lib = Library(mdp)
    n = len(run["acts"]) if upto is None else upto
    s0 = mdp.sweeps
    for a, k, w in zip(run["acts"][:n], run["kinds"][:n], run["weights"][:n]):
        lib.add(a, mdp.evaluate(np.asarray(a, int), tol=1e-12), k, w)
    mdp.sweeps = s0
    return lib


def cmd_build(domain, ladder, eps, eta=0.0, max_queries=3000):
    mdp, face = get(domain)
    b = Builder(mdp, face, float(eps), ladder=ladder, eta=float(eta), verbose=True)

    def snapshot(bb, final=False):
        run = dict(domain=domain, ladder=ladder, eps=float(eps), eta=float(eta), log=bb.log, count=bb.count,
                   final_gap=bb.final_gap if final else [e["gap"] for e in bb.log if e["gap"] is not None][-1],
                   time=getattr(bb, "time", None), acts=[a.astype(np.int16) for a in bb.lib.acts],
                   kinds=bb.lib.kinds, weights=bb.lib.weights, open=[(r, w) for _, r, w in bb.open],
                   complete=final)
        save(run, "run", domain, ladder, eps, eta)
    b.checkpoint = snapshot
    b.run(max_queries=int(max_queries), exact_final=True)
    snapshot(b, final=True)
    print(f"{domain} {ladder} eps {eps} eta {eta}: queries {len(b.log) - len(b.face)} library {len(b.lib)} learned "
          f"{b.lib.kinds.count('learned')} {b.count} sweeps {b.log[-1]['sweeps']} cert {b.final_gap:.2e} "
          f"time {b.time:.0f}s", flush=True)


def test_weights(domain, mdp, rng):
    """Held-out weights inside the certified region: a 2001-point grid for double slit; otherwise 400
    weights, each drawn uniformly on a random face of the union (and 1/4 of them on a random sub-face)."""
    if domain == "ds":
        x = np.linspace(0, 1, 2001)
        return np.stack([x, 1 - x], 1)
    _, faces = get(domain)
    W = []
    for i in range(400):
        F = np.asarray(faces[rng.integers(len(faces))])
        if i % 4 == 3 and len(F) > 2:
            F = rng.choice(F, rng.integers(2, len(F)), replace=False)
        w = np.zeros(mdp.D); w[F] = rng.dirichlet(np.ones(len(F)))
        W.append(w)
    return np.array(W)


def cmd_truth(domain):
    mdp, _ = get(domain)
    W = test_weights(domain, mdp, np.random.default_rng(0))
    t0 = time.perf_counter()
    VS = vstar(mdp, W)
    save(dict(W=W, VS=VS), "truth", domain)
    print(f"truth {domain}: {len(W)} weights in {time.perf_counter() - t0:.0f}s", flush=True)


def checkpoints(n, dense=40):
    ks = sorted(set(np.unique(np.round(np.geomspace(1, n, dense)).astype(int))) | {n})
    return [k for k in ks if k >= 1]


def cmd_curve(domain, ladder, eps, eta=0.0):
    mdp, face = get(domain)
    run = load("run", domain, ladder, eps, eta)
    T = load("truth", domain)
    full = library_from(mdp, run)
    out = []
    cache = {}
    for k in checkpoints(len(run["acts"])):
        lib = Library(mdp)
        lib.stack, lib.acts, lib.kinds = full.stack[:k], full.acts[:k], full.kinds[:k]
        L = gpi_loss(mdp, lib, T["W"], T["VS"], cache)
        out.append(dict(library=k, learned=lib.kinds.count("learned"), max_loss=float(L.max()),
                        mean_loss=float(L.mean()), max_loss_start=float(L[:, 0].max())))
        print(f"  {domain} {ladder} library {k:4d} learned {out[-1]['learned']:3d} true max {L.max():.2e}", flush=True)
    save(out, "curve", domain, ladder, eps, eta)


def tasks_for(domain, mdp, rng):
    if domain == "windy":
        return windy_tasks(60, rng)
    if domain.startswith("office"):
        return list(PAPER_TASKS.values()) + random_tasks(100, rng, max_states=4, max_out=2)
    raise ValueError("FSA tasks are defined for office and windy")


def cmd_tasks(domain, ladder, eps, eta=0.0):
    mdp, face = get(domain)
    run = load("run", domain, ladder, eps, eta)
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    f = fname("taskopt", domain)
    if os.path.exists(f):
        opt = load("taskopt", domain)
    else:
        cache = {}
        opt = [optimal_value(mdp, t, cache) for t in tasks]
        save(opt, "taskopt", domain)
    full = library_from(mdp, run)
    ok = np.array(opt) > 1e-9
    out = []
    for k in checkpoints(len(run["acts"]), dense=25):
        lib = Library(mdp)
        lib.stack, lib.acts, lib.kinds = full.stack[:k], full.acts[:k], full.kinds[:k]
        cache = {}
        vals = np.array([library_value(mdp, t, lib, cache) for t in tasks])
        reg = (np.array(opt) - vals)
        rel = reg[ok] / np.array(opt)[ok]
        out.append(dict(library=k, learned=lib.kinds.count("learned"), max_regret=float(reg.max()),
                        mean_regret=float(reg.mean()), max_rel=float(rel.max()), mean_rel=float(rel.mean())))
        print(f"  tasks {domain} {ladder} library {k:4d} learned {out[-1]['learned']:3d} max regret {reg.max():.2e} "
              f"(rel {rel.max():.2%})", flush=True)
    save(out, "tasks", domain, ladder, eps, eta)


def cmd_sampled(domain, n, eps, seed):
    mdp, face = get(domain)
    hat = domains.sampled(mdp, int(n), np.random.default_rng(int(seed)))
    b = Builder(hat, face, float(eps), ladder="cl").run()
    # the library's policies (chosen with the empirical SFs) executed in the TRUE MDP
    T = load("truth", domain)
    L = gpi_loss(mdp, b.lib, T["W"], T["VS"]) if True else None
    # GPI decisions use the empirical SFs; gpi_loss above uses lib.stack (empirical) for decisions and
    # evaluates the chosen policies in the true MDP, which is what deploying the library means
    res = dict(n=int(n), seed=int(seed), eps=float(eps), cert=b.final_gap, true_max=float(L.max()),
               true_mean=float(L.mean()), library=len(b.lib), learned=b.lib.kinds.count("learned"),
               count=b.count, sweeps=b.log[-1]["sweeps"])
    save(res, "sampled", domain, n, eps, seed)
    print(f"sampled {domain} n={n} seed {seed}: cert (empirical model) {b.final_gap:.2e}, true max loss "
          f"{L.max():.2e}, library {len(b.lib)}, learned {res['learned']}", flush=True)


def cmd_test(domain, eps, n_learned):
    """Sufficiency test: base = the first n_learned learned policies of the cl run (in order)."""
    mdp, face = get(domain)
    run = load("run", domain, "cl", eps, 0.0)
    idx = [i for i, k in enumerate(run["kinds"]) if k == "learned"][:int(n_learned)]
    b = Builder(mdp, face, float(eps), ladder="test")
    for i in idx:
        a = np.asarray(run["acts"][i], int)
        w = run["weights"][i]
        psi = mdp.evaluate(a, tol=1e-12)
        b._add(a, psi, "learned", w, mdp.reach_bound(mdp.gap(psi, a, w)))
    b.run(init=False)                       # the base already contains the vertices (first learned)
    T = load("truth", domain)
    L = gpi_loss(mdp, b.lib, T["W"], T["VS"])
    res = dict(n_learned=len(idx), cert=b.final_gap, true_max=float(L.max()), library=len(b.lib),
               count=b.count, open=len(b.open), sweeps=b.log[-1]["sweeps"] if b.log else 0)
    save(res, "test", domain, eps, n_learned)
    print(f"test {domain} base {len(idx)}: certified {b.final_gap:.2e} true {L.max():.2e} library {len(b.lib)} "
          f"{b.count} open corners {len(b.open)}", flush=True)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"build": cmd_build, "truth": cmd_truth, "curve": cmd_curve, "tasks": cmd_tasks,
     "sampled": cmd_sampled, "test": cmd_test}[cmd](*args)
