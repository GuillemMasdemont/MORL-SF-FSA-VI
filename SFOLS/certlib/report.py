"""Tables and figures of the certified-library note from the saved runs (notes/certlib_results.tex).

usage: PYTHONPATH=SFOLS python -m certlib.report <table> ...
  stops <domain> <eps,...>     for each method's run, the state at which tolerance eps is certified, the true
                               GPI loss of that library on held-out weights, and (office/windy) zero-shot task regret
The anytime property makes one run per method enough: tolerance eps is reached at the first query whose
largest gap is <= eps; the library at that moment is the prefix built before that query.
"""
import glob
import os
import pickle
import sys

import numpy as np

from .experiments import OUT, get, load, save, tasks_for
from .fsa import library_value
from .library import Library
from .truth import gpi_loss

METHODS = [("sfols_cold", "SFOLS"), ("sfols", "SFOLS, warm start"), ("cog", "Certify-or-Grow"),
           ("cl", "Certified library")]


def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def find_run(domain, ladder):
    """The saved run of this method with the smallest eps (largest anytime range)."""
    n = ladder.count("_")
    fs = [f for f in glob.glob(os.path.join(OUT, f"run_{domain}_{ladder}_*_0.0.pkl"))
          if "_".join(os.path.basename(f).split("_")[2:3 + n]) == ladder
          and _isfloat(os.path.basename(f).split("_")[3 + n])]
    if not fs:
        return None
    best = min(fs, key=lambda f: float(os.path.basename(f).split("_")[3 + ladder.count("_")]))
    with open(best, "rb") as fh:
        return pickle.load(fh)


def stop_state(run, eps):
    """(library size, learned, sweeps, queries, certified gap) when eps is first certified, or None."""
    log = run["log"]
    nv = sum(1 for e in log if e["gap"] is None)
    for i, e in enumerate(log):
        if e["gap"] is not None and e["gap"] <= eps:
            prev = log[i - 1]
            return dict(library=prev["library"], learned=prev["learned"], sweeps=prev["sweeps"],
                        queries=i - nv, gap=e["gap"])
    if run["final_gap"] <= eps:
        last = log[-1]
        return dict(library=last["library"], learned=last["learned"], sweeps=last["sweeps"],
                    queries=len(log) - nv, gap=run["final_gap"])
    return None


def prefix_library(mdp, run, n, cache):
    lib = Library(mdp)
    s0 = mdp.sweeps
    for a, k, w in zip(run["acts"][:n], run["kinds"][:n], run["weights"][:n]):
        key = np.asarray(a).tobytes()
        if key not in cache:
            cache[key] = mdp.evaluate(np.asarray(a, int), tol=1e-12)
        lib.add(a, cache[key], k, w)
    mdp.sweeps = s0
    return lib


def cmd_stops(domain, eps_list):
    mdp, face = get(domain)
    T = load("truth", domain)
    eps_list = [float(e) for e in eps_list.split(",")]
    with_tasks = domain.startswith("office") or domain == "windy"
    if with_tasks:
        tasks = tasks_for(domain, mdp, np.random.default_rng(1))
        opt = np.array(load("taskopt", domain)) if os.path.exists(os.path.join(OUT, f"taskopt_{domain}.pkl")) else None
        if opt is None:
            from .fsa import optimal_value
            c = {}
            opt = np.array([optimal_value(mdp, t, c) for t in tasks])
            save(list(opt), "taskopt", domain)
    rows = []
    sfcache = {}
    for lad, name in METHODS:
        run = find_run(domain, lad)
        if run is None:
            continue
        for eps in eps_list:
            st = stop_state(run, eps)
            if st is None:
                continue
            lib = prefix_library(mdp, run, st["library"], sfcache)
            L = gpi_loss(mdp, lib, T["W"], T["VS"])
            row = dict(method=name, eps=eps, **st, true_max=float(L.max()), true_mean=float(L.mean()))
            if with_tasks:
                cache = {}
                vals = np.array([library_value(mdp, t, lib, cache) for t in tasks])
                reg = opt - vals
                ok = opt > 1e-9
                row.update(task_max=float(reg.max()), task_rel=float((reg[ok] / opt[ok]).max()),
                           task_mean=float(reg.mean()))
            rows.append(row)
            print(row, flush=True)
    if with_tasks:                                          # corners only (the d extreme policies)
        run = find_run(domain, "cl")
        nv = sum(1 for e in run['log'] if e['gap'] is None)
        lib = prefix_library(mdp, run, nv, sfcache)
        cache = {}
        vals = np.array([library_value(mdp, t, lib, cache) for t in tasks])
        reg = opt - vals
        ok = opt > 1e-9
        L = gpi_loss(mdp, lib, T["W"], T["VS"])
        rows.append(dict(method="corners only", eps=np.nan, library=nv, learned=nv, sweeps=np.nan,
                         queries=0, gap=np.nan, true_max=float(L.max()), true_mean=float(L.mean()),
                         task_max=float(reg.max()), task_rel=float((reg[ok] / opt[ok]).max()),
                         task_mean=float(reg.mean())))
        print(rows[-1], flush=True)
    save(rows, "stops", domain)


if __name__ == "__main__":
    {"stops": cmd_stops}[sys.argv[1]](*sys.argv[2:])
