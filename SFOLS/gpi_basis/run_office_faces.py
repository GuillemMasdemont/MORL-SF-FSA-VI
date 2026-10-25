"""Slippery Office on the three symbol edges {C1,C2}, {M1,M2}, {O1,O2} (all weights an out-degree-1 FSA task
can produce): anytime GPI loss and SF-FSA-VI task regret of every basis-construction algorithm.

usage: PYTHONPATH=SFOLS python -m gpi_basis.run_office_faces <p> <variant>
"""
import os
import pickle
import sys
import time

import numpy as np

from .algos_faces import run_ols_multi, run_targeted_multi
from .core import PolicyCache
from .domains import slippery_office
from .fsa import gpi_value, optimal_value, random_tasks
from .segment import Segment, true_loss, vstar_on_grid

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")

VARIANTS = {
    "sfols_full":    lambda m, s: run_ols_multi(m, s, "full", priority="envelope"),
    "sfols_cog":     lambda m, s: run_ols_multi(m, s, "cog", priority="envelope"),
    "gpiols_cog":    lambda m, s: run_ols_multi(m, s, "cog", priority="gpi"),
    "gpils_full":    lambda m, s: run_ols_multi(m, s, "full", priority="gpils", max_queries=400),
    "targeted_cog":  lambda m, s: run_targeted_multi(m, s, "cog", eps=1e-6, select="chord", max_queries=300),
    "targeted_bound_cog": lambda m, s: run_targeted_multi(m, s, "cog", eps=1e-6, select="bound", max_queries=300),
}


def symbol_edges(mdp):
    o = mdp.objects
    E = np.eye(mdp.D)
    return [Segment(E[o.index(a)], E[o.index(b)]) for a, b in (("C1", "C2"), ("M1", "M2"), ("O1", "O2"))]


def main(p, variant):
    mdp = slippery_office(p)
    segs = symbol_edges(mdp)
    xg = np.linspace(0, 1, 401)
    f = os.path.join(OUT, f"vstar_office_edges_p{p}.npz")
    if os.path.exists(f):
        VS = list(np.load(f)["V"])
    else:
        VS = [vstar_on_grid(mdp, s, xg) for s in segs]
        np.savez_compressed(f, V=np.stack(VS))
    rng = np.random.default_rng(0)
    tasks = random_tasks(40, rng, max_states=4, max_out=1)
    ft = os.path.join(OUT, f"tasks_office_deg1_p{p}.pkl")
    if os.path.exists(ft):
        opt = pickle.load(open(ft, "rb"))
    else:
        opt = [optimal_value(mdp, t)[0] for t in tasks]
        pickle.dump(opt, open(ft, "wb"))

    t0 = time.perf_counter()
    G = VARIANTS[variant](mdp, segs)
    t_run = time.perf_counter() - t0
    print(f"p={p} {variant}: {G.c} pool {len(G.pool)} in {t_run:.0f}s", flush=True)

    cache = PolicyCache(mdp, maxsize=3000)
    curve = []
    n = len(G.pool)
    ks = sorted(set(range(mdp.D, min(n, 60) + 1)) | set(range(60, n + 1, 5)) | {n})
    for k in ks:
        losses = [true_loss(G.pool, s, cache, xg, V, upto=k)[0] for s, V in zip(segs, VS)]
        lm = np.concatenate(losses)
        entry = dict(pool=k, max_rel=float(lm.max()), mean_rel=float(lm.mean()),
                     **{kk: G.log[k - 1][kk] for kk in ("full", "meta_ok", "meta_fail", "sweeps_full",
                                                         "sweeps_meta", "lin_solves", "time")})
        if k in (mdp.D, mdp.D + 2, mdp.D + 5, mdp.D + 10, mdp.D + 20, mdp.D + 40, n) or k % 50 == 0:
            vals = np.array([gpi_value(mdp, t, G.pool, upto=k) for t in tasks])
            entry["task_regret_max"] = float(np.max(np.array(opt) - vals))
            entry["task_regret_mean"] = float(np.mean(np.array(opt) - vals))
        curve.append(entry)
    res = dict(p=p, variant=variant, counters=G.c, log=G.log, curve=curve, t_run=t_run,
               cert_log=getattr(G, "cert_log", None), kinds=G.pool.kinds,
               weights=[w.tolist() for w in G.pool.weights],
               actions=np.stack([r.actions for r in G.pool.records]).astype(np.int16))
    with open(os.path.join(OUT, f"office_edges_p{p}_{variant}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    print(f"p={p} {variant}: done; final max loss {curve[-1]['max_rel']:.2e}, "
          f"total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
