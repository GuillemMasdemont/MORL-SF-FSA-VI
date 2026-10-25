"""Windy two-stage domain: where the corner policies are NOT enough at the task level.

usage: PYTHONPATH=SFOLS python -m gpi_basis.run_windy stream
       PYTHONPATH=SFOLS python -m gpi_basis.run_windy <sfols_full|sfols_cog|targeted_cog|tchord_cog>
'stream': 60 two-stage tasks solved one after another with demand-driven certified SF-FSA-VI (shared pool),
          compared with the paper's planning over the 6 corner policies.
others:   zero-shot construction on the hedging {A, B} edge (exact GPI cells + certificate), true loss of GPI
          at the relevant states (start + exits) on a 401-point grid.
"""
import os
import pickle
import sys
import time

import numpy as np

from .algos1d import run_sfols, run_targeted
from .algosnd import GrowthND
from .core import PolicyCache
from .domains import windy_two_stage
from .fsa import gpi_value, optimal_value, windy_tasks
from .fsa_certified import solve_task_certified
from .segment import Segment, true_loss, vstar_on_grid

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")


def stream(mdp, n=60):
    rng = np.random.default_rng(11)
    tasks = windy_tasks(n, rng)
    opt = [optimal_value(mdp, t)[0] for t in tasks]
    G = GrowthND(mdp)
    for i in range(mdp.D):
        G.grow(np.eye(mdp.D)[i], "full")
    corners = [gpi_value(mdp, t, G.pool, upto=mdp.D) for t in tasks]
    rows = []
    for i, t in enumerate(tasks):
        before = dict(G.c)
        v, st = solve_task_certified(mdp, t, G)
        rows.append(dict(task=i, opt=opt[i], certified=v, corners_estimate=corners[i], grows=st["grows"],
                         new_full=G.c["full"] - before["full"], new_meta=G.c["meta_ok"] - before["meta_ok"],
                         pool=len(G.pool)))
        print(f"task {i:2d} {t}: regret certified {opt[i] - v:.1e}, corners {opt[i] - corners[i]:.1e} "
              f"(rel {(opt[i] - corners[i]) / opt[i]:.1%}); grows {st['grows']} (full {rows[-1]['new_full']}, "
              f"meta {rows[-1]['new_meta']}), pool {len(G.pool)}", flush=True)
    pickle.dump(dict(rows=rows, counters=G.c, tasks=tasks), open(os.path.join(OUT, "windy_stream.pkl"), "wb"))
    print("final", G.c)


VARIANTS = {
    "sfols_full":   lambda m, s: run_sfols(m, s, "full", tol=1e-6),
    "sfols_cog":    lambda m, s: run_sfols(m, s, "cog", tol=1e-6),
    "targeted_cog": lambda m, s: run_targeted(m, s, "cog", eps=1e-6, max_queries=400),
    "tchord_cog":   lambda m, s: run_targeted(m, s, "cog", eps=1e-6, select="chord", max_queries=400),
}


def edge(mdp, variant):
    o = mdp.objects
    seg = Segment(np.eye(mdp.D)[o.index("A")], np.eye(mdp.D)[o.index("B")])
    xg = np.linspace(0, 1, 401)
    f = os.path.join(OUT, "vstar_windy_edgeAB.npz")
    if os.path.exists(f):
        VS = np.load(f)["V"]
    else:
        VS = vstar_on_grid(mdp, seg, xg); np.savez_compressed(f, V=VS)
    t0 = time.perf_counter()
    G = VARIANTS[variant](mdp, seg)
    print(f"{variant}: {G.c} pool {len(G.pool)} in {time.perf_counter() - t0:.0f}s", flush=True)
    cache = PolicyCache(mdp, maxsize=3000)
    n = len(G.pool)
    ks = sorted(set(range(2, min(n, 60) + 1)) | set(range(60, n + 1, 5)) | {n})
    curve = []
    for k in ks:
        lr, _ = true_loss(G.pool, seg, cache, xg, VS, upto=k)
        curve.append(dict(pool=k, max_rel=float(lr.max()), mean_rel=float(lr.mean()),
                          **{kk: G.log[k - 1][kk] for kk in ("full", "meta_ok", "meta_fail", "time")}))
    res = dict(variant=variant, counters=G.c, curve=curve, cert_log=getattr(G, "cert_log", None),
               kinds=G.pool.kinds, weights=[w.tolist() for w in G.pool.weights],
               actions=np.stack([r.actions for r in G.pool.records]).astype(np.int16))
    pickle.dump(res, open(os.path.join(OUT, f"windy_edge_{variant}.pkl"), "wb"))
    print(f"{variant}: done, final max loss {curve[-1]['max_rel']:.2e}", flush=True)


if __name__ == "__main__":
    mdp = windy_two_stage()
    if sys.argv[1] == "stream":
        stream(mdp)
    else:
        edge(mdp, sys.argv[1])
