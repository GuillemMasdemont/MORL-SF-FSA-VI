"""Demand-driven growth on double slit: a stream of random query weights (each = one 'task').

At each weight: if GPI over the pool is certified optimal there (Bellman gap of the GPI policy <= tol),
nothing is learned; otherwise Certify-or-Grow (metapolicy, else full solve). Also records whether GPI is
eps-optimal at the start (true loss) BEFORE any growth, i.e. the zero-shot quality of the current pool.

usage: PYTHONPATH=SFOLS python -m gpi_basis.run_ds_stream
"""
import os
import pickle
import time

import numpy as np

from .algosnd import GrowthND
from .domains import double_slit

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")


def main(n=300, tol=1e-9, seed=3):
    mdp = double_slit()
    rng = np.random.default_rng(seed)
    G = GrowthND(mdp)
    for w in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        G.grow(w, "full")
    rows = []
    t0 = time.perf_counter()
    for i in range(n):
        x = rng.random()
        w = np.array([x, 1 - x])
        rec = G.cache.get(G.pool.actions(w))
        gap = rec.gap(w).max()
        vstar = mdp.optimal(w)[1][mdp.start]                       # for measurement only
        loss0 = vstar - rec.psi_s[mdp.start] @ w
        before = dict(G.c)
        if gap > tol:
            G.grow(w, "cog")
        rows.append(dict(x=x, gpi_gap=gap, loss_before=loss0, grew=gap > tol,
                         full=G.c["full"] - before["full"], meta=G.c["meta_ok"] - before["meta_ok"],
                         pool=len(G.pool)))
    res = dict(rows=rows, counters=G.c, time=time.perf_counter() - t0)
    pickle.dump(res, open(os.path.join(OUT, "ds_stream.pkl"), "wb"))
    g = np.array([r["grew"] for r in rows]); l0 = np.array([r["loss_before"] for r in rows])
    for a, b in ((0, 50), (50, 100), (100, 200), (200, 300)):
        print(f"weights {a:3d}-{b:3d}: GPI already certified at {100 * (1 - g[a:b].mean()):.0f}%, "
              f"true loss before growth: max {l0[a:b].max():.1e}, mean {l0[a:b].mean():.1e}")
    print(f"final pool {len(G.pool)}: {G.c}, {res['time']:.0f}s")


if __name__ == "__main__":
    main()
