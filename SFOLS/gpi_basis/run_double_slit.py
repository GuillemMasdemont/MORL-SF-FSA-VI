"""Double slit (wind +-3, gamma 0.9): anytime GPI loss of every basis-construction algorithm.

usage: PYTHONPATH=SFOLS python -m gpi_basis.run_double_slit <variant>
Writes SFOLS/results/gpi_basis/ds_<variant>.pkl with the growth log, the certificate log (targeted
variants) and the TRUE loss of GPI over every prefix of the pool on a 2001-point weight grid.
"""
import os
import pickle
import sys
import time

import numpy as np

from .algos1d import run_gpils, run_sfols, run_targeted
from .core import PolicyCache
from .domains import double_slit
from .segment import Segment, true_loss, vstar_on_grid

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")

VARIANTS = {
    "sfols_full":      lambda m, s: run_sfols(m, s, "full", tol=1e-6),
    "sfols_cog":       lambda m, s: run_sfols(m, s, "cog", tol=1e-6),
    "gpils_full":      lambda m, s: run_gpils(m, s, "full", max_queries=203),
    "targeted_full":   lambda m, s: run_targeted(m, s, "full", eps=1e-6),
    "targeted_cog":    lambda m, s: run_targeted(m, s, "cog", eps=1e-6),
    "targeted_sup":    lambda m, s: run_targeted(m, s, "full", eps=1e-6, use_reach=False, use_chord=False,
                                                max_queries=150),
    "targeted_cog_b4": lambda m, s: run_targeted(m, s, "cog", eps=1e-6, batch=4),
    "tchord_full":     lambda m, s: run_targeted(m, s, "full", eps=1e-6, select="chord"),
    "tchord_cog":      lambda m, s: run_targeted(m, s, "cog", eps=1e-6, select="chord"),
    "talt_full":       lambda m, s: run_targeted(m, s, "full", eps=1e-6, select="alt"),
}


def main(variant):
    mdp = double_slit()
    seg = Segment([1, 0], [0, 1])
    f = os.path.join(OUT, "vstar_double_slit.npz")
    if os.path.exists(f):
        z = np.load(f); xg, VS = z["x"], z["V"]
    else:
        xg = np.linspace(0, 1, 2001); VS = vstar_on_grid(mdp, seg, xg)
        np.savez_compressed(f, x=xg, V=VS)
    t0 = time.perf_counter()
    G = VARIANTS[variant](mdp, seg)
    t_run = time.perf_counter() - t0
    print(f"{variant}: {G.c} pool {len(G.pool)} in {t_run:.0f}s", flush=True)
    cache = PolicyCache(mdp, maxsize=3000)
    curve = []
    for k in range(2, len(G.pool) + 1):
        lr, la = true_loss(G.pool, seg, cache, xg, VS, upto=k)
        curve.append(dict(pool=k, max_start=float(lr.max()), mean_start=float(lr.mean()),
                          max_all=float(la.max()), **{kk: G.log[k - 1][kk] for kk in
                                                       ("full", "meta_ok", "meta_fail", "sweeps_full",
                                                        "sweeps_meta", "lin_solves", "time")}))
    res = dict(variant=variant, counters=G.c, log=G.log, curve=curve, t_run=t_run,
               cert_log=getattr(G, "cert_log", None), kinds=G.pool.kinds,
               weights=[w.tolist() for w in G.pool.weights],
               actions=np.stack([r.actions for r in G.pool.records]).astype(np.int16))
    with open(os.path.join(OUT, f"ds_{variant}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    last = curve[-1]
    print(f"{variant}: done; final true max loss at start {last['max_start']:.2e}, "
          f"total time {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
