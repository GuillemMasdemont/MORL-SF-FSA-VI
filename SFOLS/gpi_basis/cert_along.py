"""Our GPI certificate evaluated along the pool prefixes of any finished run (segment version).

usage: PYTHONPATH=SFOLS python -m gpi_basis.cert_along <domain> <run-pickle> [<segment index>]
Adds 'cert_curve' (pool size -> certified max loss at the relevant states, for every weight on the
segment) to the pickle. The known exact values come from the run's own queried weights (all of them are
optimal: full solves, or certified metapolicies).
"""
import pickle
import sys

import numpy as np

from .core import Pool, PolicyCache
from .segment import Segment, certify, gpi_cells


def cert_curve(mdp, seg, actions_list, weights, ks):
    cache = PolicyCache(mdp, maxsize=3000)
    pool = Pool(mdp)
    known_x, known_V = [], []
    out = []
    for i, (act, w) in enumerate(zip(actions_list, weights)):
        rec = cache.get(act)
        pool.add(rec, "?", w)
        # x such that w = x a + (1-x) b on the segment
        x = float(np.dot(np.asarray(w) - seg.b, seg.d) / np.dot(seg.d, seg.d))
        known_x.append(x); known_V.append(rec.psi_s[mdp.relevant] @ np.asarray(w))
        if len(pool) in ks:
            cells = gpi_cells(pool, seg, cache)
            cr = certify(mdp, cells, seg, known_x, known_V, eps=0.0)
            out.append(dict(pool=len(pool), cert=cr["cert"], sup_cert=cr["sup_cert"], cells=len(cells)))
            print(f"  pool {len(pool):4d} cert {cr['cert']:.2e}", flush=True)
    return out


if __name__ == "__main__":
    import os
    from .algos1d import run_gpils, run_sfols
    from .domains import double_slit
    algo = sys.argv[1]
    mdp = double_slit(); seg = Segment([1, 0], [0, 1])
    G = {"sfols_full": lambda: run_sfols(mdp, seg, "full", tol=1e-6),
         "sfols_cog": lambda: run_sfols(mdp, seg, "cog", tol=1e-6),
         "gpils_full": lambda: run_gpils(mdp, seg, "full", max_queries=203)}[algo]()
    ks = set(range(2, 61)) | set(range(60, len(G.pool) + 1, 5)) | {len(G.pool)}
    acts = [r.actions for r in G.pool.records]
    out = cert_curve(mdp, seg, acts, G.pool.weights, ks)
    f = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis", f"ds_{algo}.pkl")
    res = pickle.load(open(f, "rb"))
    res["cert_curve"] = out
    res["actions"] = np.stack(acts).astype(np.int16)
    pickle.dump(res, open(f, "wb"))
