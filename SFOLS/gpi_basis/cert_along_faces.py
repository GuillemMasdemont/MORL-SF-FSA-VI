"""Our certificate on the Office symbol edges, evaluated along the pool prefixes of a finished run.

usage: PYTHONPATH=SFOLS python -m gpi_basis.cert_along_faces <p> <variant>
Every pool member is optimal at its queried weight (full solve or certified metapolicy), so the queried
weights provide the exact values for the chord bounds.
"""
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

from .algos_faces import certify_all
from .core import Pool, PolicyCache
from .domains import slippery_office
from .run_office_faces import symbol_edges

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")


def main(p, variant):
    mdp = slippery_office(p)
    segs = symbol_edges(mdp)
    f = os.path.join(OUT, f"office_edges_p{p}_{variant}.pkl")
    res = pickle.load(open(f, "rb"))
    acts, weights = res["actions"].astype(int), [np.array(w) for w in res["weights"]]
    cache = PolicyCache(mdp, maxsize=3000)
    G = SimpleNamespace(mdp=mdp, cache=cache, pool=Pool(mdp), known_W=[], known_V=[])
    n = len(acts)
    ks = set(range(mdp.D, min(n, 60) + 1)) | set(range(60, n + 1, 5)) | {n}
    out = []
    for a, w in zip(acts, weights):
        rec = cache.get(a)
        G.pool.add(rec, "?", w)
        G.known_W.append(w); G.known_V.append(rec.psi_s[mdp.relevant] @ w)
        if len(G.pool) in ks:
            cert, _ = certify_all(G, segs, 0.0)
            out.append(dict(pool=len(G.pool), cert=cert))
            print(f"  pool {len(G.pool):4d} cert {cert:.2e}", flush=True)
    res["cert_curve"] = out
    pickle.dump(res, open(f, "wb"))


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
