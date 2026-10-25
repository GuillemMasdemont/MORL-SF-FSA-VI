"""Re-run notebook 14's exact-solve OLS on deterministic Office (eps = 0.05, 7 start states) with EXACT corner
enumeration, to check whether float corner enumeration changed its 63-query epsilon-CCS."""
import time
import numpy as np
from .domains import slippery_office
from .algosnd import run_sfols
from .evaluate import vstar_table

mdp = slippery_office(0.0)                    # p = 0: the deterministic Office of notebook 14
for exact in (False, True):
    t = time.time()
    G = run_sfols(mdp, "full", tol=0.05, exact=exact)
    P = np.array([r.psi_s[mdp.relevant].mean(0) for r in G.pool.records])
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(mdp.D), 300)
    V = vstar_table(mdp, W)[:, mdp.relevant].mean(1)
    gap = V - (W @ P.T).max(1)
    print(f"exact corners={exact}: {G.c['queries']} queries, CCS {len(G.ccs)}, true gap on 300 weights max "
          f"{gap.max():.4f} (eps 0.05), {time.time() - t:.0f}s", flush=True)
