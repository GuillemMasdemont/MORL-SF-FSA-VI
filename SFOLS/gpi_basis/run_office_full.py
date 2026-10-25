"""Slippery Office on the FULL weight simplex (d = 6): anytime GPI loss on held-out weights and SF-FSA-VI
task regret, for SFOLS / Certify-or-Grow / GPI-LS / GPI-aware OLS / an oracle greedy (analysis only).

usage: PYTHONPATH=SFOLS python -m gpi_basis.run_office_full <p> <variant>
"""
import os
import pickle
import sys
import time

import numpy as np

from .algosnd import GrowthND, run_gpils, run_probe, run_sfols, run_targeted
from .domains import slippery_office
from .evaluate import gpi_loss, test_weights, vstar_table
from .fsa import PAPER_TASKS, gpi_value, optimal_value, random_tasks

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")
GROUPS = [[0, 1], [2, 3], [4, 5]]                    # C, O, M in Office.PHI_OBJ_TYPES order
BUDGET = 150


def run_oracle(mdp, how="cog", max_queries=BUDGET, n_cand=300, seed=1):
    """Greedy on the TRUE GPI loss over fresh random candidates (uses V* at candidates: analysis only)."""
    rng = np.random.default_rng(seed)
    G = GrowthND(mdp)
    for i in range(mdp.D):
        G.grow(np.eye(mdp.D)[i], "full")
    rel = mdp.relevant
    while G.c["queries"] < max_queries:
        cand, _ = test_weights(mdp.D, GROUPS, rng, n_random=n_cand, n_face=20)
        V = vstar_table(mdp, cand)
        loss = gpi_loss(mdp, G.pool, cand, V)
        G.grow(cand[int(np.argmax(loss))], how)
    return G


VARIANTS = {
    "sfols_full":  lambda m: run_sfols(m, "full", max_queries=BUDGET),
    "sfols_cog":   lambda m: run_sfols(m, "cog", max_queries=BUDGET),
    "gpiols_cog":  lambda m: run_sfols(m, "cog", max_queries=BUDGET, priority="gpi"),
    "gpils_full":  lambda m: run_gpils(m, "full", max_queries=BUDGET),
    "oracle_cog":  lambda m: run_oracle(m, "cog"),
    "targeted_cog": lambda m: run_targeted(m, "cog", eps=1e-6, max_queries=BUDGET, groups=GROUPS),
    "probe":       lambda m: run_probe(m, max_queries=BUDGET, groups=GROUPS),
}


def main(p, variant):
    mdp = slippery_office(p)
    rng = np.random.default_rng(0)
    f = os.path.join(OUT, f"vstar_office_full_p{p}.npz")
    if os.path.exists(f):
        z = np.load(f); W, lab, VS = z["W"], z["lab"], z["V"]
    else:
        W, lab = test_weights(mdp.D, GROUPS, rng)
        VS = vstar_table(mdp, W)
        np.savez_compressed(f, W=W, lab=lab, V=VS)
    trng = np.random.default_rng(1)
    tasks = list(PAPER_TASKS.values()) + random_tasks(30, trng, max_out=2) + random_tasks(10, trng, max_out=3)
    ft = os.path.join(OUT, f"tasks_office_full_p{p}.pkl")
    if os.path.exists(ft):
        opt = pickle.load(open(ft, "rb"))
    else:
        opt = [optimal_value(mdp, t)[0] for t in tasks]
        pickle.dump(opt, open(ft, "wb"))
    opt = np.array(opt)

    t0 = time.perf_counter()
    G = VARIANTS[variant](mdp)
    t_run = time.perf_counter() - t0
    print(f"p={p} {variant}: {G.c} pool {len(G.pool)} in {t_run:.0f}s", flush=True)

    curve = []
    n = len(G.pool)
    ks = sorted(set(range(mdp.D, min(n, 40) + 1)) | set(range(40, n + 1, 5)) | {n})
    for k in ks:
        L = gpi_loss(mdp, G.pool, W, VS, upto=k)
        entry = dict(pool=k, **{f"max_{c}": float(L[lab == c].max()) for c in np.unique(lab)},
                     **{f"mean_{c}": float(L[lab == c].mean()) for c in np.unique(lab)},
                     **{kk: G.log[k - 1][kk] for kk in ("full", "meta_ok", "meta_fail", "sweeps_full",
                                                         "sweeps_meta", "lin_solves", "time")})
        if k in (6, 8, 11, 16, 26, 46, 76, 106, 150) or k == n:
            vals = np.array([gpi_value(mdp, t, G.pool, upto=k) for t in tasks])
            reg = opt - vals
            entry.update(task_regret_max=float(reg.max()), task_regret_mean=float(reg.mean()),
                         task_regret_paper=[float(r) for r in reg[:3]])
        curve.append(entry)
    res = dict(p=p, variant=variant, counters=G.c, log=G.log, curve=curve, t_run=t_run, kinds=G.pool.kinds,
               weights=[w.tolist() for w in G.pool.weights],
               actions=np.stack([r.actions for r in G.pool.records]).astype(np.int16))
    with open(os.path.join(OUT, f"office_full_p{p}_{variant}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    print(f"p={p} {variant}: done; final max loss (simplex) {curve[-1]['max_simplex']:.2e}, "
          f"total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
