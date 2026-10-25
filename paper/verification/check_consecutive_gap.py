import numpy as np
from ecell1_solvers import *

rng_master = np.random.default_rng(31415)
gamma = GAMMA
n_trials = 20
n_nonmonotone_trials = 0
worst_increase_ratio = -np.inf

for trial in range(n_trials):
    n = int(rng_master.integers(4, 9))
    m = int(rng_master.integers(2, 5))
    d = int(rng_master.integers(2, 4))
    k = int(rng_master.integers(2, min(4, m) + 1))
    rng = np.random.default_rng(rng_master.integers(0, 2**31))
    deterministic = bool(rng_master.integers(0, 2))
    nxt, P, phi = make_mdp(n, m, d, rng, deterministic)

    base_pis = []
    for _ in range(k):
        w_train = rng.dirichlet(np.ones(d))
        pi_j, _ = policy_iteration(phi @ w_train, nxt, P, gamma, n, m)
        base_pis.append(pi_j)
    base_pis_arr = np.array(base_pis)

    w = rng.dirichlet(np.ones(d))

    # random (non-zero) starting point, to stress-test more than the "start at 0" case
    Psi_sj = rng.normal(scale=2.0, size=(n, k, d))

    diffs = []
    for it in range(200):
        vals = Psi_sj @ w
        j_star = np.argmax(vals, axis=1)
        boot = Psi_sj[np.arange(n), j_star, :]

        new_Psi_sj = np.zeros_like(Psi_sj)
        for j in range(k):
            a_j = base_pis_arr[j]
            for s in range(n):
                row = trans_row(s, a_j[s], nxt, P, n)
                new_Psi_sj[s, j] = phi[s, a_j[s]] + gamma * (row @ boot)

        diff = np.max(np.abs(new_Psi_sj - Psi_sj))
        diffs.append(diff)
        Psi_sj = new_Psi_sj

    diffs = np.array(diffs)
    ratios = diffs[1:] / np.maximum(diffs[:-1], 1e-15)
    n_violations = np.sum(ratios > 1.0 + 1e-9)  # step where gap INCREASED vs previous step
    worst_ratio = np.max(ratios) if len(ratios) else np.nan
    if n_violations > 0:
        n_nonmonotone_trials += 1
    worst_increase_ratio = max(worst_increase_ratio, worst_ratio)
    print(f"trial {trial}: n={n} m={m} d={d} k={k} det={deterministic}  "
          f"steps where ||Psi_n+1-Psi_n|| increased: {n_violations}/199  worst ratio={worst_ratio:.3f}")

print(f"\ntrials with at least one non-monotone (increasing) step: {n_nonmonotone_trials}/{n_trials}")
print(f"worst single-step increase ratio seen: {worst_increase_ratio:.3f}")
