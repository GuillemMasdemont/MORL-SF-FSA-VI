import numpy as np
from ecell1_solvers import *

rng_master = np.random.default_rng(555)
gamma = GAMMA
worst_violation = -np.inf  # want V_meta - max_j V_own <= 0 to NEVER happen (i.e. always >= 0)
n_trials = 500
n_viol = 0

for trial in range(n_trials):
    n = int(rng_master.integers(3, 10))
    m = int(rng_master.integers(2, 6))
    d = int(rng_master.integers(2, 5))
    k = int(rng_master.integers(1, min(5, m)))
    rng = np.random.default_rng(rng_master.integers(0, 2**31))
    deterministic = bool(rng_master.integers(0, 2))
    nxt, P, phi = make_mdp(n, m, d, rng, deterministic)

    base_pis = []
    for _ in range(k):
        w_train = rng.dirichlet(np.ones(d))
        pi_j, _ = policy_iteration(phi @ w_train, nxt, P, gamma, n, m)
        base_pis.append(pi_j)

    w = rng.dirichlet(np.ones(d))
    phi_w = phi @ w
    V_own = [evaluate_scalar(pi_j, nxt, P, phi_w, gamma, n) for pi_j in base_pis]
    V_own_max = np.max(np.stack(V_own, axis=0), axis=0)

    allowed = [np.unique([base_pis[j][s] for j in range(k)]) for s in range(n)]
    pi_meta, V_meta = policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=allowed)

    gap = V_meta - V_own_max  # should ALWAYS be >= 0 if V_meta dominates every base policy's own value
    worst_violation = max(worst_violation, np.max(-gap))
    if np.max(-gap) > 1e-6:
        n_viol += 1
        print(f"trial {trial}: k={k} VIOLATION found, worst gap {np.max(-gap):.6f}")

print(f"\n{n_trials} trials: violations of 'V_meta >= max_j V_own' : {n_viol}, worst violation (should be <=0): {worst_violation:.10f}")
