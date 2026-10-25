import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecell1_solvers import *

rng_master = np.random.default_rng(42)
gamma = GAMMA

for deterministic in [True, False]:
    worst_identity = -np.inf   # should be exactly 0: Qhat_B(s,a) == phi(s,a).w + gamma*max_j E[V_j(s')]
    worst_jensen = -np.inf     # should be >=0: E[max_j V_j] - max_j E[V_j] >= 0 (Jensen)
    n_trials = 300
    for trial in range(n_trials):
        n = int(rng_master.integers(3, 10))
        m = int(rng_master.integers(2, 6))
        d = int(rng_master.integers(2, 5))
        k = int(rng_master.integers(1, min(5, m)))
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        nxt, P, phi = make_mdp(n, m, d, rng, deterministic)
        base_pis = []
        for _ in range(k):
            w_train = rng.dirichlet(np.ones(d))
            pi_j, _ = policy_iteration(phi @ w_train, nxt, P, gamma, n, m)
            base_pis.append(pi_j)
        w = rng.dirichlet(np.ones(d))
        phi_w = phi @ w
        V_own = [evaluate_scalar(pi_j, nxt, P, phi_w, gamma, n) for pi_j in base_pis]
        V_own_stack = np.stack(V_own, axis=0)  # (k, n)

        Qhat, pi_gpi, V_gpi = qhat_B(nxt, P, phi, base_pis, w, gamma, n, m, d)

        for s in range(n):
            for a in range(m):
                row = trans_row(s, a, nxt, P, n)
                # max_j E[V_j(s')] under this (s,a)'s transition row
                max_of_avg = np.max(V_own_stack @ row)
                avg_of_max = row @ np.max(V_own_stack, axis=0)
                rhs_identity = phi_w[s, a] + gamma * max_of_avg
                worst_identity = max(worst_identity, abs(Qhat[s, a] - rhs_identity))
                worst_jensen = max(worst_jensen, -(avg_of_max - max_of_avg))  # want >=0

    tag = "deterministic" if deterministic else "stochastic"
    print(f"[{tag}] |Qhat_B - (phi.w + gamma*max_j E[V_j])| worst = {worst_identity:.10f}  (should be ~0)")
    print(f"[{tag}] Jensen gap E[max_j V_j] - max_j E[V_j] worst-negative = {worst_jensen:.10f}  (should be <=0, i.e. gap>=0)")
