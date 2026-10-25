import numpy as np
from ecell1_solvers import *

def full_sf_table_actions(pi_actions, nxt, P, phi, gamma, n, m, d):
    V_vec = evaluate_vector(pi_actions, nxt, P, phi, gamma, n, d)
    Psi = np.zeros((n, m, d))
    for s in range(n):
        for a in range(m):
            row = trans_row(s, a, nxt, P, n)
            Psi[s, a, :] = phi[s, a, :] + gamma * (row @ V_vec)
    return Psi

rng_master = np.random.default_rng(9001)
gamma = GAMMA
n_trials = 15
worst_sf_err = -np.inf
n_policy_mismatch = 0
max_stabilize_iter = 0

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
    base_pis_arr = np.array(base_pis)  # (k, n)

    w = rng.dirichlet(np.ones(d))
    phi_w = phi @ w
    allowed = [np.unique([base_pis[j][s] for j in range(k)]) for s in range(n)]
    meta_policy_true, V_meta_true = policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=allowed)
    Psi_meta_true = full_sf_table_actions(meta_policy_true, nxt, P, phi, gamma, n, m, d)

    # EXACT synchronous vector value iteration on Psi(s,j), j=1..k (own-action indices only)
    Psi_sj = np.zeros((n, k, d))  # Psi(s,j)
    prev_greedy = None
    stabilize_iter = None
    N_ITERS = 2000
    for it in range(N_ITERS):
        # greedy meta-index at every state under current estimate
        vals = Psi_sj @ w  # (n, k)
        j_star = np.argmax(vals, axis=1)  # (n,)
        if prev_greedy is not None and stabilize_iter is None and np.array_equal(j_star, prev_greedy):
            stabilize_iter = it
        prev_greedy = j_star

        # bootstrap vector at the greedy index, per state
        boot = Psi_sj[np.arange(n), j_star, :]  # (n, d)

        new_Psi_sj = np.zeros_like(Psi_sj)
        for j in range(k):
            a_j = base_pis_arr[j]  # (n,) actions
            for s in range(n):
                row = trans_row(s, a_j[s], nxt, P, n)
                new_Psi_sj[s, j] = phi[s, a_j[s]] + gamma * (row @ boot)
        Psi_sj = new_Psi_sj

    # final greedy policy + full-table extension
    vals = Psi_sj @ w
    j_star_final = np.argmax(vals, axis=1)
    meta_policy_learned = base_pis_arr[j_star_final, np.arange(n)]
    boot_final = Psi_sj[np.arange(n), j_star_final, :]

    Psi_full = np.zeros((n, m, d))
    for s in range(n):
        for a in range(m):
            row = trans_row(s, a, nxt, P, n)
            Psi_full[s, a] = phi[s, a] + gamma * (row @ boot_final)

    mismatch = np.sum(meta_policy_learned != meta_policy_true)
    sf_err = np.max(np.abs(Psi_full - Psi_meta_true))
    worst_sf_err = max(worst_sf_err, sf_err)
    n_policy_mismatch += mismatch
    if stabilize_iter is not None:
        max_stabilize_iter = max(max_stabilize_iter, stabilize_iter)
    print(f"trial {trial}: n={n} m={m} d={d} k={k} det={deterministic}  "
          f"policy mismatches={mismatch}/{n}  max SF err={sf_err:.6f}  "
          f"greedy stabilized at iter={stabilize_iter}")

print(f"\ntotal policy mismatches: {n_policy_mismatch}")
print(f"worst SF error: {worst_sf_err:.6f}")
print(f"latest stabilization iter seen: {max_stabilize_iter}")
