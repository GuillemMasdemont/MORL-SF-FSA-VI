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

rng_master = np.random.default_rng(777)
gamma = GAMMA
worst_sf_err = -np.inf
n_policy_mismatch = 0
n_trials = 10

for trial in range(n_trials):
    n = int(rng_master.integers(4, 8))
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

    # ground truth via exact restricted PI
    phi_w = phi @ w
    allowed = [np.unique([base_pis[j][s] for j in range(k)]) for s in range(n)]
    meta_policy_true, V_meta_true = policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=allowed)
    Psi_meta_true = full_sf_table_actions(meta_policy_true, nxt, P, phi, gamma, n, m, d)

    # combined Q-learning: single table Psihat(s,a) for ALL (s,a); bootstrap restricted to
    # the k base actions' own values at s' (this defines/learns pi_meta); behavior policy
    # explores ALL actions broadly (exploring starts, uniform random action).
    Psihat = np.zeros((n, m, d))
    visits = np.zeros((n, m))
    T = 500_000
    EP_LEN = 20
    s = rng.integers(0, n)
    for t in range(T):
        if t % EP_LEN == 0:
            s = rng.integers(0, n)
        a = rng.integers(0, m)
        if nxt is not None:
            ns = nxt[s, a]
        else:
            ns = rng.choice(n, p=P[s, a])
        r = phi[s, a]
        visits[s, a] += 1
        alpha = 1.0 / (1 + 0.02 * visits[s, a])

        # meta-greedy action at s', restricted to the k base policies' own actions there
        cand_actions = base_pis_arr[:, ns]              # (k,)
        cand_vals = Psihat[ns, cand_actions] @ w         # (k,)
        j_star = np.argmax(cand_vals)
        a_star = cand_actions[j_star]

        target = r + gamma * Psihat[ns, a_star]
        Psihat[s, a] += alpha * (target - Psihat[s, a])
        s = ns

    # derive learned metapolicy from converged table
    meta_policy_learned = np.zeros(n, dtype=int)
    for st in range(n):
        cand_actions = base_pis_arr[:, st]
        cand_vals = Psihat[st, cand_actions] @ w
        meta_policy_learned[st] = cand_actions[np.argmax(cand_vals)]

    policy_mismatch = np.sum(meta_policy_learned != meta_policy_true)
    sf_err = np.max(np.abs(Psihat - Psi_meta_true))
    worst_sf_err = max(worst_sf_err, sf_err)
    n_policy_mismatch += policy_mismatch
    print(f"trial {trial}: n={n} m={m} d={d} k={k} det={deterministic}  "
          f"policy mismatches={policy_mismatch}/{n}  max SF err={sf_err:.4f}  min visits={visits.min():.0f}")

print(f"\ntotal policy mismatches over all trials: {n_policy_mismatch}")
print(f"worst SF error over all trials: {worst_sf_err:.4f}")
