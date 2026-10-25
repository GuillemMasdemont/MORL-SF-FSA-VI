import numpy as np

np.set_printoptions(suppress=True, precision=4)

GAMMA = 0.9


def make_mdp(n, m, d, rng, deterministic, sparse_reward, jitter=1e-3):
    """A random finite MDP: n states, m actions everywhere, d objectives.
    deterministic=False draws a genuinely stochastic Dirichlet transition kernel.
    sparse_reward=True mimics a "pickup" domain (each (s,a) triggers at most one
    objective's reward, matching phi's structure in DST/Office/the maze); jitter
    breaks the many exact Q-value ties that a clean sparse/one-hot reward otherwise
    creates, so that any argmax discrepancy found reflects a real inequality rather
    than floating-point noise resolving a genuine tie in opposite directions."""
    if deterministic:
        nxt = rng.integers(0, n, size=(n, m))
        P = None
    else:
        nxt = None
        alpha = 0.3
        P = np.stack([rng.dirichlet(alpha * np.ones(n), size=m) for _ in range(n)], axis=0)

    if sparse_reward:
        phi = np.zeros((n, m, d))
        mask = rng.random((n, m)) < 0.2
        idx = rng.integers(0, d, size=(n, m))
        for s in range(n):
            for a in range(m):
                if mask[s, a]:
                    phi[s, a, idx[s, a]] = 1.0
    else:
        phi = rng.random((n, m, d))
    phi = phi + jitter * rng.random((n, m, d))
    return nxt, P, phi


def trans_row(s, a, nxt, P, n):
    if nxt is not None:
        row = np.zeros(n)
        row[nxt[s, a]] = 1.0
        return row
    return P[s, a, :]


def build_Ppi(pi, nxt, P, n):
    Ppi = np.zeros((n, n))
    for s in range(n):
        Ppi[s, :] = trans_row(s, pi[s], nxt, P, n)
    return Ppi


def evaluate_scalar(pi, nxt, P, phi_w, gamma, n):
    Ppi = build_Ppi(pi, nxt, P, n)
    r_pi = np.array([phi_w[s, pi[s]] for s in range(n)])
    return np.linalg.solve(np.eye(n) - gamma * Ppi, r_pi)


def evaluate_vector(pi, nxt, P, phi, gamma, n, d):
    Ppi = build_Ppi(pi, nxt, P, n)
    r_pi = np.stack([phi[s, pi[s], :] for s in range(n)], axis=0)
    return np.linalg.solve(np.eye(n) - gamma * Ppi, r_pi)


def full_sf_table(pi, nxt, P, phi, gamma, n, m, d):
    """Psi(s,a) for the FIXED policy pi, evaluated at every (s,a): take a once, then
    follow pi's own greedy action from the landing state onward."""
    V_vec = evaluate_vector(pi, nxt, P, phi, gamma, n, d)
    Psi = np.zeros((n, m, d))
    for s in range(n):
        for a in range(m):
            row = trans_row(s, a, nxt, P, n)
            Psi[s, a, :] = phi[s, a, :] + gamma * (row @ V_vec)
    return Psi, V_vec


def policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=None, max_iter=1000):
    """Exact policy iteration, optionally restricted to `allowed[s]` actions per
    state -- this is Definition 2's T_B when `allowed` is the k base policies' own
    actions, and ordinary unrestricted PI when `allowed=None`."""
    if allowed is None:
        allowed = [np.arange(m) for _ in range(n)]
    pi = np.array([allowed[s][0] for s in range(n)])
    for _ in range(max_iter):
        V = evaluate_scalar(pi, nxt, P, phi_w, gamma, n)
        new_pi = pi.copy()
        for s in range(n):
            best_a, best_q = None, -np.inf
            for a in allowed[s]:
                row = trans_row(s, a, nxt, P, n)
                q = phi_w[s, a] + gamma * (row @ V)
                if q > best_q + 1e-12:
                    best_q, best_a = q, a
            new_pi[s] = best_a
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
    return pi, evaluate_scalar(pi, nxt, P, phi_w, gamma, n)


def full_greedy(Qsa):
    return np.argmax(Qsa, axis=1)


def solve_basis_and_compare(nxt, P, phi, base_pis, w, gamma, n, m, d):
    """Given an already-solved basis (list of policies), compute, for query weight w:
    metapolicy (restricted PI), pi_hyb (full-action greedy off V_meta), and
    pi_gpi (full-action greedy off the pool estimate Q-hat). Returns realized scalar
    values V_meta, V_hyb, V_gpi (each length n) plus the three policies."""
    k = len(base_pis)
    base_psis = [full_sf_table(pi_j, nxt, P, phi, gamma, n, m, d)[0] for pi_j in base_pis]
    phi_w = phi @ w

    allowed = [np.unique([base_pis[j][s] for j in range(k)]) for s in range(n)]
    pi_meta, V_meta = policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=allowed)

    Qhat = np.zeros((n, m))
    for s in range(n):
        for a in range(m):
            Qhat[s, a] = max(base_psis[j][s, a, :] @ w for j in range(k))
    pi_gpi = full_greedy(Qhat)
    V_gpi = evaluate_scalar(pi_gpi, nxt, P, phi_w, gamma, n)

    Q_hybsrc = np.zeros((n, m))
    for s in range(n):
        for a in range(m):
            row = trans_row(s, a, nxt, P, n)
            Q_hybsrc[s, a] = phi_w[s, a] + gamma * (row @ V_meta)
    pi_hyb = full_greedy(Q_hybsrc)
    V_hyb = evaluate_scalar(pi_hyb, nxt, P, phi_w, gamma, n)

    return dict(pi_meta=pi_meta, pi_gpi=pi_gpi, pi_hyb=pi_hyb,
                V_meta=V_meta, V_gpi=V_gpi, V_hyb=V_hyb, Qhat=Qhat, Q_hybsrc=Q_hybsrc)
