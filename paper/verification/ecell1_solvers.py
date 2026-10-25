import numpy as np

GAMMA = 0.9


def make_mdp(n, m, d, rng, deterministic):
    if deterministic:
        nxt = rng.integers(0, n, size=(n, m))
        P = None
    else:
        alpha = 0.3
        P = np.stack([rng.dirichlet(alpha * np.ones(n), size=m) for _ in range(n)], axis=0)
        nxt = None
    phi = rng.random((n, m, d)) + 1e-3 * rng.random((n, m, d))
    return nxt, P, phi


def trans_row(s, a, nxt, P, n):
    if nxt is not None:
        row = np.zeros(n)
        row[nxt[s, a]] = 1.0
        return row
    return P[s, a, :]


def evaluate_scalar(pi, nxt, P, phi_w, gamma, n):
    Ppi = np.array([trans_row(s, pi[s], nxt, P, n) for s in range(n)])
    r_pi = np.array([phi_w[s, pi[s]] for s in range(n)])
    return np.linalg.solve(np.eye(n) - gamma * Ppi, r_pi)


def evaluate_vector(pi, nxt, P, phi, gamma, n, d):
    Ppi = np.array([trans_row(s, pi[s], nxt, P, n) for s in range(n)])
    r_pi = np.stack([phi[s, pi[s], :] for s in range(n)], axis=0)
    return np.linalg.solve(np.eye(n) - gamma * Ppi, r_pi)


def full_sf_table(pi, nxt, P, phi, gamma, n, m, d):
    """Only used here to CHECK Qhat_B -- pi_star itself never needs this."""
    V_vec = evaluate_vector(pi, nxt, P, phi, gamma, n, d)
    Psi = np.zeros((n, m, d))
    for s in range(n):
        for a in range(m):
            row = trans_row(s, a, nxt, P, n)
            Psi[s, a, :] = phi[s, a, :] + gamma * (row @ V_vec)
    return Psi


def policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=None):
    if allowed is None:
        allowed = [np.arange(m) for _ in range(n)]
    pi = np.array([allowed[s][0] for s in range(n)])
    for _ in range(1000):
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


def solve_pi_star(nxt, P, phi, base_pis, w, gamma, n, m, d):
    """Everything pi_star actually needs: each base policy's OWN scalar value
    (from its cached own-action SF vector, dotted with the query weight -- NOT a full
    per-action SF table), plus the metapolicy's value from cheap restricted PI."""
    k = len(base_pis)
    phi_w = phi @ w
    V_own = [evaluate_scalar(pi_j, nxt, P, phi_w, gamma, n) for pi_j in base_pis]

    allowed = [np.unique([base_pis[j][s] for j in range(k)]) for s in range(n)]
    pi_meta, V_meta = policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=allowed)

    U = np.maximum(V_meta, np.max(np.stack(V_own, axis=0), axis=0))

    Q_U = np.zeros((n, m))
    for s in range(n):
        for a in range(m):
            row = trans_row(s, a, nxt, P, n)
            Q_U[s, a] = phi_w[s, a] + gamma * (row @ U)
    pi_star = np.argmax(Q_U, axis=1)
    V_star = evaluate_scalar(pi_star, nxt, P, phi_w, gamma, n)
    return pi_star, V_star, U, Q_U, pi_meta, V_meta


def qhat_B(nxt, P, phi, base_pis, w, gamma, n, m, d):
    """The full-pool GPI estimate -- needs every base policy's FULL per-action SF
    table. Only computed here to check pi_star against it, never to compute pi_star."""
    k = len(base_pis)
    base_psis = [full_sf_table(pi_j, nxt, P, phi, gamma, n, m, d) for pi_j in base_pis]
    Qhat = np.zeros((n, m))
    for s in range(n):
        for a in range(m):
            Qhat[s, a] = max(base_psis[j][s, a, :] @ w for j in range(k))
    pi_gpi = np.argmax(Qhat, axis=1)
    phi_w = phi @ w
    V_gpi = evaluate_scalar(pi_gpi, nxt, P, phi_w, gamma, n)
    return Qhat, pi_gpi, V_gpi
