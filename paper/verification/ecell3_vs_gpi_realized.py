def realized_gap_trial(rng, n, m, d, k, gamma, deterministic, n_query_weights=6):
    nxt, P, phi = make_mdp(n, m, d, rng, deterministic)
    base_pis = []
    for _ in range(k):
        w_train = rng.dirichlet(np.ones(d))
        pi_j, _ = policy_iteration(phi @ w_train, nxt, P, gamma, n, m)
        base_pis.append(pi_j)
    worst = -np.inf
    for _ in range(n_query_weights):
        w = rng.dirichlet(np.ones(d))
        pi_star, V_star, U, Q_U, pi_meta, V_meta = solve_pi_star(nxt, P, phi, base_pis, w, gamma, n, m, d)
        Qhat, pi_gpi, V_gpi = qhat_B(nxt, P, phi, base_pis, w, gamma, n, m, d)
        worst = max(worst, np.max(V_gpi - V_star))  # positive: GPI's REALIZED value beats pi_star's
    return worst


print("Honest check: does GPI's REALIZED value (not just its estimate) ever beat pi_star's?")
print("Theorem 3 only proves Q_pi_star >= Qhat_B (the estimate) -- this is not proven either way.\n")
gamma = GAMMA
for deterministic in [True, False]:
    n_trials = 500
    worst_all = -np.inf
    n_gpi_wins = 0
    rng_master = np.random.default_rng(200 if deterministic else 201)
    for trial in range(n_trials):
        n = int(rng_master.integers(3, 10))
        m = int(rng_master.integers(2, 6))
        d = int(rng_master.integers(2, 5))
        k = int(rng_master.integers(1, min(5, m)))
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        gap = realized_gap_trial(rng, n, m, d, k, gamma, deterministic)
        worst_all = max(worst_all, gap)
        if gap > 1e-6:
            n_gpi_wins += 1
    tag = "deterministic" if deterministic else "stochastic"
    print(f"[{tag}] GPI's realized value beats pi_star's: {n_gpi_wins}/{n_trials} trials, worst gap={worst_all:.6f}")
