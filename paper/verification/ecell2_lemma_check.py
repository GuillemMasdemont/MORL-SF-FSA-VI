def check_trial(rng, n, m, d, k, gamma, deterministic, n_query_weights=6):
    nxt, P, phi = make_mdp(n, m, d, rng, deterministic)
    base_pis = []
    for _ in range(k):
        w_train = rng.dirichlet(np.ones(d))
        pi_j, _ = policy_iteration(phi @ w_train, nxt, P, gamma, n, m)
        base_pis.append(pi_j)

    worst_lemma, worst_main, best_margin = -np.inf, -np.inf, -np.inf
    for _ in range(n_query_weights):
        w = rng.dirichlet(np.ones(d))
        pi_star, V_star, U, Q_U, pi_meta, V_meta = solve_pi_star(nxt, P, phi, base_pis, w, gamma, n, m, d)

        # Step-1 lemma check: U <= T U
        TU = np.max(Q_U, axis=1)
        worst_lemma = max(worst_lemma, np.max(U - TU))

        # Main theorem check: Q_pi_star >= Qhat_B, everywhere
        Qhat, pi_gpi, V_gpi = qhat_B(nxt, P, phi, base_pis, w, gamma, n, m, d)
        phi_w = phi @ w
        Q_pistar = np.zeros((n, m))
        for s in range(n):
            for a in range(m):
                row = trans_row(s, a, nxt, P, n)
                Q_pistar[s, a] = phi_w[s, a] + gamma * (row @ V_star)
        gap = Q_pistar - Qhat
        worst_main = max(worst_main, np.max(-gap))
        best_margin = max(best_margin, np.max(gap))
    return worst_lemma, worst_main, best_margin


gamma = GAMMA
TOL = 1e-6
print("Stress-testing Theorem 3 (envelope GPI): Step-1 lemma (U <= TU) and the main")
print("claim (Q_pi_star >= Qhat_B, every (s,a)), on random MDPs, deterministic and")
print("stochastic, 500 trials each, 6 query weights per trial.\n")
for deterministic in [True, False]:
    n_trials = 500
    worst_lemma_all, worst_main_all, best_margin_all = -np.inf, -np.inf, -np.inf
    n_lemma_viol, n_main_viol = 0, 0
    rng_master = np.random.default_rng(100 if deterministic else 101)
    for trial in range(n_trials):
        n = int(rng_master.integers(3, 10))
        m = int(rng_master.integers(2, 6))
        d = int(rng_master.integers(2, 5))
        k = int(rng_master.integers(1, min(5, m)))
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        wl, wm, bm = check_trial(rng, n, m, d, k, gamma, deterministic)
        worst_lemma_all = max(worst_lemma_all, wl)
        worst_main_all = max(worst_main_all, wm)
        best_margin_all = max(best_margin_all, bm)
        if wl > TOL:
            n_lemma_viol += 1
        if wm > TOL:
            n_main_viol += 1
    tag = "deterministic" if deterministic else "stochastic"
    print(f"[{tag}] Step-1 lemma (U<=TU): {n_lemma_viol}/{n_trials} violations, worst={worst_lemma_all:.8f}")
    print(f"[{tag}] Theorem (Q_pi*>=Qhat_B): {n_main_viol}/{n_trials} violations, worst={worst_main_all:.8f}, "
          f"largest positive margin found={best_margin_all:.4f}")
