import zlib


def seed_of(tag):
    return zlib.crc32(tag.encode()) % (2**32)


def make_terminating_mdp(n_live, m, d, rng, deterministic, jitter=1e-3):
    """n_live live states plus one absorbing zero-reward sink, mirroring the
    "terminate on reward pickup" structure of DST/Office/the maze -- unlike
    make_mdp's infinite-horizon random walk, a reward event here ends the episode."""
    n = n_live + 1
    sink = n_live
    if deterministic:
        nxt = rng.integers(0, n_live, size=(n, m))
        nxt[sink, :] = sink
        P = None
    else:
        alpha = 0.3
        P = np.zeros((n, m, n))
        for s in range(n_live):
            for a in range(m):
                P[s, a, :n_live] = rng.dirichlet(alpha * np.ones(n_live))
        P[sink, :, sink] = 1.0
        nxt = None

    phi = np.zeros((n, m, d))
    mask = rng.random((n_live, m)) < 0.25
    idx = rng.integers(0, d, size=(n_live, m))
    for s in range(n_live):
        for a in range(m):
            if mask[s, a]:
                phi[s, a, idx[s, a]] = 1.0
                if deterministic:
                    nxt[s, a] = sink
                else:
                    P[s, a, :] = 0.0
                    P[s, a, sink] = 1.0
    phi[:n_live] += jitter * rng.random((n_live, m, d))
    return nxt, P, phi


def run_trial_terminating(rng, n_live, m, d, k, gamma, deterministic, n_query_weights=8):
    nxt, P, phi = make_terminating_mdp(n_live, m, d, rng, deterministic)
    n = n_live + 1
    base_pis = []
    for _ in range(k):
        w_train = rng.dirichlet(np.ones(d))
        pi_j, _ = policy_iteration(phi @ w_train, nxt, P, gamma, n, m)
        base_pis.append(pi_j)
    worst = 0.0
    for _ in range(n_query_weights):
        w = rng.dirichlet(np.ones(d))
        out = solve_basis_and_compare(nxt, P, phi, base_pis, w, gamma, n, m, d)
        worst = max(worst, np.max(np.abs(out["V_hyb"] - out["V_gpi"])))
    return worst


print("Hypothesis: DST/Office/the maze's zero counterexamples come from their absorbing,")
print("terminate-on-pickup structure, not a general k>=2 guarantee. Test: add an explicit")
print("absorbing sink on every reward event and re-run the sweep.")
print()
gamma = GAMMA
n_trials = 150
for deterministic in [True, False]:
    for k in [1, 2, 3, 4]:
        tag = f"TERMINATING det={deterministic} k={k}"
        max_diff, violations = 0.0, 0
        rng_master = np.random.default_rng(seed_of(tag))
        for _ in range(n_trials):
            n_live = int(rng_master.integers(max(3, k + 2), 8))
            m = int(rng_master.integers(max(3, k + 1), 6))
            d = int(rng_master.integers(2, 4))
            rng = np.random.default_rng(rng_master.integers(0, 2**31))
            worst = run_trial_terminating(rng, n_live, m, d, k, gamma, deterministic)
            max_diff = max(max_diff, worst)
            violations += worst > 1e-6
        print(f"{tag:25s} max|V_hyb-V_gpi|={max_diff:8.4f}  violations={violations:3d}/{n_trials}")
