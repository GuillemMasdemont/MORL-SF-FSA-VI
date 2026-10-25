import time
import zlib


def seed_of(tag):
    return zlib.crc32(tag.encode()) % (2**32)


def run_trial(rng, n, m, d, k, gamma, deterministic, sparse_reward, n_query_weights=8):
    nxt, P, phi = make_mdp(n, m, d, rng, deterministic, sparse_reward)
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


gamma = GAMMA
n_trials_per_cell = 150
k_values = [1, 2, 3, 4]
TOL = 1e-6
results = {}

t0 = time.perf_counter()
for deterministic in [True, False]:
    for sparse_reward in [True, False]:
        for k in k_values:
            tag = f"det={deterministic} sparse={sparse_reward} k={k}"
            max_diff, violations = 0.0, 0
            rng_master = np.random.default_rng(seed_of(tag))
            for _ in range(n_trials_per_cell):
                n = int(rng_master.integers(max(3, k + 2), 9))
                m = int(rng_master.integers(max(3, k + 1), 6))
                d = int(rng_master.integers(2, 4))
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                worst = run_trial(rng, n, m, d, k, gamma, deterministic, sparse_reward)
                max_diff = max(max_diff, worst)
                violations += worst > TOL
            results[tag] = (max_diff, violations, n_trials_per_cell)
            print(f"{tag:35s} max|V_hyb-V_gpi|={max_diff:8.4f}  violations={violations:3d}/{n_trials_per_cell}")
print(f"\n({time.perf_counter()-t0:.0f}s)")
