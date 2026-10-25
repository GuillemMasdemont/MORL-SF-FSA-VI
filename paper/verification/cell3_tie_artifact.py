print("Demonstrating the floating-point tie artifact at k=1 (a fixed, clean, non-jittered example)")
print()

nxt = np.array([[1, 0, 2, 1], [0, 0, 3, 5], [4, 1, 3, 2], [3, 4, 5, 3], [3, 0, 4, 4], [5, 0, 1, 0]])
phi = np.array([
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
])
w = np.array([0.07726823409979573, 0.7834612875183348, 0.13927047838186965])
pi1 = np.array([0, 2, 0, 1, 3, 2])
n, m, d, gamma = 6, 4, 3, GAMMA
P = None

Psi1, _ = full_sf_table(pi1, nxt, P, phi, gamma, n, m, d)
phi_w = phi @ w
allowed = [np.array([pi1[s]]) for s in range(n)]
pi_meta, V_meta = policy_iteration(phi_w, nxt, P, gamma, n, m, allowed=allowed)
Qhat = Psi1 @ w
Q_hybsrc = np.zeros((n, m))
for s in range(n):
    for a in range(m):
        row = trans_row(s, a, nxt, P, n)
        Q_hybsrc[s, a] = phi_w[s, a] + gamma * (row @ V_meta)

with np.printoptions(precision=17, suppress=False):
    print("row 0 of Qhat,     full precision:", Qhat[0])
    print("row 0 of Q_hybsrc, full precision:", Q_hybsrc[0])
print("Same real-valued function (Lemma 1) -- a genuine 3-way tie among actions {0,2,3}")
print("at value 0.62587... -- but rounded via two different linear solves, so argmax flips:")
print("argmax(Qhat)     =", full_greedy(Qhat), " (picks action 0 at s=0)")
print("argmax(Q_hybsrc) =", full_greedy(Q_hybsrc), " (picks action 2 at s=0)")
V_g = evaluate_scalar(full_greedy(Qhat), nxt, P, phi_w, gamma, n)
V_h = evaluate_scalar(full_greedy(Q_hybsrc), nxt, P, phi_w, gamma, n)
print(f"realized value gap from this single flipped tie-break: {np.max(np.abs(V_g - V_h)):.4f}")
print()
print("This is why the stratified sweep below jitters phi by 1e-3: it turns exact ties")
print("into a generic (measure-zero-excluded) strict order, so genuine k=1 violation")
print("counts drop to exactly 0/150 in every configuration.")
