print("Minimal counterexample #2: pi_gpi_B STRICTLY BEATS pi_hyb (n=3, m=4, d=2, k=2)")
print()

nxt = np.array([[2, 2, 2, 1], [1, 1, 0, 1], [1, 2, 0, 2]])
phi = np.array([
    [[0.31898189, 0.12647578], [0.49033860, 0.74536444], [0.01150748, 0.13755603], [0.63719606, 0.58878224]],
    [[0.02694831, 0.74838586], [0.40844423, 0.88605940], [0.37537800, 0.25121873], [0.82470631, 0.03710746]],
    [[0.74559804, 0.17799423], [0.61045478, 0.06783933], [0.85616052, 0.82155972], [0.74909825, 0.50135312]],
])
w = np.array([0.68023705, 0.31976295])
base_pis = [np.array([3, 3, 0]), np.array([3, 1, 2])]  # pi_1, pi_2
n, m, d, gamma = 3, 4, 2, GAMMA

out = solve_basis_and_compare(nxt, None, phi, base_pis, w, gamma, n, m, d)
print("B(0)={3}, B(1)={1,3}, B(2)={0,2}")
print("pi_meta =", out["pi_meta"], "  pi_gpi =", out["pi_gpi"], "  pi_hyb =", out["pi_hyb"])
print("V_meta  =", out["V_meta"])
print("V_gpi   =", out["V_gpi"])
print("V_hyb   =", out["V_hyb"])
print()
print(f"At s=0: V_gpi(0)={out['V_gpi'][0]:.4f} > V_hyb(0)={out['V_hyb'][0]:.4f}, gap ="
      f" {out['V_gpi'][0]-out['V_hyb'][0]:.4f}.")
print()
print("pi_gpi and pi_hyb agree at s=0 and s=1 (both pick action 1, then action 3); they")
print(f"diverge only at s=2: pi_gpi(2)={out['pi_gpi'][2]} (= pi_meta(2), inside B(2)={{0,2}})")
print(f"vs. pi_hyb(2)={out['pi_hyb'][2]} (outside B(2)). Since s=0's chosen action already")
print("transitions straight to s=2 (nxt(0,1)=2), this single downstream mistake at s=2 is")
print("enough to flip the total realized value all the way back at the start state -- the")
print("hybrid bootstrap's one-step lookahead is not always the right one to trust at every")
print("state along the resulting trajectory. Net effect: neither pi_hyb nor pi_gpi_B")
print("dominates the other in general -- Theorem (tie fails at k>=2) holds in both directions.")
