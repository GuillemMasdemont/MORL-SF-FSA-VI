print("Minimal counterexample #1: pi_hyb STRICTLY BEATS pi_gpi_B (n=3, m=3, d=2, k=2)")
print()

nxt = np.array([[0, 0, 1], [2, 0, 0], [1, 1, 1]])
phi = np.array([
    [[0.19788131, 0.76069892], [0.72670145, 0.50731725], [0.27641115, 0.21776944]],
    [[0.70319209, 0.56015840], [0.53220768, 0.13021757], [0.26588580, 0.69896058]],
    [[0.98280998, 0.06645318], [0.72847186, 0.63121623], [0.75073821, 0.08131848]],
])
w = np.array([0.15245458, 0.84754542])
base_pis = [np.array([1, 0, 1]), np.array([2, 0, 0])]  # pi_1, pi_2
n, m, d, gamma = 3, 3, 2, GAMMA

out = solve_basis_and_compare(nxt, None, phi, base_pis, w, gamma, n, m, d)
print("B(0)={1,2}, B(1)={0}, B(2)={0,1}  (the two base policies' own actions per state)")
print("pi_meta =", out["pi_meta"], "  pi_gpi =", out["pi_gpi"], "  pi_hyb =", out["pi_hyb"])
print("V_meta  =", out["V_meta"])
print("V_gpi   =", out["V_gpi"], " (= V_meta here: GPI happens to reproduce the metapolicy)")
print("V_hyb   =", out["V_hyb"])
print()
print("pi_hyb(0)=0 is a self-loop (0 -> 0) NOT in B(0)={1,2}: the metapolicy could never")
print("choose it. Its one-step lookahead value, bootstrapped off V_meta(0), is")
print(f"  phi(0,0).w + gamma*V_meta(0) = {phi[0,0]@w:.4f} + {gamma}*{out['V_meta'][0]:.4f}"
      f" = {phi[0,0]@w + gamma*out['V_meta'][0]:.4f}")
print(f"which exceeds V_meta(0)={out['V_meta'][0]:.4f} itself, so pi_hyb takes it -- and since")
print("it is a genuine self-loop, the REALIZED value is the resulting geometric series:")
print(f"  phi(0,0).w / (1-gamma) = {phi[0,0]@w:.4f} / {1-gamma} = {phi[0,0]@w/(1-gamma):.4f}")
print(f"matching V_hyb(0)={out['V_hyb'][0]:.4f}, strictly above V_gpi(0)={out['V_gpi'][0]:.4f}.")
print()
Psi1, _ = full_sf_table(base_pis[0], nxt, None, phi, gamma, n, m, d)
Psi2, _ = full_sf_table(base_pis[1], nxt, None, phi, gamma, n, m, d)
print("Why GPI misses this action: its pool estimate for action 0 at state 0 bootstraps off")
print("EACH base policy's OWN (mediocre) continuation, not off the composed metapolicy's:")
print(f"  Psi_pi1(0,0).w = {Psi1[0,0]@w:.4f}   Psi_pi2(0,0).w = {Psi2[0,0]@w:.4f}"
      f"  ->  Qhat(0,0) = {max(Psi1[0,0]@w, Psi2[0,0]@w):.4f}")
print(f"  Qhat(0, pi_gpi(0)={out['pi_gpi'][0]}) = {out['Qhat'][0, out['pi_gpi'][0]]:.4f}"
      f"  (this wins the pool's own argmax)")
print(f"  Q_hybsrc(0,0) = {out['Q_hybsrc'][0,0]:.4f}  (bootstraps off V_meta instead -- higher)")
