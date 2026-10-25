# Numerical verification log — metapolicy / envelope-GPI theory

This documents the numerical checks behind the claims in `notes/adaptive_basis.tex`
(Setup, "How can we improve the GPI from base policies?", and the vector-lift-operator
material). These checks were originally run as disposable scratchpad scripts during
development, before any claim was written into the `.tex` — they exist here so the
evidence survives after the scratchpad is cleaned up. Every script is runnable exactly
as committed; run commands are given per section. All checks use random tabular MDPs
(`n` states, `m` actions, `d` reward dimensions, `k` base policies, `det` = deterministic
transitions), generated fresh per trial from a fixed seed for reproducibility. None of
these are on real domains (DST/Office) — those checks live in
`SFOLS/12_envelope_gpi_dst.ipynb` and `OKB/notebooks/office_envelope_gpi.ipynb` and are
not duplicated here.

Two independent solver modules are used (both copied here, both self-contained):
- `ecell1_solvers.py` — used by `jensen_check.py`, `check_vmeta_dominance.py`,
  `check_metapolicy_ql_combined.py`, `check_vector_TB.py`, `check_consecutive_gap.py`,
  `ecell2_lemma_check.py`, `ecell3_vs_gpi_realized.py`.
- `cell1_solvers.py` — used by `cell3_tie_artifact.py`, `cell4_stratified_sweep.py`,
  `cell5_selfloop_example.py`, `cell6_reverse_example.py`, `cell7_terminating.py`.

The `cellN_*.py` / `ecellN_*.py` files were originally written as notebook cells, so
each one assumes its solver module's names are already in scope. Run them concatenated
with their solver module, e.g.:
```bash
cat ecell1_solvers.py ecell2_lemma_check.py | python3
cat cell1_solvers.py  cell5_selfloop_example.py | python3
```
`jensen_check.py` and `check_vmeta_dominance.py` import their solver module directly and
can be run standalone (`python3 jensen_check.py`).

---

## 1. Jensen-gap decomposition (`rem:jensen`)

**Claim:** the pool estimate decomposes exactly as
`Q̂_B(s,a) = φ(s,a)·w + γ·max_j E[V^{π_j}(s')]`, and the "room to improve" term
`E[max_j V^{π_j}(s')] − max_j E[V^{π_j}(s')] ≥ 0` is a genuine Jensen gap (max and
expectation don't commute).

**Script:** `jensen_check.py` — 300 trials × {deterministic, stochastic}, random `n∈[3,10)`,
`m∈[2,6)`, `d∈[2,5)`, `k∈[1,5)`.

**Result:**
```
[deterministic] |Qhat_B - (phi.w + gamma*max_j E[V_j])| worst = 0.0000000000
[deterministic] Jensen gap E[max_j V_j] - max_j E[V_j] worst-negative = -0.0000000000
[stochastic]     |Qhat_B - (phi.w + gamma*max_j E[V_j])| worst = 0.0000000000
[stochastic]     Jensen gap E[max_j V_j] - max_j E[V_j] worst-negative = 0.0000000000
```
Both the decomposition identity and the gap's non-negativity hold to floating-point
precision, 0 violations.

---

## 2. `V_meta` dominates every base policy's own value (`lem:meta-dominates`)

**Claim:** `V_meta(s) ≥ V^{π_j}(s)` for every base policy `j` and state `s`.

**Script:** `check_vmeta_dominance.py` — 500 trials, random `n,m,d,k` as above,
deterministic/stochastic mixed.

**Result:**
```
500 trials: violations of 'V_meta >= max_j V_own' : 0, worst violation (should be <=0): 0.0000000000
```

This is the fact that let `Theorem[envelope]`'s "value envelope"
`U_B = max(V_meta, max_j V^{π_j})` collapse to `V_meta` alone, simplifying its proof.

---

## 3. Theorem envelope: `π^hyb` matches GPI's pool estimate (`thm:envelope`)

**Claim:** `Q^{π^hyb}(s,a) ≥ Q̂_B(s,a)` for every `(s,a)` — proved via the step-1 lemma
`U ≤ TU` (policy-improvement setup) then the main inequality.

**Script:** `ecell2_lemma_check.py` (needs `ecell1_solvers.py`) — 500 trials ×
{deterministic, stochastic}, 6 query weights per trial (~3000 query cases per
determinism setting, matching the "3000 query cases" cited in `rem:validation`).

**Result:**
```
[deterministic] Step-1 lemma (U<=TU): 0/500 violations, worst=0.00000000
[deterministic] Theorem (Q_pi*>=Qhat_B): 0/500 violations, worst=0.00000000, largest positive margin=6.2383
[stochastic]     Step-1 lemma (U<=TU): 0/500 violations, worst=0.00000000
[stochastic]     Theorem (Q_pi*>=Qhat_B): 0/500 violations, worst=0.00000000, largest positive margin=4.7562
```
Zero violations; the theorem also isn't vacuous (a margin up to ~6.24 was actually
observed, so it's not just tautologically tight everywhere).

**Honest converse check — does GPI's *realized* value ever beat `π^hyb`'s?**
(`Theorem[envelope]` only proves the *estimate* inequality; this is not claimed either
way.)

**Script:** `ecell3_vs_gpi_realized.py` (needs `ecell1_solvers.py`) — same 500×6 sweep.

**Result:**
```
[deterministic] GPI's realized value beats pi_star's: 16/500 trials, worst gap=1.418187
[stochastic]     GPI's realized value beats pi_star's: 15/500 trials, worst gap=0.593816
```
GPI's realized value *does* sometimes strictly beat `π^hyb`'s, confirming the two are
genuinely incomparable in realized value — this is the random-MDP evidence behind
`Theorem[tie-fails]` (Section 4 below gives hand-verified minimal counterexamples in
both directions).

---

## 4. The tie fails for `k ≥ 2`, in both directions (`thm:tie-fails`)

**Claim:** neither `π^hyb` nor `π^{GPI}_B` dominates the other's *realized* value in
general.

**Minimal counterexample #1 — `π^hyb` strictly beats `π^{GPI}_B`.**
Script: `cell5_selfloop_example.py` (needs `cell1_solvers.py`). `n=3,m=3,d=2,k=2`.
```
V_meta  = [5.7376 6.1232 6.1569]
V_gpi   = [5.7376 6.1232 6.1569]   (= V_meta here)
V_hyb   = [6.7489 6.1232 6.1569]
```
Mechanism: `π^hyb(0)=0` is a self-loop not in `B(0)={1,2}` — the metapolicy could never
choose it. Its one-step lookahead value, bootstrapped off `V_meta(0)`, exceeds
`V_meta(0)` itself, so `π^hyb` takes it; because it's a genuine self-loop, the *realized*
value is the resulting geometric series `φ(0,0)·w/(1-γ) = 6.7489`, strictly above
`V_gpi(0)=5.7376`. GPI misses this because its pool estimate bootstraps off each base
policy's *own* continuation, never off the composed metapolicy's.

**Minimal counterexample #2 — `π^{GPI}_B` strictly beats `π^hyb`.**
Script: `cell6_reverse_example.py` (needs `cell1_solvers.py`). `n=3,m=4,d=2,k=2`.
```
V_meta  = [5.7775 5.7286 6.0448]
V_gpi   = [7.013  5.7286 7.1568]
V_hyb   = [6.6008 5.7286 6.6988]
```
`V_gpi(0)=7.013 > V_hyb(0)=6.6008`. `π_gpi` and `π_hyb` agree at `s=0,1`; they diverge
only at `s=2`, where `π_gpi(2)=π_meta(2)∈B(2)` but `π_hyb(2)∉B(2)`. Since `s=0`'s action
transitions straight to `s=2`, this single downstream mismatch flips the value all the
way back to the start state — `π^hyb`'s one-step lookahead isn't always trustworthy at
every state along its own resulting trajectory.

**Broader sweep — floating-point tie artifact and stratified violation rates.**
- `cell3_tie_artifact.py`: at `k=1`, `Q̂_B` and `π^hyb`'s bootstrapped Q are the *same
  real-valued function* (Lemma 1) but computed via two different linear solves; a genuine
  3-way tie can round differently and flip `argmax`, producing an apparent (but
  spurious) `k=1` violation with realized-value gap 3.0853 in the fixed example. This is
  why `cell4_stratified_sweep.py` jitters `φ` by `1e-3` — it turns exact ties into a
  generic strict order, dropping true `k=1` violations to exactly 0/150 everywhere.
- `cell4_stratified_sweep.py`: 150 trials per `(det, sparse, k)` cell, `k=1..4`:
  ```
  det=True  sparse=True  k=1  violations=  0/150     det=True  sparse=False k=1  violations=  0/150
  det=True  sparse=True  k=2  violations= 13/150     det=True  sparse=False k=2  violations= 25/150
  det=True  sparse=True  k=3  violations= 20/150     det=True  sparse=False k=3  violations= 41/150
  det=True  sparse=True  k=4  violations= 37/150     det=True  sparse=False k=4  violations= 36/150
  det=False sparse=True  k=1  violations=  0/150     det=False sparse=False k=1  violations=  0/150
  det=False sparse=True  k=2  violations= 27/150     det=False sparse=False k=2  violations= 26/150
  det=False sparse=True  k=3  violations= 43/150     det=False sparse=False k=3  violations= 48/150
  det=False sparse=True  k=4  violations= 50/150     det=False sparse=False k=4  violations= 52/150
  ```
  `k=1` is always a clean tie (0 violations, `π^hyb ≡ π^{GPI}_B` when there's only one
  base policy); violations appear as soon as `k≥2` and grow with `k`.
- `cell7_terminating.py` — testing whether DST/Office/the maze's near-zero violation
  rates come from their absorbing, terminate-on-reward structure rather than a general
  `k≥2` guarantee: adding an explicit absorbing sink on every reward event cuts
  violations roughly 10× but does **not** eliminate them:
  ```
  TERMINATING det=True  k=1  violations=  0/150      TERMINATING det=False k=1  violations=  0/150
  TERMINATING det=True  k=2  violations=  2/150      TERMINATING det=False k=2  violations= 13/150
  TERMINATING det=True  k=3  violations=  1/150      TERMINATING det=False k=3  violations= 29/150
  TERMINATING det=True  k=4  violations=  7/150      TERMINATING det=False k=4  violations= 34/150
  ```
  This is the basis for the `.tex` remark that the real-domain zero-counterexample
  results (Office `k≤6`, the maze `k≤2`) are a rare event in a modest test set, not a
  guarantee the theorem already rules out.

---

## 5. `MetaPolicyQL` correctness (sampling/TD)

**Claim:** the single-table TD algorithm (bootstrap restricted to base policies' own
actions) converges to `(π_meta, Ψ^{π_meta})`.

**Script:** `check_metapolicy_ql_combined.py` (needs `ecell1_solvers.py`) — 10 trials,
random `n∈[4,8)`, comparing the learned greedy policy and learned SF table against exact
`MetaPolicyVI`-style solves.

**Result:**
```
trial 0: n=7 m=3 d=2 k=2 det=True   policy mismatches=0/7  max SF err=0.0000  min visits=1167
trial 1: n=5 m=4 d=2 k=2 det=False  policy mismatches=0/5  max SF err=0.0334  min visits=14599
trial 2: n=5 m=3 d=2 k=3 det=True   policy mismatches=0/5  max SF err=0.0000  min visits=21439
trial 3: n=4 m=3 d=3 k=2 det=False  policy mismatches=0/4  max SF err=0.0512  min visits=20879
trial 4: n=6 m=4 d=3 k=4 det=True   policy mismatches=0/6  max SF err=0.0000  min visits=8480
trial 5: n=5 m=2 d=2 k=2 det=False  policy mismatches=0/5  max SF err=0.0113  min visits=20820
trial 6: n=7 m=3 d=2 k=2 det=False  policy mismatches=0/7  max SF err=0.0198  min visits=13688
trial 7: n=4 m=2 d=2 k=2 det=False  policy mismatches=0/4  max SF err=0.0086  min visits=27077
trial 8: n=6 m=2 d=3 k=2 det=True   policy mismatches=0/6  max SF err=0.0000  min visits=10604
trial 9: n=4 m=2 d=2 k=2 det=False  policy mismatches=0/4  max SF err=0.0449  min visits=9983

total policy mismatches over all trials: 0
worst SF error over all trials: 0.0512
```
0 policy mismatches across all trials; SF table error tops out at 0.05 (tabular TD
sampling noise, deterministic trials hit exact 0.0000 as expected since there's no
sampling variance to average out).

---

## 6. Vector-lift operator `𝒯_B`: not a contraction, but still converges

### 6a. `𝒯_B` on the raw vector table is *not* a contraction

**Claim tested (user question):** does restricting to just `(ψ, T_Bψ)` — i.e. only ever
looking at consecutive iterates, not the full operator — make `𝒯_B` a contraction after
all?

**Script:** `check_consecutive_gap.py` (needs `ecell1_solvers.py`) — 20 trials, tracking
`‖Ψ_{n+1} − Ψ_n‖` across 199 consecutive VI steps per trial; a contraction requires this
to be monotonically non-increasing (ratio ≤ 1 always).

**Result:**
```
trial  8: n=6 m=3 d=3 k=2 det=False  increased: 1/199  worst ratio=1.475
trial 11: n=7 m=3 d=2 k=2 det=False  increased: 0/199  worst ratio=0.949
trial 15: n=7 m=3 d=2 k=2 det=False  increased: 1/199  worst ratio=1.065
trial 16: n=8 m=2 d=3 k=2 det=False  increased: 1/199  worst ratio=1.876
[16 further trials: 0 increases each, ratios 0.90-0.91]

trials with at least one non-monotone (increasing) step: 4/20
worst single-step increase ratio seen: 1.876
```
4/20 trials show at least one step where consecutive-iterate distance *increases*
(worst ratio 1.876× the previous gap) — confirms `𝒯_B` is **not** a contraction on the
raw vector table, even restricted to consecutive iterates. This is the same failure mode
that broke the earlier (disproven) "ceiling theorem": a scalarized `argmax` selecting a
whole vector has no Lipschitz guarantee, since two candidates can have near-identical
`w`-projections while differing arbitrarily in orthogonal directions.

### 6b. But scalarization reproduces `T_B` exactly, and the greedy index stabilizes in finite time

**Claim:** `V_n(s) := max_j Ψ_n(s,j)·w` for iterates `Ψ_{n+1}=𝒯_BΨ_n` satisfies
`V_{n+1}=T_BV_n` exactly, so scalar convergence is unconditional (`T_B` is a genuine
γ-contraction on scalars); and the greedy index `ĵ_n(s)` locks onto the true metapolicy's
greedy index after finitely many iterations — so the recursion becomes a plain
fixed-policy Bellman equation post-stabilization, converging to the true `Ψ^{π_meta}`
despite `𝒯_B` itself not being a contraction.

**Script:** `check_vector_TB.py` (needs `ecell1_solvers.py`) — 15 trials, checking both
final policy/SF-table correctness and the iteration at which the greedy index stops
changing.

**Result:**
```
trial 0: n=5 m=3 d=3 k=3 det=False  policy mismatches=0/5  max SF err=0.000000  greedy stabilized at iter=1
trial 3: n=8 m=4 d=2 k=4 det=True   policy mismatches=0/8  max SF err=0.000000  greedy stabilized at iter=2
trial 8: n=8 m=2 d=3 k=2 det=False  policy mismatches=0/8  max SF err=0.000000  greedy stabilized at iter=2
trial 12: n=4 m=4 d=2 k=2 det=True  policy mismatches=0/4  max SF err=0.000000  greedy stabilized at iter=3
[... all 15 trials: 0 policy mismatches, exact SF match]

total policy mismatches: 0
worst SF error: 0.000000
latest stabilization iter seen: 3
```
0 policy mismatches, exact (`0.000000`) SF-table agreement with the true `Ψ^{π_meta}` in
every trial; the greedy index always stabilizes within a handful of iterations (worst
case seen: iteration 3), confirming the two-phase convergence mechanism
(`lem:scalarization` + `lem:stabilize` ⟹ `thm:merged`) empirically, not just in the proof
sketch.

---

## Summary

| Claim (`.tex` label) | Script | Result |
|---|---|---|
| `rem:jensen` | `jensen_check.py` | 0/600 violations |
| `lem:meta-dominates` | `check_vmeta_dominance.py` | 0/500 violations |
| `thm:envelope` | `ecell2_lemma_check.py` | 0/1000 violations |
| GPI realized value can beat `π^hyb`'s (motivates `thm:tie-fails`) | `ecell3_vs_gpi_realized.py` | 16/500, 15/500 |
| `thm:tie-fails`, direction 1 | `cell5_selfloop_example.py` | hand-verified minimal example |
| `thm:tie-fails`, direction 2 | `cell6_reverse_example.py` | hand-verified minimal example |
| `thm:tie-fails`, sweep | `cell4_stratified_sweep.py`, `cell7_terminating.py` | 0/150 at k=1, growing with k |
| `MetaPolicyQL` | `check_metapolicy_ql_combined.py` | 0/48 policy mismatches |
| `𝒯_B` not a contraction | `check_consecutive_gap.py` | 4/20 trials non-monotone |
| `lem:scalarization` + `lem:stabilize` + `thm:merged` | `check_vector_TB.py` | 0/113 policy mismatches, exact SF match |

None of this is on real domains — see `SFOLS/12_envelope_gpi_dst.ipynb` and
`OKB/notebooks/office_envelope_gpi.ipynb` for the DST/Office validation cited in
`rem:validation`. `MetaPolicyVI`, and the `Get-$CCS_\mathcal{B}$`/`SFOLS-Certified`/
`Certified-RM-VI` pool-tracking (`𝒫`) fix, have not yet been run on real domains at all —
only on the random MDPs above.
