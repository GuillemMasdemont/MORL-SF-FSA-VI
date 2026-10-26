# Handoff: SF-FSA-VI in stochastic domains with metapolicies (state on 2026-10-25)

This file is for an agent continuing this research on another machine (a CPU cluster). It summarises the goal, what
was established, every experiment with its numbers and files, the problems found, and what is left. Read sections 1–3
first, then the experiment you need. Paths are relative to the repository root `MORL/` unless absolute.

---

## 0. TL;DR

- **Goal.** Solve FSA (reward-machine) tasks optimally with SF-FSA-VI (Kuric, Infante et al., ICAPS 2024) in
  **stochastic** domains without computing the full CCS, using the user's **metapolicy / Certify-or-Grow** work.
- **Deterministic domains: closed.** The d extreme policies (optimal for e_1..e_d) make GPI exact for every weight
  (Theorem 2 in `paper/article.tex`), so SF-FSA-VI over them is optimal for every task.
- **Stochastic domains, with a model: solved.** *Lazy SF-FSA-VI* (Algorithm 4 in the paper): run Kuric's iteration over
  a small base, certify each visited weight w(u) (GPI estimate is a Bellman fixed point), and only where it fails call
  Certify-or-Grow (metapolicy first, then pi_hyb, then policy iteration) and add the policy. Exact on all 126 test tasks;
  about 2 added policies per task, 13–30 stored in total, against eps-CCSs of 117–572 policies that are not even exact.
  91–100% of the added policies (base kept across tasks) are metapolicy/pi_hyb compositions.
- **With a model the metapolicy saves no compute** versus policy iteration; its value is that new policies are
  compositions of stored ones (cheap to learn) and pi_hyb handles hedging.
- **Without a model (E10, cluster session):** with an oracle certificate that checks the GPI *policy and estimate* at
  w(u), Lazy SF-FSA-VI brings every windy task within 0.4% (zero-shot over the learned extremes: up to 2.1%) with ~1
  new policy and ~1M steps per task; with the added policies kept, 87–97% of new tasks need no new policy. It needs
  2–4x fewer samples than SF-FSA-VI over SFOLS (20–260x after the shared extremes); SFOLS's plan even degrades as
  noisy tables accumulate. Metapolicy growth costs as much as Q-learning over all actions. Flat Q-learning on the
  product MDP is 120x more expensive per task in windy but cheaper than learning the extremes in Office.
  **Certification without a model is still the main open problem** (the experiments use an oracle).
- **CCS under noise (E5, 10 seeds):** Certify-or-Grow's gain over SFOLS mostly vanishes (Office: +9% with the model
  certificate, −11% model-free with 16% false certificates; double slit: as fast as warm SFOLS).
- **Paper.** Section 4 (SF-FSA-VI) written in the laptop session; the cluster session added Figure 2 (stochastic CCS
  cost), Figure 5 (model-free Lazy SF-FSA-VI), their paragraphs, the zero-shot/planning sentences for Figure 4, and
  updated the abstract, contributions, outline and conclusion (§6). Analysis note: `notes/lazy_sf_fsa_vi.tex`.

---

## 1. People, goal and documents

- User: Guillem Masdemont (master's research, UPF), supervisor Vicenç Gómez; Guillermo Infante is a co-author of
  SF-FSA-VI (Kuric et al.). The paper draft is by Masdemont & Gómez.
- **Paper draft**: `paper/article.tex` ("Faster convex coverage sets via metapolicies over base policies").
  Sections: Background (SFs, GPI, CCS, OLS/SFOLS); Formalism (metapolicy over the base actions B(s); restricted SF
  Q-learning, Alg. 2; Bellman certificate; Certify-or-Grow, Alg. 3; hybrid policy pi_hyb, Eq. 14); **Section 4
  SF-FSA-VI**; Results (Figure 1: model-free CCS cost in DST/Office; Figure 2: the same in stochastic Office and
  the double slit; Figure 3: double slit, pi_hyb; Figure 4: Lazy SF-FSA-VI with exact SFs; Figure 5: Lazy SF-FSA-VI
  without a model); Conclusion; Appendix (convergence of the metapolicy recursion). 27 pages. Compile on the
  laptop with `cd paper && pdflatex article && biber article && pdflatex article && pdflatex article`, on the cluster
  with `bash cluster/tex.sh paper article` (identical output).
- **Notes** (LaTeX, `notes/`):
  - `gpi_basis.tex` — 5-page note: Theorem (deterministic), Proposition (pi_hyb dominance / action gap / coverage),
    hedging example figure, per-automaton-state ladder (GPI -> pi_hyb -> PI), per-task-type tables, model-free
    results. Uses the **acyclic** task semantics (see §3). Older versions in `notes/archive/`.
  - `lazy_sf_fsa_vi.tex` — **the 6-page analysis note** of the cluster session: CoG for the CCS under noise (E5,
    10 seeds), Lazy SF-FSA-VI with the model (E8 + zero-shot share, planning cost) and without it (E10), with tables
    and `notes/figures/lazy_mf_all.pdf`. Compile with `bash cluster/tex.sh notes lazy_sf_fsa_vi nobib`.
  - `certified_library.tex` — earlier 8-page note on an all-weights certificate (superseded; see §5 E1).
- **User's analysis document**: `fsa-rm-vi-directions.md` (directions D1–D6: cache certification, lazy CCS, local
  grow, CCS traversal by flips, compressed metapolicy, large action spaces). D2+D3 became Lazy SF-FSA-VI.
- **Papers**: Kuric et al. ICAPS 2024 (`00200-ICAPS.KuricD (3).pdf` at the root); Sheebaelhamd et al., NeurIPS 2026,
  "Split the differences, pool the rest" / MA-BC (`SFOLS/2605.12000v3 (2).pdf`): Pareto-path theorem (neighbouring
  CCS policies differ by single-state flips, all optimal at their corner weight) and the idea of pooling where experts
  agree, splitting where they differ.
- Repository guide for agents: `CLAUDE.md` (environments, subprojects). The relevant code lives in `SFOLS/`.

## 2. Working preferences of the user (follow these)

- Be **concise and critical**; give a recommendation, not a survey. When asked for a verdict, give the honest one
  (several claims were corrected when the comparison turned out unfair).
- Keep **math light in chat** (the chat does not render LaTeX); LaTeX is fine in files.
- **No linear solves inside the algorithms** (iterative evaluation only); linear solves are allowed for measurement
  (oracle values, regret). A "certificate" that runs value iteration from above was rejected as "cheating" (it solves
  control).
- **Fair comparisons**: same eps, same budget, same warm start, baselines truncated at the same size; count total
  iterations / environment steps as the efficiency measure. Policy iteration from scratch is a weak baseline; with a
  model the honest baseline is value iteration on the product MDP from scratch.
- Use **small wind / small slip** for compute-heavy runs.
- In the paper, **keep the user's writing style**: plain "we" sentences, connectors like "Note that", "Hence",
  "Moreover"; make it concise; do not rewrite their stochastic-domain text more than needed.
- The metapolicy is the user's main contribution: the final algorithm should use it, but report honestly where it
  helps (compositions, hedging via pi_hyb) and where it does not (compute with a model).
- The user starts each reply with a question about status: report what finished, what is running, and the numbers.

## 2b. Conversation history (condensed, in order)

The full transcript of the last session is on the user's laptop at
`~/.claude/projects/-Users-guillemmasdemont-Documents-Master-Summer-Research-MORL/9d95a8e3-0bb8-4182-a77e-b972662c24d8.jsonl`.

1. Starting point: the d extremes suffice in deterministic domains; in stochastic domains the CCS is huge. First idea:
   an all-weights certified library (E1). Verdict: no cheaper than warm SFOLS. Switched to per-task certification.
2. A per-task certificate by value iteration from above was rejected by the user as "cheating" (it solves control).
   Adopted instead: evaluate GPI at each FSA weight, check its Bellman gap, learn only on failure (task-level CoG).
3. Final direction then: deterministic theorem + stochastic pi_hyb with a per-task certificate; 3-page note
   `notes/gpi_basis.tex`; model-free Figure 1 replica on slippery Office (E5, original version).
4. "Is Algorithm 1 worth it?" -> with a model, GPI + PI is cheaper than pi_hyb + PI; the 8–19x claim used a weak
   baseline (E3). The user asked to compare against warm-started PI over all actions: done (E3).
5. The user wants Q-learning everywhere (PI = cheating model-free). Discussion of why pi_hyb + Q-learning would not beat
   GPI + Q-learning under noise.
6. "Why is CoG faster in Figure 1?" -> DST ablation (E4): the gain is not faster control but skipping the off-policy
   SF entries; under noise the on-path averaging dominates, so the gain flattens. Proposed lever: inherited counts.
7. Double-slit Figure 1 replica (E5): after the base, metapolicies converge in ~0 steps; found and fixed the cold-basis
   unfairness. The user asked whether a GPU cluster helps: no GPU needed; numba + many CPU cores.
8. Office version of the same (E5, fair): metapolicies still need ~1M steps; the model-free gain partly comes from a
   lenient certificate.
9. Model-free per-automaton-state experiment (E6): inherited counts are the lever; zero-shot GPI is already optimal on
   22/24 problems.
10. Note rewrite: Algorithm 1 as a ladder GPI -> pi_hyb -> PI, hedging example, per-task-type results (E7).
11. The user shared the MA-BC paper and `fsa-rm-vi-directions.md` and asked for an FSA algorithm that uses the metapolicy,
    "a constrained CCS plus a small refinement per iteration instead of the whole basis" -> Lazy SF-FSA-VI (E8, E9).
12. Paper Section 4 edited (deterministic theorem fixed, user's style kept); Section 4.2 completed with Algorithm 4,
    Proposition 1 and Figure 3. The model-free windy run was killed; the user is moving the code to a cluster.
13. Cluster session (2026-10-25, FRIDA): environment `sffsa`, sweep tool `cluster/sweep.py`, TeX container. The user
    asked to "fix everything left", add the stochastic CoG plots to the paper (concisely), and analyse Lazy SF-FSA-VI
    "like Infante's plots": how long a new task takes, how many certificates / growths, whether it stays zero-shot,
    and CoG against plain Q-learning over all actions. Done as E5 (10 seeds) and E10 below; nothing runs on the login
    node (user's instruction).

## 3. Concepts and notation

- Low-level MDP with d **exits** e_1..e_d: entering exit j emits feature e_j and ends the episode; other transitions
  have feature 0. Weight w >= 0 gives reward w.phi. SFs psi^pi(s,a) in [0,1]^d, Q^pi_w = psi^pi . w.
- **GPI** over a set Pi: pi_GPI(s) = argmax_a max_pi psi^pi(s,a).w. It can take ANY action; its weakness is that it
  values each action followed by a single base policy. **Instant GPI value** Q^GPI_w = max_j Q^{pi_j}_w (a lower bound
  on Q*_w and on the GPI policy's values).
- **Metapolicy**: optimal policy of the MDP restricted to B(s) = {pi_j(s)} (the base actions); value V_meta.
  **pi_hyb**(s) = argmax over ALL actions of Q_meta(s,a) (one greedy step on top of the metapolicy).
- **Bellman certificate**: a policy is optimal for w iff its Bellman gap is zero at every state; eps gap ->
  eps/(1-gamma)-optimal. **GPI-estimate certificate** (used by Lazy SF-FSA-VI): Q^GPI_w <= Q_{V^GPI} always (V^GPI =
  max_a Q^GPI), and Q^GPI_w = Q*_w iff Q^GPI_w = Q_{V^GPI} (one backup to check).
- **Certify-or-Grow (CoG)**: at a query weight, learn the metapolicy; if certified return it, else solve and add the
  policy to the basis. Inside OLS it builds the same CCS as SFOLS with far fewer trained policies.
- **SF-FSA-VI** (Kuric): automaton states U; entering exit j in u moves to tau(u,j). Weights iterate
  w_j(u) = max_a max_pi psi^pi(e_j,a).w(tau(u,j)), with w_j(u) = 1 if tau(u,j) is accepting. Exact iff Pi is a CCS.
- **Two task semantics in the code** (important):
  - *acyclic* (`certlib/hybrid.py`, `certlib/pertask.py`, `gpi_basis/fsa.py`): exits whose symbol has no edge from u
    get weight 0 (dead ends); tasks are solved in one backward pass, reverse topological order.
  - *Kuric semantics* (`certlib/lazyfsa.py`, `q_lazy_fsa.py`): such exits are **self-loops** (tau(u,j) = u), so the
    high-level problem is iterative. **Assumption**: the jump between automaton states is discounted by gamma,
    w_j(u) = gamma * V_{w(tau(u,j))}(e_j); without it, bumping into a wall at an exit cell re-enters the exit with an
    undiscounted self-loop and Kuric's iteration is not a contraction. **To confirm with Infante.**
- **Inherited counts**: when a learner is warm-started from a stored SF table, each entry keeps the visit count of the
  entry it was copied from, so the step size 1/(n+1)^0.7 does not overwrite a good initialisation with noise.
- *fresh* = every task starts from the 6 extremes; *stream* = added policies are kept across the task sequence.
  *round* = one Kuric fixed point + certification pass in Lazy SF-FSA-VI.

## 4. Theory established

1. **Theorem (deterministic)** — `paper/article.tex` Thm. 2 (Section 4.1); `notes/gpi_basis.tex` Thm. 1.
   max_k psi^{pi_k}(s,a).w = Q*_w(s,a) for every w >= 0 (the paper states it on the simplex, which is equivalent by
   scaling). Proof: a deterministic trajectory enters at most one exit, so Q*_w = max_k w_k D_k with D_k = Q*_{e_k};
   and psi^{pi_k}.w >= w_k D_k by non-negativity; sandwich. Consequences: GPI, metapolicy and pi_hyb over the d
   extremes are optimal; SF-FSA-VI over them is optimal for every task; Lazy SF-FSA-VI adds nothing. Fails with
   stochastic transitions because a policy can split its probability between exits (hedging).
2. **Proposition (stochastic, pi_hyb)** — `notes/gpi_basis.tex` Prop. 1: (i) max_j Q_j <= Q_meta <= Q^{pi_hyb};
   (ii) action-gap condition gamma*||V* - V_meta|| < Delta makes pi_hyb optimal (weaker than GPI's); (iii) if an
   optimal policy uses only base actions, metapolicy and pi_hyb are optimal. No bound on the base size in terms of d.
3. **Lazy SF-FSA-VI** — paper Alg. 4 + Prop. 1 (Section 4.2): terminates (each added policy is new, finitely many
   deterministic policies), exact at termination (GPI estimates at the exits are V*, so Kuric's fixed point is the
   optimal one), adds nothing in deterministic domains. Lower bounds are monotone.
4. **Why Certify-or-Grow beats SFOLS in the deterministic Figure 1** (measured, §5 E4): not faster control. SFOLS must
   learn and confirm the SF entries of actions it rarely takes (reached only by eps-exploration, while their targets
   keep moving); CoG learns only the on-path entries (|B(s)| ~ 1.3 in DST) and fills the rest once at the end.
5. **Why the metapolicy is about as expensive as policy iteration with a model**: computing it is a restricted solve
   over all states; |B(s)| is close to |A| with corner bases (2.9 of 4 in Office), so the restriction saves little.

## 5. Experiments and results (with files)

All run with the `okb` conda env, from `SFOLS/`, `OMP_NUM_THREADS=1`. "Sweeps" = |S||A| one-step backups (an
evaluation sweep counts 1/|A|; an SF sweep counts d). Results in `SFOLS/results/` and `SFOLS/results/certlib/`.

### E1. All-weights certified library (earlier sessions) — superseded
- Code `SFOLS/certlib/` (`cl.py`, `certificate.py`, `experiments.py`, ...), note `notes/certified_library.tex`.
- Finding: certifying GPI's *estimate* over all weights needs about an eps-CCS (cost ~ warm SFOLS); it cuts trained
  policies 10–20x but not stored SFs or compute. GPI's realised value is much better than its estimate, but has
  hundreds to thousands of GPI cells. Led to per-task certification.

### E2. Per-task certification and pi_hyb curves (acyclic semantics)
- Code `certlib/hybrid.py` (`curve`, `tasks`), figure `certlib/figures_hyb.py` -> `notes/figures/hyb_optimal.pdf`.
- Share of held-out weights where each policy is optimal with the 6 extremes: Office p=0.1 GPI 86% / metapolicy 65% /
  pi_hyb 97.5%; p=0.3 69 / 64 / 93%; windy 93 / 91 / 96%. Double slit: pi_hyb optimal everywhere from 9 policies
  (CCS 181). Policy iteration from pi_hyb reaches the optimum in at most 3 steps.
- Per task (base = 6 extremes; Office 103 tasks, windy 60): zero-shot regret 1.6e-5 / 8.3e-4 / 3.0e-3; pi_hyb without
  PI 1.5e-5 / 1.0e-6 / 0.

### E3. Warm starts for certified planning (acyclic) — `certlib/hybrid.py warm`, results `hybwarm_*_6.pkl`
Sweeps per task to the certified optimum (Office p=0.1 / p=0.3 / windy):

| start, then PI over all actions | sweeps |
|---|---|
| PI from scratch | 773 / 672 / 166 |
| value iteration from scratch (best tol) | 59 / 120 / 94 |
| GPI | **33 / 68 / 11** |
| best base policy per state | 35 / 73 / 12 |
| pi_hyb (metapolicy rung) | 40 / 85 / 16 |

GPI + PI is the cheapest; the old "8–19x" claim was against PI from scratch (weak). Fair speed-up of the base vs VI
from scratch: ~1.8x Office, ~8.6x windy.

### E4. Why CoG is faster in the deterministic Figure 1 — `q_speedup_ablation.py`, `results/q_speedup_ablation_dst.pkl`
DST, same warm start (GPI over exact tables of earlier queries), queries 3–19, 10 seeds, median totals:
greedy policy optimal from 5.7k (all actions) vs 0.3k (B(s)) steps; greedy values exact 12.4k vs 8.8k; last change of
any SF entry (Figure 1's metric) 334k vs 17.5k; stop 1.24M vs 156k; tracked pairs 4,216 vs 1,343. Figure 1's 3.2x is
diluted by the ~120k steps both pay to learn the two extremes.

### E5. Model-free CCS construction (Figure 1 replicas) — `q_ccs_stochastic.py`, `plot_ds_q_cost.py`
- Learner: SF Q-learning, step 1/(n+1)^0.7, tight optimistic init (Psi = e_argmax(w), so Q = max w), eps = 0.1,
  exploring starts; "steps to converge" = oracle (last check where the greedy value at the start was more than 1e-3
  below the optimum).
- **Office slip 0.05, eps = 0.05, 3 seeds, medians** (`results/office_q_runs_slip0.05.pkl`, fair CoG arms in
  `office_q_runs_slip0.05_bw.pkl`; figure `notes/figures/office_slip_q_ccs_cost.pdf`):

  | arm | steps to converge | all steps | notes |
  |---|---|---|---|
  | SFOLS | 134.9M | 190.9M | ~99 learns |
  | SFOLS + warm start | 129.5M | 189.7M | |
  | CoG, model certificate (warm base) | 149.9M | 204.4M | 14–23 fallbacks (exact gap <= 1e-2 at all states fails under noise) |
  | CoG, model-free certificate (warm base) | 112.2M | 169.6M | 14–18 false certificates |

  Every arm reaches the same CCS quality (worst eps-gap 0.005–0.006). Each metapolicy still costs ~1M steps (warm full
  learn ~1.2M). With N0 = 10 pseudo-visits CoG reached 115M.
- **Double slit, wind +-1, gamma 0.99, eps 1e-3, 3 seeds** (`results/ds_q_runs_wind1_g0.99_eps0.001_cert0.01_stab0*.pkl`,
  figure `notes/figures/ds_q_ccs_cost.pdf`): steps to converge (mean) SFOLS 12.6M, warm 4.40M, CoG 3.87M, CoG
  model-free 4.21M (with the base warm-started; with a cold base CoG lost, 5.6M). After the base, metapolicies converge
  in ~0 steps; the base is 80–100% of the cost. Learned psi0 error 2–6e-3 (so only 5–6 OLS queries). Wind +-3 does not
  converge in 3M steps per learn.
- The all-steps panels are dominated by the step cap: the noise-robust stopping rule (policy stable for 30 checks)
  almost never fires because near-tie actions keep flipping.
- **10 seeds (cluster, 2026-10-25; paper Figure 2, `paper/figures/q_ccs_cost_stochastic.pdf`; data
  `SFOLS/results/q_ccs_10seeds/`; sweeps `cluster/sweeps/{office,ds}_ccs_mf.yaml`; all arms with QBASISWARM=1; seeds
  0–2 reproduce the 3-seed runs exactly).** Steps to converge, mean +- std:

  | arm | Office slip 0.05 | double slit wind +-1 |
  |---|---|---|
  | SFOLS | 135.7 +- 5.6M (all steps 191.6M) | 13.2 +- 1.3M (26.0M) |
  | SFOLS + warm start | 133.4 +- 5.4M (193.2M) | 4.24 +- 0.66M (30.6M) |
  | CoG, model certificate | 147.4 +- 12.6M (206.5M); 17.5 fallbacks / 96.5 queries | 4.49 +- 1.36M (21.3M) |
  | CoG, model-free certificate | 120.9 +- 10.9M (176.6M); 15.5 false of 94 certified | 4.94 +- 1.59M (22.5M) |

  The 3-seed claim "CoG 12% faster than warm SFOLS in the double slit" does not hold with 10 seeds (4.49 vs 4.24M);
  CoG stops earlier (21 vs 31M all steps). Under noise the gain of CoG over SFOLS mostly vanishes.

### E6. Model-free learning per automaton state — `q_fsa_stochastic.py`, `results/fsa_q_office_slip0.05.pkl`
Office slip 0.05, 20 tasks = 24 distinct automaton-state problems, base = 6 extremes learned model-free (2.5M steps
each), 2 seeds, scalar Q-learning at w(u), oracle metric: steps until within 1e-3 of optimal at the entry states
(budget 2M). Zero-shot GPI is within 1e-3 on 22/24 problems.

| learner | mean steps per task | never left the optimum (of 48) |
|---|---|---|
| from scratch | 1.08M | 0 |
| GPI start | 0.89M | 15 |
| GPI start + inherited counts | 0.58M | 33 |
| metapolicy (B(s)) | 0.99M | 17 |
| metapolicy + inherited counts | **0.51M** | 34 |

Per task type (metapolicy + inherited vs GPI + inherited): sequential 0.59M vs 1.63M, composite 1.56 vs 1.89,
disjunction 1.40 vs 1.65, random 0.39 vs 0.38. A 20-policy base was no better. Indicative (one base, 2 seeds).

### E7. Ladder per task type and the hedging example (acyclic) — `certlib/hybrid.py fsa`, `figures_hyb.py example`
- Results `results/certlib/hybfsa_*_6.pkl`; table in `notes/gpi_basis.tex` (Table 1). Office p=0.3: 9/103 tasks lose
  > 1e-4 with GPI (up to 0.13%); the metapolicy fixes 8, pi_hyb all 9. Windy: 12/52 hedging tasks lose > 1e-4 (up to
  1.2%); the metapolicy fixes none, pi_hyb all. GPI rarely passes the all-state zero-gap certificate in Office (8–18%)
  although its regret is tiny.
- Hedging example (`notes/figures/hyb_example_windy.pdf`, windy task 7, w(u0) = (0.63, 0.63, 0, 0, 0, 0)): loss at the
  start GPI 1.19%, metapolicy 1.25%, pi_hyb 0 (max 0.04%); pi_hyb takes RIGHT (not in B(s)) at 50 central states.

### E8. Lazy SF-FSA-VI with a model (Kuric semantics) — `certlib/lazyfsa.py`, `plot_lazy_fsa.py`
- Results `results/certlib/lazyfsa_{office1f2_30,office3f2_30,windy_60}.pkl`; logs `results/logs/lazyfsa_*.log`;
  figure `paper/figures/lazy_fsa.pdf` (= `notes/figures/lazy_fsa.pdf`). eps-CCS libraries from stored SFOLS runs
  `results/certlib/run_<domain>_sfols_<eps>_0.0.pkl` (Office eps 2e-3, windy 1e-5).

| | Office p=0.1 (33) | Office p=0.3 (33) | Windy (60) |
|---|---|---|---|
| zero-shot over 6 extremes, max rel. regret | 1.1e-3 | 3.2e-3 | 1.4e-2 |
| zero-shot over eps-CCS (size) | 3.1e-6 (262) | 2.5e-4 (572) | 0 (117) |
| OLS-order CCS prefix that is optimal | never on 17/33 | median 100, never on 5/33 | 11–15 |
| Lazy: regret | 0 | 0 | 0 |
| fresh: rounds / added per task | 2.15 / 1.79 | 2.61 / 2.42 | 2.00 / 2.20 |
| stream: added per task / final base | 0.67 / 28 | 0.73 / 30 | 0.12 / 13 |
| added kinds fresh (metapolicy / pi_hyb / PI) | 10 / 22 / 27 | 31 / 14 / 35 | 24 / 81 / 27 |
| added kinds stream | 18 / 2 / 2 | 19 / 4 / 1 | 3 / 4 / 0 |
| cost per task, fresh / stream (sweeps) | 200 / 72 | 558 / 160 | 229 / 24 |
| product MDP by VI from scratch | 67 | 130 | 282 |

  Growing by GPI + PI instead (ablation) adds the same number of policies; only 8–71% are the GPI policy itself.
  Deterministic Office (`office0f2`, 5 tasks): nothing added, as Theorem 2 predicts.

### E9. Lazy SF-FSA-VI without a model — `q_lazy_fsa.py` (uses `q_fsa_stochastic.py` learners)
- *Superseded by E10: `q_lazy_fsa.py` was rewritten (the E9 version is in git, commit 06c7d91).*
- Assumption: the certificate is an oracle check that GPI over the stored tables is within delta = 1e-3 of optimal at
  the start and the exits; learners stop after STABLE=100 good oracle checks or a 3M budget; added policies are learned
  first on B(s) with inherited counts (metapolicy), then on all actions if not certified.
- **Office slip 0.05, 13 tasks** (`results/lazy_fsa_mf_slip0.05.pkl`): zero-shot over the learned extremes (6 x 2.5M
  steps) already passes on all tasks (max rel. regret 4.2e-4, mean 7.6e-5): nothing is learned, versus ~130M steps for
  SFOLS's eps-CCS in the same domain.
- **Windy, 12 tasks**: killed after ~3 h with no result (stream mode prints only at the end; learns likely hit the 3M
  cap because the strict oracle stop rarely holds under noise, and certification kept failing). Needs a rerun (§8).

### E10. Model-free Lazy SF-FSA-VI against the previous algorithms (cluster) — `q_lazy_fsa.py` (rewritten), `mf.py`
- **Question (user):** how long a new task takes, how many certificates / growths, whether it stays zero-shot, and CoG
  against plain Q-learning over all actions; plots "like Infante's" (Kuric et al. Figure 3: performance vs samples).
- **Methods** (all learn from samples with the numba learners of `mf.py`, same step size 1/(n+1)^0.7, eps 0.1,
  exploring starts; the model only measures): the d extremes (shared); Lazy SF-FSA-VI growing with the metapolicy
  (model-free CoG: learn on B(s) until the policy converges; keep it if optimal, else continue on all actions) or
  with Q-learning over all actions, both from the GPI table, from the extremes on every task (fresh, parallel) or
  keeping the added policies (stream), at most 5 rounds; SF-FSA-VI over all policies learned by SFOLS from the same
  extremes (OLS eps 1e-3, up to 100 queries, plan evaluated after every query); flat Q-learning on the product MDP.
- **Certificate (assumption):** oracle check that at w(u) the GPI policy is within 1e-3 and the GPI estimate within
  1e-2 of V*_w(u) at the start and exits (relative to max V*); learners stop after 100 good checks of 1000 steps or
  50M steps (flat: 300M; flat's tolerance is 1e-3 V* at the start). SF tables start at zero (pessimistic).
- **Data:** `SFOLS/results/lazy_mf/inherit{0,1}/lazy_mf_<domain>_s<seed>.pkl` (10 seeds; windy 60 tasks, Office
  p = 0.1 / 0.3 33 tasks, double slit 1 task), sweeps `cluster/sweeps/lazy_mf.yaml`, `lazy_mf_ds.yaml`; figures
  `plot_lazy_mf.py` -> `paper/figures/lazy_mf.pdf` (windy, Office p = 0.3), `notes/figures/lazy_mf_all.pdf`.
  Each method with its better count setting: Lazy without inherited counts, SFOLS with (windy, Office) / without (ds).

  | | windy | Office 0.1 | Office 0.3 | double slit |
  |---|---|---|---|---|
  | learned extremes: steps; zero-shot regret mean / max | 42.6M; 0.90 / 2.1% | 128M; 0.008 / 0.15% | 231M; 0.054 / 0.33% | 33.6M; 2.1% |
  | Lazy (metapolicy) from extremes: policies, steps / task; regret | 0.85, 1.07M; 0.17 / 0.32% | 0.68, 0.32M; 0.008 / 0.11% | 1.31, 1.03M; 0.014 / 0.12% | 1, 2.2M; 0.024% |
  | Lazy (Q-learning) from extremes: steps / task | 1.04M | 0.23M | 0.82M | 2.26M |
  | kept: tasks with no new policy (2nd half); steps / task | 97% (100%); 0.043M | 91% (95%); 0.045M | 87% (91%); 0.15M | – |
  | SFOLS: steps after the extremes; final regret mean / max | 61.8M; 0.21 / 0.50% | 384M; 0.004 / 0.11% | 646M; 0.021 / 0.27% | 40.7M; 0.012% |
  | flat Q on the product: steps / task | 127M (96% converge) | 0.87M | 2.85M | 2.07M |
  | total SFOLS / total Lazy; tasks for SFs to beat flat Q | 2.3x; <1 | 4.0x; 155 | 3.7x; 85 | 2.1x; – |

- **Findings:** (1) the oracle must check the *estimates*: a policy-only check certified windy tasks with 15% regret
  (pilot 1); (2) with the policies kept most new tasks are zero-shot; (3) metapolicy and Q-learning growth cost the
  same (0.93–1.4x); the metapolicy is certified alone for 8–19% of the new policies (33–58% when kept); (4) in windy
  80% of the new policies are the hedging one at u0 (~220 of 611 states differ from any stored policy); (5) SFOLS
  without inherited counts degrades: mean regret ~1e-3 after 9–17 queries, 1.3–9.2% at 62–100 (max over noisy
  tables overestimates the exits); (6) the shared extremes are 94–99% of Lazy's total cost.
- **Pilots** (`cluster/sweeps/lazy_mf_pilot{,2,3,4,5}.yaml`, 8 tasks, 1 seed) fixed: estimate checks (pilot 2), cheap
  model-free CoG that decides on the restricted policy before refining estimates (3), estimate tolerance 1e-2 (4),
  no inherited counts for Lazy (5). Optimistic vs zero start: `SFOLS/checks/check_init.py` (zero is 20–60x faster on
  windy corridor extremes, slower on one Office extreme). `SFOLS/checks/check_mf.py` validates the numba learners
  against the Python loop (same Q error at fixed budgets) and the evaluators against exact solves.

## 6. Paper edits made in this session (`paper/article.tex`)

- New **Section 4 "SF-FSA-VI algorithm"** (`\label{sec:fsa}`): setup paragraph with Kuric's update (Eq. 15).
- 4.1: Theorem 2 (deterministic) and proof fixed (wrong inequality, undefined macros, notation omega -> **w**). The user
  later re-edited it (stated on Delta(d); proof wording). **Typo "hemce" in the proof is the user's, left untouched.**
- 4.2: user's motivating paragraph kept; added Lazy SF-FSA-VI: certificate (Q^GPI = Q_{V^GPI}), **Algorithm 4**,
  **Proposition 1** with proof, one closing sentence.
- End of Section 5: setup paragraph (slippery Office p in {0.1, 0.3}, windy two-stage, Kuric semantics + gamma jump
  assumption), **Figure 3** (`lazy_fsa.pdf`), results paragraph. Exact SFs only; the model-free FSA results are NOT in
  the paper.
- Cluster session: Figure 2 `q_ccs_cost_stochastic.pdf` + paragraph (E5, 10 seeds); Figure 5 `lazy_mf.pdf` +
  paragraph (E10); two sentences after Figure 4 (zero-shot share with kept policies, planning 10–18x cheaper); abstract
  and first contribution qualified for stochastic domains, a third contribution (SF-FSA-VI), outline mentions
  Section 4, conclusion mentions SF-FSA-VI and the model-free certificate. The user's typos (e.g. "hemce",
  "optimallity") are still untouched.

## 7. Problems, pitfalls and fixed bugs

- **Float pycddlib**: "numerical inconsistency" crashes on noisy learned points, and float cdd can silently drop
  corners in 6-D. Fixed with an exact Fraction fallback in `q_ccs_stochastic.corner_weights` (counted as "exact cdd
  fallbacks"); per-job try/except so one failure does not kill the pool.
- **Noise and step sizes**: optimistic init Psi = 1 (Q = sum w) takes millions of steps to unlearn -> use Q = max w;
  constant step sizes never settle -> decaying 1/(n+1)^0.7; warm starts are overwritten by the first noisy update
  (first step 0.62) -> inherited counts.
- **Stopping rules under noise**: "no SF change > tol" never fires; "greedy policy unchanged for 30 checks" rarely fires
  (near-tie flips, 1–2 states per check late in learning); a looser rule (<= 10 flips) stops before convergence and
  biases results. Report oracle "steps to converge"; the all-steps panel then mostly shows the step cap.
- **Model-free certificate**: completing off-B(s) entries with 100 samples per pair and tolerance 1e-2 false-certifies
  15–22% of metapolicies; the exact gap on noisy learned policies is too strict (many fallbacks). A sound model-free
  certificate is the main open problem.
- **Unfair baselines found and fixed**: CoG learned its basis cold while warm SFOLS warm-started its 2nd..d-th extreme
  (fixed with `QBASISWARM=1`; the paper's deterministic Figure 1 code also learns the CoG basis cold); PI from scratch
  as a baseline (use VI on the product MDP).
- **Prefix baseline bug** (fixed): "exit weights exact" does not imply an optimal plan (the policy inside u0 can still
  be wrong); `lazyfsa.prefix_needed` now uses executed regret.
- **Zero-gap certificate over all states is strict**: GPI fails it in Office at 82–92% of automaton states although its
  regret is ~1e-5 (tiny gaps far from the task's entry states). A reach-weighted certificate would be closer to what
  SF-FSA-VI needs (`MOMDP.reach_bound` exists in `certlib/mdp.py` but is loose at gamma = 0.99).
- **Metapolicy restriction is weak with corner bases**: |B(s)| ~ 2.9 of 4 in Office; the "compressed metapolicy" (D5:
  solve only on states where base policies disagree) does not pay here.
- **Python stdout is buffered when piped**: use `flush=True`; long multiprocessing runs should log per job / per task.
- **Model-free SF-FSA-VI needs accurate estimates, not only good policies** (E10): an oracle certificate that checks
  only the GPI *policy* at w(u) certified windy tasks with 15% regret, because Kuric's iteration reads the GPI
  *estimates* at the exits. Check the estimates too (`q_lazy_fsa.py`, QESTIMATE=1).
- **Optimistic starts in windy**: Psi = e_i takes 7–26M steps to unlearn in corridors that cannot reach the exit
  (0.2–0.4M from zero); use the pessimistic (zero) start, which also keeps GPI an approximate lower bound.
- **Inherited counts hurt when the new policy is new** (hedging): the copied counts make the step sizes tiny, ~10x
  slower learning (E10 pilot 5). They help SFOLS (queries close to stored policies) in windy/Office.
- **Many noisy tables overestimate**: SF-FSA-VI over the SFOLS policies gets *worse* after ~15 queries (mean regret
  1e-3 -> 1–9e-2 at 100 queries, windy): the max over noisy learned estimates is biased upwards at the exits.
- **Estimate accuracy has a noise floor**: requiring estimates within 1e-3 at every exit is out of reach with
  1/(n+1)^0.7 steps in Office (extremes capped at 50M); 1e-2 removes the gross biases.
- **Cluster**: the `amd` node is shared and memory-bound (other users hold ~720 of 773 GB); ask for what the job uses
  (~0.6 GB per process). The sbatch wrapper prints a banner on stdout. The login node's /tmp (and the scratchpad)
  are not visible from compute nodes: use the repository's `.build/` (git-ignored). `os.cpu_count()` returns the
  whole node inside an allocation.
- **Runtime**: everything is single-core Python (~50k vector SF-Q steps/s, ~140k scalar Q steps/s per core). The
  scratchpad directory is wiped between sessions; keep results under `SFOLS/results/`.

## 8. Open problems and next steps (in order)

*Cluster-session status:* item 2 is done (E10, all fixes in); item 4 is done (outline, conclusion). Still open: 1
(the experiments use an oracle certificate), 3, 5. New: (6) the model-free plan's precision is limited by the noise
of the learned SFs at the exits (estimates to 1e-2); a pessimistic estimate (confidence bound) for the GPI max would
also remove the overestimation of SF-FSA-VI over many tables; (7) learning the extremes accurately at every exit is
94–99% of the model-free cost — reusing samples across extremes (one behaviour policy, all d SF components) is the
obvious lever.

1. **Model-free certificate with confidence bounds.** Use the visit counts (already stored for inherited counts) to put
   a confidence interval on the Bellman gap of the GPI estimate / metapolicy; certify when gap + interval <= eps, with
   eps at the noise floor (~1e-3 here). This makes the model-free Lazy SF-FSA-VI loop terminate and be sound w.h.p.
2. **Rerun model-free Lazy SF-FSA-VI on windy** (the domain where zero-shot loses 1.4%): `QDOM=windy python
   q_lazy_fsa.py 12 6` with (a) per-task logging with flush, (b) a cap of 3 rounds per task, (c) the fresh mode
   parallelised over tasks, (d) numba for `learn` / `q_learn` / `sf_learn`, (e) on the cluster. Then fill the
   `WINDY_MF` placeholder in `notes/lazy_sf_fsa_vi.tex` and decide whether to add a model-free paragraph to the paper.
3. **Confirm the gamma-discounted jump** between automaton states with Infante (paper Section 5 states it).
4. Paper polish: mention Section 4 in the introduction's outline and in the conclusion; check figure placement.
5. Optional: large or continuous action spaces (the metapolicy maximises over k proposals instead of all actions,
   which is the one place where it can save compute with a model); CCS traversal by single-state flips (D4) for d = 2.

## 9. How to run

### Environment
- Python 3.10.18, numpy 1.26.4, scipy 1.12.0, matplotlib 3.10.9, pycddlib 2.1.6 (`import cdd`), gym 0.23.1 (old gym,
  for `RM/sf-fsa-vi/envs`), mo_gymnasium 1.3.1 (DST only), Pillow (only for page previews). TeX (pdflatex, biber) for
  figures with `text.usetex` and the paper. Locally: `/opt/anaconda3/envs/okb/bin/python`.
- The scripts find `RM/sf-fsa-vi` through relative paths (`../RM/sf-fsa-vi` from `SFOLS/`): copy the whole `MORL/`
  tree (at least `SFOLS/`, `RM/sf-fsa-vi/`, `notes/`, `paper/`). The stored SFOLS runs
  `SFOLS/results/certlib/run_*.pkl` and `taskopt_*.pkl` are inputs to several experiments: copy them.
- Always `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (numpy would otherwise oversubscribe cores in the pools).

### Commands (from `SFOLS/`)
```bash
# Lazy SF-FSA-VI with a model (E8); ~1 min (Office), ~40 min (windy, mostly the prefix baseline)
PYTHONPATH=. python -m certlib.lazyfsa office1f2 30
PYTHONPATH=. python -m certlib.lazyfsa office3f2 30
PYTHONPATH=. python -m certlib.lazyfsa windy 60
PYTHONPATH=. python -m certlib.lazyfsa prefix windy 60      # recompute the OLS-prefix baseline of a saved run
python plot_lazy_fsa.py                                     # Figure 3 of the paper

# Ladder / warm starts / pi_hyb curves (acyclic semantics; E2, E3, E7)
PYTHONPATH=. python -m certlib.hybrid curve office1f2 6,9,12,20,40
PYTHONPATH=. python -m certlib.hybrid tasks office1f2 6
PYTHONPATH=. python -m certlib.hybrid warm office1f2 6
PYTHONPATH=. python -m certlib.hybrid fsa office1f2 6
PYTHONPATH=. python -m certlib.figures_hyb [example]

# Model-free CCS construction (E5); hours per run at Office scale
python q_ccs_stochastic.py run 0.05 3 0 8                    # Office slip 0.05, 3 seeds, n0 = 0, 8 workers
ARMS=certify,certify-mf QBASISWARM=1 python q_ccs_stochastic.py run 0.05 3 0 6
QDOMAIN=ds QWIND=1 QGAMMA=0.99 QEPS=1e-3 QCERT=1e-2 QSTABK=0 QMAX=5e6 python q_ccs_stochastic.py run 0 3 0 8
QDOMAIN=ds ... python q_ccs_stochastic.py exact 0            # exact OLS size and exact-CoG fallbacks (fast)
python plot_ds_q_cost.py office slip0.05 slip0.05_bw         # Office figure; without 'office': double slit

# Model-free learners per automaton state (E6) and model-free Lazy SF-FSA-VI (E9)
python q_fsa_stochastic.py 17 14 8                           # 20 tasks, 6 + 14 base policies, 8 workers
python q_lazy_fsa.py 10 6                                    # Office slip 0.05
QDOM=windy python q_lazy_fsa.py 12 6                         # windy (needs the fixes of §8.2)

# DST ablation of Figure 1 (E4)
python q_speedup_ablation.py 10
```

Environment variables: `q_ccs_stochastic.py`: QDOMAIN (office|ds), QWIND, QGAMMA, QEPS (OLS eps), QMAX (step cap),
QBASISWARM, QSTABK, QMC (completion samples), QCERT (certificate tolerance), ARMS (subset of standard,
standard-warm, certify, certify-mf). `q_fsa_stochastic.py`: QDOM (office|windy), QSLIP, QBASE (base steps), QBUDGET,
QSTABLE, QSEEDS. `q_lazy_fsa.py`: QBUDGET, QSTABLE (plus the q_fsa_stochastic ones).

### Cluster workflow (FRIDA, since 2026-10-25) — see `cluster/README.md`
```bash
bash cluster/setup.sh                                            # env sffsa + RM/sf-fsa-vi (gitlink, cloned)
PY=~/miniconda3/envs/sffsa/bin/python
$PY cluster/sweep.py submit cluster/sweeps/smoke.yaml            # every entry point, ~1 min
$PY cluster/sweep.py submit cluster/sweeps/lazy_mf.yaml          # E10 (60 runs) and lazy_mf_ds.yaml (20)
$PY cluster/sweep.py status [SWEEP_DIR]                          # done / failed / lost / running / queued
$PY cluster/sweep.py merge SWEEP_DIR --into DIR [--only i,j]     # gather outputs (list pickles concatenated)
bash cluster/tex.sh --run 'cd SFOLS && python plot_lazy_mf.py'   # figures with TeX (TeX Live container)
bash cluster/tex.sh paper article                                # paper/article.pdf (pdflatex + biber)
```
Every run of a sweep has its own directory `SFOLS/runs/<sweep>-<ts>/r/<id>/` (log, status, outputs via `SF_OUT`) and
runs from a code snapshot. Final outputs were merged into `SFOLS/results/q_ccs_10seeds/` (E5) and
`SFOLS/results/lazy_mf/inherit{0,1}/` (E10).

### Cluster notes (older)
- No GPU needed: tabular, sample-by-sample Python loops. Throughput scales with cores; per-job speed needs numba (the
  inner loops `sfql`, `learn`, `q_learn`, `sf_learn` are plain Python over numpy rows; expect 10–50x with numba).
- Embarrassingly parallel units: (domain, arm, seed) for `q_ccs_stochastic.py` (use `ARMS` and one seed per job);
  domains for `certlib.lazyfsa`; problems x learners for `q_fsa_stochastic.py` (already a Pool); tasks for the fresh
  mode of `q_lazy_fsa.py` (not yet parallelised). Stream mode is sequential by design (each task reuses the base).
- Suggested SLURM pattern: one array task per (domain, arm, seed), `--cpus-per-task` = workers of the script,
  `OMP_NUM_THREADS=1`, results written to `SFOLS/results/` (file names include the configuration tag).

## 10. File map (main pieces)

| path | what |
|---|---|
| `SFOLS/certlib/mdp.py` | tabular MOMDP: SF backups, iterative evaluation, gaps, reach bound, policy iteration |
| `SFOLS/certlib/library.py` | SF library (stack of action-SF tables), GPI |
| `SFOLS/certlib/domains.py`, `SFOLS/gpi_basis/domains.py` | exact models: double slit, slippery Office, windy two-stage |
| `SFOLS/certlib/experiments.py` | domain registry (`get`: office0f2/office1f2/office3f2/windy/ds), runs, tasks (`tasks_for`) |
| `SFOLS/gpi_basis/fsa.py` | FSA tasks (PAPER_TASKS, random_tasks, windy_tasks), acyclic planning |
| `SFOLS/certlib/hybrid.py` | GPI / metapolicy / pi_hyb, per-task certified planning, ladder, warm starts (acyclic) |
| `SFOLS/certlib/lazyfsa.py` | **Lazy SF-FSA-VI** (Kuric semantics), product-MDP oracle, baselines |
| `SFOLS/certlib/pertask.py`, `zeroshot.py` | older per-task certification and exit-value refinement |
| `SFOLS/q_ccs_stochastic.py` | model-free Figure 1 replicas (Office slip, double slit) |
| `SFOLS/q_fsa_stochastic.py` | model-free per-automaton-state learners, inherited counts; base learner `sf_learn` |
| `SFOLS/q_lazy_fsa.py` | model-free Lazy SF-FSA-VI (oracle certificate) |
| `SFOLS/q_speedup_ablation.py` | DST ablation explaining Figure 1 |
| `SFOLS/plot_*.py`, `SFOLS/certlib/figures_hyb.py` | figures (paper style from `plot_ccs_cost_combined.py`) |
| `SFOLS/13_*.ipynb`, `14_*.ipynb` | deterministic Figure 1 experiments of the paper (DST, Office) |
| `notes/gpi_basis.tex`, `notes/lazy_sf_fsa_vi.tex` | current notes (see §1) |
| `paper/article.tex`, `paper/figures/` | the paper |
| `fsa-rm-vi-directions.md` | the user's direction document (D1–D6) |
| `SFOLS/mf.py` | numba learners (vector SFQL, flat Q on the product MDP) and Gauss–Seidel oracle evaluators |
| `SFOLS/q_lazy_fsa.py` | **rewritten**: model-free Lazy SF-FSA-VI vs SF-FSA-VI over SFOLS, Q-learning growth, flat Q (E10) |
| `SFOLS/plot_ccs_cost_stochastic.py`, `SFOLS/plot_lazy_mf.py` | the two new paper figures |
| `cluster/` | `sweep.py`, `setup.sh`, `tex.sh`, `sweeps/*.yaml`, `README.md` |
| `notes/lazy_sf_fsa_vi.tex` | **the analysis note** of this session (CoG under noise, Lazy with and without a model) |
