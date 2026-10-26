# Certify-or-Grow under noise: where the samples go, and what to try next

Observations from the stochastic CCS experiments behind the paper's Figure 2 (2026-10-26). Setup as in `HANDOFF.md`
(E5): SF Q-learning with step 1/(n+1)^0.7, eps-greedy 0.1, exploring starts; a learn stops when its greedy policy is
unchanged for 30 checks of 1000 steps and every tracked pair has 30 visits, or at the cap (3M steps in Office, 5M in the
double slit); certificate tolerance 1e-2; 10 seeds. Office: slip 0.05, eps-CCS with eps = 0.05. Double slit: wind +-1,
gamma 0.99, eps = 1e-3.

Data and code: `SFOLS/results/q_ccs_10seeds/` (E5), `SFOLS/results/q_ccs_cog2/` (E5b, E5c),
`SFOLS/results/q_ccs_cover/` (instrumented runs); `SFOLS/analyze_ccs_speed.py` prints every number below,
`SFOLS/plot_ccs_breakdown.py` draws `notes/figures/ccs_cog_breakdown.pdf`.

## 1. The two metrics measure different things

- **Steps to converge** (oracle): the last check at which the greedy policy was more than 1e-3 below the optimum of the
  learner's own problem (full MDP, or the MDP restricted to B(s) for a metapolicy). It measures how good the warm start
  is and how fast the greedy policy settles. It needs the model, so no learner can stop there.
- **Environment steps**: what the learner actually spends, with its own stopping rule. Under noise this is dominated by
  the rule "greedy policy unchanged for 30 checks": near-tied actions keep flipping, so a learn over all 4 actions often
  never stabilizes and runs to the cap (double slit: 100% of SFOLS's learns; Office: 10-15 learns per run).
- Hence a method can win on one metric and not on the other. In the double slit, warm-started SFOLS already gets an
  optimal greedy policy almost immediately after the 2 extremes (0.08M steps to converge per query), so CoG has nothing
  left to gain there; its gain is that a learn restricted to B(s) (2 actions per state) stabilizes and stops (1.2M
  steps), while a learn over 4 actions runs to the 5M cap. The environment-steps comparison is therefore partly set by
  the arbitrary cap.

## 2. Where CoG lost its samples in the first version (E5)

Steps to converge, Office, mean over 10 seeds (`analyze_ccs_speed.py`):

| | learns per run | total | per learn |
|---|---|---|---|
| warm SFOLS, every query after the extremes | 92.4 | 122.8M | 1.33M |
| CoG, certified metapolicies | 75.6 | 75.3M | 1.00M |
| CoG, OLS's re-queried extremes (relearned as metapolicies) | 6.0 | 8.1M | 1.35M |
| CoG, failed metapolicy learns | 14.9 | 21.4M | 1.44M |
| CoG, full learns after a failed certificate (cold start) | 17.5 | 32.0M | 1.83M |

- A certified metapolicy is 25% cheaper than a warm full learn (13% in environment steps), but the overheads (61.5M)
  ate the saving (25M): CoG needed 147.4M against 133.4M.
- **Re-queried extremes.** OLS's first d queries are the extremes, which CoG had already learned as its basis. Pure
  waste: the answer is the basis policy. (The deterministic notebook 14 does the same, but there it only costs a
  confirmation pass, ~16k environment steps per query and 0 steps to converge.)
- **Fallbacks paid twice**: the failed metapolicy learn, and then a full learn from the optimistic start, unlike every
  learn of warm SFOLS and CoG's own warm-started basis.
- In the double slit the only fallbacks were at re-queried extremes (2 of 10 seeds): the whole 4.5 vs 4.2M gap.

## 3. The fixes (E5b, paper Figure 2)

`QREUSE=1` answers the re-queried extremes with the basis. `QFALLBACK` sets the full learn after a failed certificate:
`gpi` = GPI warm start over every table learned so far (the failed metapolicy's included), `continue` = the metapolicy
learn goes on over all actions with its table and visit counts (entries outside B(s) from GPI over the basis and pool,
or from the completion samples in the model-free version). `continue` is the warm start with pi_hyb that the paper
already proposes (Sec. "Combining the metapolicy with GPI"), and the same design as the Lazy SF-FSA-VI experiment.

| steps to converge (environment steps) | Office slip 0.05 | double slit |
|---|---|---|
| SFOLS + warm start | 133.4 +- 5.4M (193.1M) | 4.24 +- 0.66M (30.6M) |
| CoG, E5 | 147.4 +- 12.6M (206.5M) | 4.49 +- 1.36M (21.3M) |
| + reuse | 133.6M (187.2M) | 3.91M (12.95M) |
| + reuse + gpi fallback | 128.3M (182.1M) | same (no fallbacks) |
| **+ reuse + continue** | **113.8 +- 4.7M (164.9M)** | **3.91 +- 0.69M (12.95M)** |
| CoG model-free, E5 -> reuse + continue | 120.9 -> 102.5M (176.6 -> 150.4M) | 4.94 -> 4.15M (22.5 -> 13.8M) |

- Continuing makes a fallback cost 0.88M instead of 1.83M (cold) or 1.49M (GPI restart): the restricted learn is not
  wasted, its table is already right on B(s), and its visit counts keep the noisy first updates from overwriting it.
  Restarting with GPI and fresh counts barely helps, because the first update (step 0.62) overwrites the warm start.
- Certified learns also got cheaper (1.00 -> 0.89M) and the variance dropped (std 12.6 -> 4.7M): the fallback policies
  that join the basis are better learned, so later warm starts are better.
- Same CCS quality: worst eps-gap over 300 weights 0.0052-0.0063 (warm SFOLS 0.0052-0.0062). The default flags
  reproduce E5 exactly (checked on double slit seed 0), so the SFOLS arms were not rerun.
- Learns at the step cap per run: warm SFOLS 14.2, CoG 3.7.

## 4. What is left in Office

- **Failed metapolicy learns are now the largest overhead** (16.9 per run, 21.6M of 113.8M). 97% of the fallbacks
  happen in the first half of the OLS queries, while the basis is small; afterwards almost every query is certified.
- **The restriction is mild and does not grow.** |B(s)| is 2.9 of 4 actions with the 6 extremes and only 3.0 with
  26 basis policies (instrumented runs, `office_ccs_cover.yaml`). So B(s) does not fill up as noisy policies join the
  basis; the metapolicy learn is cheaper only because about a quarter of the pairs are gone.
- **Failed certificates are mostly real.** 19-22% of the learned metapolicies have a true gap above 1e-2, and only
  15-29% of those lose at most 1e-3 at the start states (median loss 2.7e-3). A certificate weighted by the start
  distribution would therefore save few fallbacks. Conversely, 5-7% pass the gap test but lose more than 1e-3 at the
  start (the tolerance 1e-2 on the gap is loose).
- **Model-free certificate.** 15% of its certificates are false (true gap > 1e-2), with a typical start loss of 3e-3.
  Here that is far below the OLS eps (0.05), so the CCS is as good, but with a smaller eps it would matter.
- **Pseudo-visits for warm starts (E5c, `office_ccs_n0.yaml`)**: counting every warm start as 10 prior visits helps
  warm SFOLS (133.4 -> 126.5M) more than CoG (113.8 -> 112.2M, larger variance), so the gap shrinks from 15% to 11%.
  Not a lever for CoG.

## 5. Lazy SF-FSA-VI without a model (paper Figure 5)

The two CCS fixes do not apply here: Lazy never runs OLS (no re-queried extremes), and its CoG growth already continued
the metapolicy learn over all actions when the certificate failed. `SFOLS/analyze_lazy_growth.py` compares the two ways
of growing a policy (metapolicy, Q-learning over all actions), both from the GPI table without inherited counts.

- **The overhead was the decision, not the learning.** CoG learned the metapolicy until its policy had been optimal in
  the restricted MDP for 100 checks of 1000 steps, then decided; these 100k steps were charged to every new policy,
  while Q-learning's confirmation is never charged (its cost stops at its last failed check). With a decision after 10
  checks (`QSTABLE1=10`, sweep `lazy_cog_k.yaml`, E10b) CoG costs 0.81-1.06x Q-learning's environment steps on a new
  task (E10: 0.98-1.40x). A wrong early decision costs little: an early certificate is still an (oracle) certificate of
  optimality, and an early fallback keeps the table and visit counts. 1 or 30 checks give 0.80-1.10x (no clear trend),
  same mean regret; in windy one task in 600 hit the 5-round cap with 0.74% regret (p99 unchanged, 0.28%).
- **Paired by weight** (first-round growths happen at the same w(u) for both methods): a certified metapolicy is
  18-40% cheaper than Q-learning (windy 3.15 vs 4.00M, Office p=0.3 0.27 vs 0.45M); a failed one costs about as much
  (windy 0.68 vs 0.71M, Office p=0.3 0.56 vs 0.53M). Certified: 8-17% of the new policies from the extremes, 28-50%
  with kept policies. So CoG cannot beat Q-learning by much here: the policies Lazy needs are mostly new behaviour
  (windy: the hedging policy at u0 differs from the nearest stored one at ~220 of 611 states), not compositions.
- Q-learning's numbers are reproduced exactly by the new runs (same seeds); the windy certified metapolicies are expensive
  (2-6M) because those weights are hard, not because of the restriction (Q-learning needs 4.0M at the same weights).

Figure 5 now shows E10d (below); `COGSRC=k10` / `COGSRC=inherit0 python plot_lazy_mf.py` redraw E10b / E10.

**Why the CCS gain does not carry over** (expected gain ~ share of new policies that are compositions x saving per
composed policy). The saving per certified metapolicy is similar in both settings (CCS Office: 33% per learn; Lazy:
18-40% at the same weight), but about 80% of the CCS queries are certified and only 8-17% of the policies a fresh Lazy
task needs (28-50% with kept policies): OLS's corner weights interpolate between learned policies, while Lazy's w(u)
ask for new behaviour (hedging, risk-aware routes under p = 0.3). Moreover, part of the CCS gain in environment steps
came from the "no greedy change" stopping rule, which restricted learns satisfy sooner; Lazy's learners stop by the
oracle, so that part does not exist there. And in Figure 5 the extremes are 94-99% of every Lazy curve on the right,
and a 10-20% difference is a 0.05-0.1 decade shift on the left. In the double slit no metapolicy is ever certified, yet
CoG is 20% cheaper: the short restricted learn (0.06M) gives the continued learn a better start.

**Algorithm 4's middle rung, pi_hyb (E10c, exploratory, not in the paper).** The model-free CoG growth skips pi_hyb,
which with exact SFs certifies 17-61% of the added policies (Figure 4c; in windy it is the hedging policy). Sweeps
`lazy_cog_hyb.yaml`, `lazy_cog_hybn.yaml`, `lazy_cog_hyblocal.yaml` (`QHYB=1`: complete the metapolicy's table outside
B(s) with 100 sampled backups per pair, counted, and check its greedy policy pi_hyb with the oracle).
- pi_hyb is certified for 22-36% of the growths whose metapolicy fails (windy 93 of 431, Office p=0.3 138 of 383),
  but these growths are not cheaper than Q-learning at the same weight (windy 0.49 vs 0.40M): the learn that follows
  over all actions treats the completed entries as unvisited, so the first noisy updates flip the optimal policy again.
  Per new task: windy 1.03x, Office 1.06x / 1.00x, double slit 0.97x.
- Counting the completed entries as their 100 samples (`QHYBN=1`), and refining a certified pi_hyb on B(s) plus its
  own actions (`QHYBLOCAL=1`): fresh tasks 0.89-0.96x Q-learning in windy, Office p=0.3 and the double slit (Office
  p=0.1 1.04-1.08x). But the trusted entries are biased (they estimate the metapolicy's values) and with step sizes of
  1/101^0.7 barely move: in windy seed 2 the kept library degraded into 36 growths and 121M steps (other seeds 1-4
  growths, 0.5-10M). Not robust; a confidence-aware count (e.g. counts that decay when the policy changes) would be
  needed.

## 6. Ideas for future research (most promising first)

1. **Decide the fallback earlier in the CCS loop too.** With `continue`, falling back loses nothing (the full learn
   starts from the same table and counts), so the metapolicy learn can stop as soon as it looks insufficient instead of
   after its full stopping rule. In Lazy this (10 checks instead of 100) removed CoG's overhead (section 5); in the CCS
   experiment the metapolicy learn still runs to `q_ccs_stochastic`'s stopping rule (30 stable checks), so the 21.6M of
   failed metapolicy learns in Office are the target.
2. **Grow B(s) locally instead of opening all actions.** When the certificate fails, add to B(s) only the actions with a
   positive gap (pi_hyb's actions) at the violating states and continue the restricted learn. Each round is a policy
   improvement step, so it converges like policy iteration, and the learn keeps the fast stabilization of a restricted
   action set.
3. **A sound model-free certificate**, e.g. confidence bounds on the gap from the visit counts. Needed both here (15%
   false certificates) and in Lazy SF-FSA-VI, which uses an oracle.
4. **A noise-aware stopping rule** (confidence intervals of the action gaps instead of "no greedy change for 30
   checks"), so that the environment-steps comparison is not set by the cap. It would likely favour restricted learns,
   which have fewer pairs to settle.
5. Deterministic Figure 1 code (notebook 14) still relearns the extremes as metapolicies; reusing the basis would
   remove ~0.1M environment steps of confirmation in Office (0 steps to converge). Not worth rerunning for the paper.

**E10d: Algorithm 4's ladder without a model, decided before running (now in Figure 5).** Sweep
`lazy_cog_hybc.yaml`, seeds 0-19 (10-19 never used to choose a variant): learn the metapolicy on B(s), decide after 10
checks; if it is not certified, complete its table and check pi_hyb; a certified pi_hyb is refined on B(s) plus its own
actions with its completed entries counted as their 100 samples (they belong to the policy being refined); otherwise
continue over all actions from the completed table without trusting it (`QHYBN=2 QHYBLOCAL=1`).
- Per new task: 0.97-1.03x Q-learning on seeds 0-9 and 0.93-1.06x on the held-out seeds; the per-seed ratio varies by
  +-10-30%, more than any systematic difference. No cascade in the kept libraries (1-17 growths per seed); windy back
  to every task within 0.4% (max 0.30%).
- By how the new policy was obtained (first-round growths at the same weight, environment steps, seeds 0-9 / 10-19):
  certified metapolicy -21% / -22% (windy), -32% / -43% (Office p=0.3); certified pi_hyb -2% / -3% (windy), -22% /
  -17% (Office p=0.3), -4% / -6% (Office p=0.1); learned over all actions +6% / +20% (windy), +3% / -7% (Office p=0.3).
  This is the visible and robust improvement, shown in Figure 5 (c, f); compositions are 32% (windy), 26% (Office
  p=0.1) and 49% (Office p=0.3) of the new policies, so they do not move the per-task cost.
- To make the per-task cost visibly lower, the fallbacks would have to stop paying the metapolicy learning (deciding
  even earlier, or predicting from the GPI table that the metapolicy will fail), or the completed entries would need a
  confidence-aware count (trusting them always gave 0.89-0.96x per task but one cascading seed).

