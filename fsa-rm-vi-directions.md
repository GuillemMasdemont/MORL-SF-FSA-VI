# Stochastic FSA-RM-VI with metapolicies: analysis and directions

*Notes, October 2026. Builds on: Kuric et al. (ICAPS 2024, SF-FSA-VI); Sheebaelhamd et al. (NeurIPS 2026, MA-BC, Pareto Path Thm 2.1); Masdemont & Gómez draft (Certify-or-Grow).*

**TL;DR.** That the metapolicy is about as fast as PI is expected. The real speed-up for FSA-RM-VI comes from doing fewer low-level solves (cache + lazy certification, D1–D2). The only computational edge specific to the metapolicy is that $M_B$ is weight-independent and can be preprocessed once (D5). This needs $|D|\ll|\mathcal S|$; otherwise motivate the metapolicy via large/continuous action spaces.

---

## 0. Notation

- Low-level family $\mathcal M^\phi$:
  - states $\mathcal S$, actions $\mathcal A$, kernel $P$, discount $\gamma$;
  - features $\phi(s,a,s')\in\mathbb R^d$, with $d=|\mathcal E|$ exits;
  - task $w\in\Delta(d)$, reward $w^\top\phi$.
- SFs $\psi^\pi(s,a)=\mathbb E_\pi\big[\sum_t\gamma^t\phi_t\mid S_0=s,A_0=a\big]$, so $Q^\pi_w=w^\top\psi^\pi$. For deterministic $\pi$, $\psi^\pi(s):=\psi^\pi(s,\pi(s))$.
- Basis $B=\{\pi_1,\dots,\pi_k\}$ and $B(s)=\{\pi_j(s)\}_{j\le k}$. $M_B$ is the MDP restricted to actions in $B(s)$, and $\Pi_B=\prod_sB(s)$ is its policy set. $\pi_{\text{meta}}$ is the optimal policy of $M_B$ for $w$, with value $V_{\text{meta}}$.
- $Q^{\text{meta}}(s,a)=w^\top\psi^{\pi_{\text{meta}}}(s,a)$ for all $a\in\mathcal A$ (one backup for $a\notin B(s)$). The hybrid policy is $\pi^{\text{hyb}}(s)=\arg\max_{a\in\mathcal A}Q^{\text{meta}}(s,a)$.
- Gap $g(s)=\max_aQ^{\text{meta}}(s,a)-V_{\text{meta}}(s)$. If $\max_sg(s)\le\varepsilon$, then $\pi_{\text{meta}}$ is $\varepsilon/(1-\gamma)$-optimal.
- SF-FSA-VI: FSA states $u\in\mathcal U$ and exit weights $W(u)\in\mathbb R^d$, with update
  $$w^{k+1}_j(u)=\max_{a,\pi}w^k\big(\tau(u,\mathcal O(\varepsilon_j))\big)^\top\psi^\pi(\varepsilon_j,a).$$

## 1. Analysis

1. **Deterministic domains.** With exit features, the optimal action at any $w$ is the action of the corner policy for the best exit from $s$. Hence $\pi^*_w\in\Pi_B$, and GPI over the $d$ corner policies is optimal (closed form).
2. **Stochastic domains.** At an intermediate $w$, the optimum can use an action that no base policy takes at $s$. In the double slit this is RIGHT, used to hedge between exits. Then $\pi^*_w\notin\Pi_B$ and the metapolicy fails.
3. **Pareto Path (MA-BC Thm 2.1).** Neighbouring Pareto-front policies are linked by single-state flips, and all of them are optimal at the corner weight between them. So the missing actions enter one state at a time as $w$ moves.
4. **Why the metapolicy is about as fast as PI (as observed).**
   - $\pi^{\text{hyb}}$ is one PI step from $\pi_{\text{meta}}$, and computing $\pi_{\text{meta}}$ is itself a solve of $M_B$ over all of $\mathcal S$.
   - Both are dominated by policy evaluation over $\mathcal S$. Restricting the max from $|\mathcal A|=3$–$4$ to $|B(s)|=2$–$3$ saves little, and $\pi^{\text{hyb}}$ and the certificate touch all of $\mathcal A$ anyway.
   - So the speed-up of both most likely comes from warm-starting with the basis.
   - Model-free, PI needs $Q$ for all actions at every improvement step, while the metapolicy needs it once. The gain is roughly the number of PI iterations saved, which is small from a good warm start.
5. **Correctness of the current plan** (metapolicy → $\pi^{\text{hyb}}$ → propagate). Without a certificate, the propagated value is only a lower bound on $V^*$. The high-level VI can then converge to a suboptimal fixed point. To fix it:
   - check $g$, which is free (same table as $\pi^{\text{hyb}}$);
   - if the check fails, evaluate $\pi^{\text{hyb}}$ and check its own gap, i.e. continue PI until certified.
6. **Certify-or-Grow.** The metapolicy is recomputed from $B$ at every $w$. The from-scratch solve does not maintain it. It keeps the answer exact and enlarges $B$ for later weights.

## 2. Tools

**Optimality region.** With $\Delta\psi^\pi(s,a)=\psi^\pi(s,a)-\psi^\pi(s,\pi(s))$, define
$$R_\pi=\{w\in\Delta(d):\ w^\top\Delta\psi^\pi(s,a)\le0\ \ \forall s,a\}.$$
- If $w\in R_\pi$, then $\pi$ is optimal for $w$ at every state (policy improvement). $R_\pi$ is a polytope, costs no samples and uses the same table as the certificate.
- $\varepsilon$-version: if $\max_{s,a}w^\top\Delta\psi^\pi(s,a)\le\varepsilon$, then $\pi$ is $\varepsilon/(1-\gamma)$-optimal.
- Facets are flips. Crossing the facet $\{w^\top\Delta\psi^\pi(s,a)=0\}$ gives (generic case) the neighbouring policy $\pi$ with $\pi(s)\leftarrow a$. Redundant constraints correspond to non-neighbours.
- Keeping only states reachable (under any policy) from $\mathrm{supp}(\mathbb P_0)$ certifies optimality from those states only. That is enough for SF-FSA-VI and gives fewer cells.

**Flip lemma.** If $\pi'=\pi$ except $\pi'(s)=a$, then for all $x$
$$\psi^{\pi'}(x)=\psi^\pi(x)+d^{\pi'}_x(s)\,\Delta\psi^\pi(s,a),\qquad d^{\pi'}_x(s)=\mathbb E_{\pi'}\Big[\textstyle\sum_t\gamma^t\mathbf 1\{X_t=s\}\,\Big|\,X_0=x\Big].$$

Proof: the vector performance-difference lemma, where only $s$ contributes. Consequences:
- Only states that can reach $s$ change.
- Returns move along $\Delta\psi^\pi(s,a)$, which is orthogonal to the breakpoint weight.
- With a model, each flip is a rank-one (Sherman–Morrison) update.

**Worked example** (MOMDP of MA-BC Thm 2.2 with $\alpha=0.1$).
- One absorbing state $x$ with $r(a_1)=[1,0]$, $r(a_2)=[0,1]$, $r(a_3)=[0.6,0.6]$.
- Weights $w=(1-\lambda,\lambda)$, and $\Delta\psi^\pi(x,a)=r(a)-r(\pi(x))$.

| $\pi$ | constraints $w^\top\Delta\psi\le0$ | $R_\pi$ |
|---|---|---|
| $a_1$ | $a_3$: $\lambda\le0.4$; $a_2$: $\lambda\le0.5$ (redundant) | $[0,0.4]$ |
| $a_3$ | $a_1$: $\lambda\ge0.4$; $a_2$: $\lambda\le0.6$ | $[0.4,0.6]$ |
| $a_2$ | $a_1$: $\lambda\ge0.5$ (redundant); $a_3$: $\lambda\ge0.6$ | $[0.6,1]$ |

- Traversal needs 3 evaluations. OLS needs 5 solves ($\lambda=0,1,0.5,0.4,0.6$), and the last two only certify.
- The metapolicy over $\{a_1,a_2\}$ at $\lambda=0.5$ fails because $a_3\notin B(x)$. The gap $g=0.1$ sits exactly at $(x,a_3)$.

## 3. Directions (ordered by risk)

### D1: Certify + cache inside FSA-RM-VI (low risk, solver-agnostic)
Before any low-level solve at $w$, check whether some stored $\pi$ has $\max_{s,a}w^\top\Delta\psi^\pi(s,a)\le\varepsilon$. If so, reuse it; this costs one matrix–vector product per stored policy. The high-level weights converge, so solves should stop once the weights stay inside known regions.

Related work: policy caches with SFs (Nemecek & Parr 2021), here with an exact test.

### D2: Lazy CCS for SF-FSA-VI (close to the current plan, made exact)
1. Plan with basis $\Pi$ to get $W_\Pi$.
2. Certify every $W_\Pi(u)$, using D1 or the metapolicy gap, and add each certifying policy to $\Pi$. Grow where certification fails, then replan.
3. Stop when every $W_\Pi(u)$ is certified by a policy already in $\Pi$.

Why this is optimal:
- At the stop, $T_\Pi W_\Pi=T^*W_\Pi$, and $W_\Pi=T_\Pi W_\Pi$, so $W_\Pi=T^*W_\Pi$.
- $T^*$ is Eq. 11 with the true $Q^*$. It is a contraction because $\|\psi(\varepsilon_j,a)\|_1<1$, so $W_\Pi=W^*$.

Example: in the double slit with the task "blue or red", $W(u_0)=(1,1)$. You need one policy optimal at $(0.5,0.5)$ instead of the 181-policy CCS. The price is that this is not pure zero-shot.

### D3: Local grow (fixes the failing domains; no speed gain vs PI)
At failing states, set $B(s)\leftarrow B(s)\cup\{a:Q_{V_{\text{meta}}}(s,a)>V_{\text{meta}}(s)+\varepsilon\}$, rerun Algorithm 2, and repeat.

$V$ strictly improves each round and $B$ only grows, so this terminates at an $\varepsilon/(1-\gamma)$-optimal policy. It is essentially PI warm-started at $\pi_{\text{meta}}$, so expect PI's speed, which matches the experiments.

### D4: CCS traversal instead of OLS (medium risk; start with $d=2$)
1. Start at a corner policy and compute $R_\pi$.
2. Cross each unexplored facet by a flip, and evaluate the new policy incrementally with the flip lemma.
3. Hash policies to avoid duplicates.
4. Stop when the regions tile $\Delta(d)$.

Notes:
- **2D.** The pair $(s,a)$ switches at $\lambda_{s,a}=\Delta_1/(\Delta_1-\Delta_2)$. The next policy is the flip with the smallest $\lambda_{s,a}$ above the current $\lambda$.
- **Degenerate facets** (several ties at once, as with the aliasing in DST). Solve the restricted MDP with only the tied actions, at a weight just past the facet; this is exactly the metapolicy.
- **Vs OLS.** Every evaluation is a new cell, whereas OLS needs about $2|\text{CCS}|-1$ solves in 2D. Completeness is certified by the tiling.
- **Prior art.** Pareto-front traversal (Li, Ju & Shroff 2024) flips and checks with a known model. Here the valid flips are read from one SF table.

### D5: Compressed metapolicy (the only metapolicy-specific edge; higher risk)
$\Pi_B$ forces the action wherever the base policies agree, for every $w$. So $M_B$ can be preprocessed once ("split the differences, pool the rest").

- Let $D=\{s:|B(s)|\ge2\}$ be the divergent states and $C=\mathcal S\setminus D$. The dynamics on $C$ do not depend on $w$.
- Precompute once, with one linear solve on $C$. Let $\tau_D$ be the hitting time of $D$; on $D$, $\tau_D=0$, so $\Phi=0$ and $H(x,\cdot)=e_x$.
$$\Phi(x)=\mathbb E\Big[\textstyle\sum_{t<\tau_D}\gamma^t\phi_t\,\Big|\,X_0=x\Big],\qquad H(x,y)=\mathbb E\big[\gamma^{\tau_D}\mathbf 1\{X_{\tau_D}=y\}\,\big|\,X_0=x\big],\ y\in D.$$
- Per $w$, solve the $|D|$-state MDP with actions $b\in B(y)$:
$$\Psi(y)=\mathbb E_{y'\sim P(\cdot|y,b)}\Big[\phi(y,b,y')+\gamma\Big(\Phi(y')+\textstyle\sum_{y''}H(y',y'')\Psi(y'')\Big)\Big].$$
  Then $\psi(x)=\Phi(x)+\sum_yH(x,y)\Psi(y)$ everywhere, and one sweep over $\mathcal S\times\mathcal A$ gives $\pi^{\text{hyb}}$ and the certificate.
- Example: in the MA-BC Y-maze, the 20-state stem collapses to $H(x,\text{fork})=(0.9\gamma)^{n(x)}$, where $n(x)$ is the number of steps to the fork.
- When it pays: $|D|\ll|\mathcal S|$ and many queries per basis.
  - With corner bases in open grids, $D\approx\mathcal S$, so there is no gain.
  - Instead, use the two stored policies adjacent to $w$. By the Pareto path, close policies differ at few states.
- Risk: in stochastic domains the base-action metapolicy may often fail the certificate.
  - In the draft's Fig. 2b, it is still suboptimal on 2.5% of weights with 20 policies, while $\pi^{\text{hyb}}$ is optimal everywhere with 9.
  - When it fails, fall back to PI; the cache then densifies.

### D6: Other motivations for the metapolicy
- **Large or continuous $\mathcal A$.** The metapolicy maximizes over $k$ proposals, while PI must maximize over $\mathcal A$. Only the certificate needs $\max_{a\in\mathcal A}$: use a local search, or the closed form for quadratic $Q$ (e.g. LQR).
- **Sample-efficient CCS construction** (model-free). It already gives 3.2× fewer steps in DST and 2.1× in Office vs warm-started SFOLS.

## 4. Next experiments

1. **Double slit, exact model.** Run the 2D sweep (D4). Check that consecutive CCS policies differ at one state, and count evaluations vs OLS solves.
2. **Is D5 viable?** For adjacent stored pairs in the double slit and Office, measure $|D|/|\mathcal S|$ and the fraction of weights where the pair-metapolicy passes the certificate.
3. **Lazy SF-FSA-VI** (D1 + D2) on the double-slit FSA. Count low-level solves with and without caching, and check optimality.
4. *(optional)* Scale $|\mathcal A|$ (e.g. 8 directions × step sizes) to test D6.

**Decision.** If $|D|$ is small and certification is frequent, motivate the metapolicy through D5. Otherwise:
- keep the metapolicy for CCS construction, where it wins in samples;
- use PI with D1/D2 inside FSA-RM-VI;
- motivate the metapolicy through large or continuous action spaces.

## 5. Caveats

- $R_\pi$ needs $\psi^\pi(s,a)$ for all actions, which the certificate already pays for. With learned SFs the breakpoints are noisy, so use $\varepsilon$-regions and re-certify.
- State-wise cells can outnumber the $\nu_0$-CCS, and their number can grow fast with $d$. Use exact traversal for small $d$ and D2 otherwise.
- Model-free, a flip still needs a TD evaluation upstream of $s$. The sample gain vs SF Q-learning has to be measured.

## References

- Kuric, Infante, Gómez, Jonsson, van Hoof. *Planning with a Learned Policy Basis to Optimally Solve Complex Tasks.* ICAPS 2024.
- Sheebaelhamd, Viano, Cevher, Vernade. *Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation.* NeurIPS 2026, arXiv:2605.12000.
- Li, Ju, Shroff. *How to Find the Exact Pareto Front for Multi-Objective MDPs?* arXiv:2410.15557, 2024.
- Alegre, Bazzan, da Silva. *Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer.* ICML 2022.
- Nemecek, Parr. *Policy Caches with Successor Features.* ICML 2021.
- Masdemont, Gómez. *Faster Convex Coverage Sets via Metapolicies over Base Policies.* Draft, Oct 2026.
