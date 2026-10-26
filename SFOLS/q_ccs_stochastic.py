"""Figure 1 of the paper (learned SFs on Office) on the STOCHASTIC Office: the chosen action is replaced by one of
the other three w.p. SLIP. Same OLS loop, arms and exploring starts as notebook 14; no learner uses the model.

What stochasticity forces to change (everything else as in notebook 14):
  step size    alpha = 1 / (n(s,a) + 1)^KAPPA (n = visits in this learn), or a constant ALPHA, instead of 1
  init         optimistic init Q = max(w) (Psi = e_argmax(w)) instead of Psi = 1 (Q = sum(w))
  stopping     every CHECK steps: stop once the greedy policy has not changed for STABLE consecutive checks and
               every tracked (s,a) has >= NMIN visits (the 'no SF change > 1e-6' rule never fires under noise)
  completion   the model-free certificate fills each (s,a) outside B(s) with M_COMPLETE sampled transitions
               (one sample is exact only in a deterministic domain), and certifies when the gap <= CERT_TOL
The model is used ONLY to measure: 'steps to converge' (oracle) = the last check at which the greedy policy was
more than DELTA below the optimum of the learner's own problem (full MDP, or the MDP restricted to B(s)).

Double slit (Kuric et al.): QDOMAIN=ds [QWIND=1] [QGAMMA=0.99] [QEPS=0.05]. The wind (uniform +-QWIND rows per step)
is the only noise, so the <slip> argument is ignored; exploring starts and the stopping rule skip the two exits.

Certify-or-Grow options (defaults = the E5 runs): QBASISWARM=1 warm-starts the basis like SFOLS + warm start; QREUSE=1
answers OLS's re-queried extremes with the basis; QFALLBACK=gpi|continue warm-starts the full learn after a failed
certificate (see CertifyOrGrowQModelFree._grow).

usage: python q_ccs_stochastic.py pilot <slip>          one learn per weight, to calibrate
       python q_ccs_stochastic.py run <slip> <seeds>    all arms x seeds (multiprocessing), cached in results/
       python q_ccs_stochastic.py exact <slip>          exact OLS (model, no learning): size of the eps-CCS
"""
import heapq
import itertools
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import cdd
import numpy as np
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "RM", "sf-fsa-vi"))
from envs.grid_envs import DoubleSlit, Office  # noqa: E402

DOMAIN = os.environ.get("QDOMAIN", "office")
WIND = int(os.environ.get("QWIND", 1))
GAMMA, OLS_EPS = float(os.environ.get("QGAMMA", 0.99)), float(os.environ.get("QEPS", 0.05))
KAPPA, ALPHA, Q_EPS, Q_HORIZON = 0.7, None, 0.1, 100      # ALPHA set -> constant step size (Kuric: 0.3)
CHECK, STABLE, NMIN, MAX_STEPS = 1000, 30, 30, int(float(os.environ.get("QMAX", 3_000_000)))
BASIS_WARM = os.environ.get("QBASISWARM", "0") == "1"   # CoG learns the 2nd..d-th extreme warm-started (as SFOLS+warm)
REUSE = os.environ.get("QREUSE", "0") == "1"   # CoG answers OLS's re-queried extremes with its basis policies (no learn)
FALLBACK = os.environ.get("QFALLBACK", "cold")  # CoG's full learn after a failed certificate: cold | gpi | continue (_grow)
STAB_K = int(os.environ.get("QSTABK", 0))   # a check counts as stable if <= STAB_K greedy actions changed (0 = none)
DELTA = 1e-3
M_COMPLETE, CERT_TOL = int(os.environ.get("QMC", 100)), float(os.environ.get("QCERT", 1e-2))
SEED0 = int(os.environ.get("QSEED0", 0))   # first seed of `run` (split seeds across jobs)
OUT_DIR = os.environ.get("SF_OUT") or os.path.join(ROOT, "results")   # where outputs go (cluster/sweep.py sets it)
N0 = 0                      # warm starts count as N0 prior visits (step size starts at 1/(N0+2)^KAPPA); 0 = none

office = (Office(add_obj_to_start=True, random_act_prob=0.0) if DOMAIN == "office" else DoubleSlit(max_wind=WIND))
A_N, D = office.a_dim, office.feat_dim
states = [office.state_to_coords[i] for i in range(office.s_dim)]
sidx = {s: i for i, s in enumerate(states)}
S = len(states)
INIT = np.array([sidx[s] for s in office.initial])
NONTERM = np.array([DOMAIN == "office" or s not in office.object_ids for s in states])
NEXT = np.zeros((S, A_N), dtype=int)
PHI = np.zeros((S, A_N, D))
DONE = np.zeros((S, A_N), dtype=bool)
WOUT = [[None] * A_N for _ in range(S)]        # double slit: per (s, a) cumulative probs, outcomes (ns, done, phi)
for s in states:
    for a in range(A_N):
        if DOMAIN == "office":
            ns = office.base_movement(s, a)
            NEXT[sidx[s], a], PHI[sidx[s], a], DONE[sidx[s], a] = sidx[ns], office.features(s, a, ns), office.is_done(s, a, ns)
        elif NONTERM[sidx[s]]:
            outs = [(p, sidx[ns], ns in office.object_ids, office.features(s, a, ns))
                    for ns, p in office.coords_act_transition_distr(s, a)]
            WOUT[sidx[s]][a] = (np.cumsum([o[0] for o in outs]).tolist(), [o[1:] for o in outs])
ROWS = np.arange(S)
FULL = np.ones((S, A_N), dtype=bool)
SLIP = 0.0                                                    # set in main / worker


def slip_matrix():
    return np.where(np.eye(A_N, dtype=bool), 1 - SLIP, SLIP / (A_N - 1))   # pr[a, b]: chose a, executed b


# ---------------------------------------------------------------- exact model: MEASUREMENT ONLY (oracle)
def _model():
    if DOMAIN != "office":
        PHIx, Pn = np.zeros((S, A_N, D)), np.zeros((S, A_N, S))
        for i in np.flatnonzero(NONTERM):
            for a in range(A_N):
                cum, outs = WOUT[i][a]
                for p, (j, done, phi) in zip(np.diff([0.0] + cum), outs):
                    PHIx[i, a] += p * phi
                    if not done:
                        Pn[i, a, j] += p
        return PHIx, Pn
    pr = slip_matrix()
    PHIx = np.einsum("ab,sbd->sad", pr, PHI)                  # expected feature
    Pn = np.zeros((S, A_N, S))                                # non-terminal transitions
    for b in range(A_N):
        cont = ~DONE[:, b]
        for a in range(A_N):
            Pn[ROWS[cont], a, NEXT[cont, b]] += pr[a, b]
    return PHIx, Pn


def sf_state(actions, M):
    PHIx, Pn = M
    A = np.eye(S) - GAMMA * Pn[ROWS, actions]
    return np.linalg.solve(A, PHIx[ROWS, actions])


def q_backup(V, w, M):
    PHIx, Pn = M
    return PHIx @ w + GAMMA * Pn @ V


def policy_iteration(w, M, allowed=FULL):
    pi = np.where(allowed, 0, -1).argmax(1) * 0 + np.argmax(allowed, axis=1)
    for _ in range(1000):
        V = sf_state(pi, M) @ w
        Q = np.where(allowed, q_backup(V, w, M), -np.inf)
        best = Q.argmax(1)
        new = np.where(Q[ROWS, pi] >= Q[ROWS, best] - 1e-12, pi, best)
        if np.array_equal(new, pi):
            return pi, V
        pi = new
    raise RuntimeError("PI did not converge")


# ---------------------------------------------------------------- bookkeeping
class Profile:
    def __init__(self):
        self.n = defaultdict(int)
        self.log, self.log_total, self.learns = [], [], []
        self.flips = []                          # per learn: greedy-action changes at every check
        self.cover = []                          # per metapolicy learn: (query, basis size, mean |B(s)|)
        self.cert = []                           # per metapolicy learn (oracle): (query, true gap, loss at the start)
        self.query = 0
        self.logfile = None


PROF = Profile()


@dataclass
class Policy:
    w: np.ndarray
    actions: np.ndarray
    psi_s: np.ndarray
    psi_sa: Optional[np.ndarray]
    source: str

    @property
    def psi0(self):
        return self.psi_s[INIT].mean(axis=0)


# ---------------------------------------------------------------- OLS (notebook 14, unchanged)
def v_smp(psi_vecs, w):
    return max((v @ w for v in psi_vecs), default=-np.inf)


def corner_weights(psi_vecs):
    P = np.asarray(psi_vecs)
    n, m = P.shape
    A = np.vstack([np.hstack([P, -np.ones((n, 1))]), np.hstack([np.ones((1, m)), [[0.0]]]),
                   np.hstack([-np.ones((1, m)), [[0.0]]]), np.hstack([-np.eye(m), np.zeros((m, 1))])])
    b = np.concatenate([np.zeros(n), [1.0, -1.0], np.zeros(m)])
    M = np.hstack([b[:, None], -A])
    try:
        mat = cdd.Matrix(M, number_type="float")
        mat.rep_type = cdd.RepType.INEQUALITY
        gens = cdd.Polyhedron(mat).get_generators()
    except RuntimeError:                       # float cdd: 'numerical inconsistency' on noisy, near-degenerate points
        PROF.n["exact cdd fallbacks"] += 1
        from fractions import Fraction
        mat = cdd.Matrix([[Fraction(float(x)).limit_denominator(10 ** 9) for x in row] for row in M],
                         number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        gens = cdd.Polyhedron(mat).get_generators()
    corners = []
    for i, g in enumerate(np.array(gens, dtype=float)):
        if g[0] != 1 or i in gens.lin_set:
            continue
        w = np.clip(g[1:m + 1], 0.0, None)
        corners.append(w / w.sum())
    return corners


def estimate_improvement(w, psi_vecs, w_exp):
    res = linprog(c=-w, A_ub=np.array(w_exp), b_ub=[v_smp(psi_vecs, wp) for wp in w_exp],
                  bounds=[(None, None)] * len(w), method="highs")
    return (-res.fun if res.success else np.inf) - v_smp(psi_vecs, w)


def _key(w):
    return tuple(np.round(w, 8))


def ols(query, tol, max_iterations=300):
    ccs, W_exp, visited, rejected = [], [], set(), set()
    counter = itertools.count()
    heap = [(-np.inf, next(counter), tuple(np.eye(D)[i])) for i in range(D)]
    n_queries = 0
    while heap and n_queries < max_iterations:
        neg_p, _, wt = heapq.heappop(heap)
        w = np.array(wt)
        if np.isfinite(neg_p):
            delta = estimate_improvement(w, [p.psi0 for p in ccs], W_exp)
            if delta < tol:
                rejected.add(_key(w))
                continue
            if heap and delta < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-delta, next(counter), wt))
                continue
        visited.add(_key(w))
        n_queries += 1
        PROF.query = n_queries
        policy = query(w)
        if PROF.logfile:
            with open(PROF.logfile, "a") as fh:
                fh.write(f"query {n_queries} |CCS| {len(ccs)} steps {PROF.n['env steps']:,} to-converge "
                         f"{PROF.n['env steps to converge']:,} {dict((k, v) for k, v in PROF.n.items() if 'steps' not in k)}\n")
        W_exp.append(w)
        psis = [p.psi0 for p in ccs]
        if psis and policy.psi0 @ w <= v_smp(psis, w) + 1e-9:
            continue
        ccs.append(policy)
        psis.append(policy.psi0)
        P_old = np.array(psis[:-1]) if len(psis) > 1 else np.zeros((0, D))

        def dominated(h):
            wq = np.array(h[2])
            return len(P_old) and policy.psi0 @ wq > (P_old @ wq).max() + 1e-9
        heap = [h for h in heap if h[0] == -np.inf or not dominated(h)]
        known = visited | rejected | {_key(np.array(h[2])) for h in heap}
        for wc in corner_weights(psis):
            key = _key(wc)
            if key in known:
                continue
            known.add(key)
            delta = estimate_improvement(wc, psis, W_exp)
            if delta > tol:
                heapq.heappush(heap, (-delta, next(counter), tuple(wc)))
            else:
                rejected.add(key)
        heapq.heapify(heap)
    return ccs, n_queries


# ---------------------------------------------------------------- stochastic SF Q-learning (no model)
_NEXT_L, _DONE_L = NEXT.tolist(), DONE.tolist()
_ALL = np.flatnonzero(NONTERM).tolist()
_PHI_L = [[PHI[s, a] for a in range(A_N)] for s in range(S)]


def env_step(s, a, rng):
    """Returns (next state, done, feature vector)."""
    if DOMAIN != "office":
        cum, outs = WOUT[s][a]
        u = rng.random() * cum[-1]
        return outs[next(i for i, c in enumerate(cum) if u < c)]
    if rng.random() < SLIP:
        a = rng.choice([b for b in range(A_N) if b != a])
    return _NEXT_L[s][a], _DONE_L[s][a], _PHI_L[s][a]


def sfql(w, psi_init, allowed, rng, oracle, n0=0, visits0=None):
    """oracle = (v_opt at INIT, model): measurement only. visits0: real visits inherited from an earlier learn of the
    same w (they set the step sizes and count towards NMIN). Returns (Psi, steps, steps_to_converge, visits)."""
    Psi = psi_init.copy()
    Qw = Psi @ w
    masked = np.where(allowed, 0.0, -np.inf)
    allowed_l = [np.flatnonzero(allowed[s]).tolist() for s in range(S)]
    visits = np.full((S, A_N), n0, dtype=int)                  # n0 pseudo-visits for a warm-started table
    if visits0 is not None:
        visits += visits0
    v_opt, M = oracle

    def greedy(s):
        return int((Qw[s] + masked[s]).argmax())

    def greedy_all():
        return (Qw + masked).argmax(1)

    steps, t_ep, s, last_bad, stable = 0, Q_HORIZON, None, 0, 0
    prev = greedy_all()
    flips = []
    while steps < MAX_STEPS:
        if s is None or t_ep >= Q_HORIZON:
            s, t_ep = rng.choice(_ALL), 0
            a = rng.choice(allowed_l[s])
        else:
            a = rng.choice(allowed_l[s]) if rng.random() < Q_EPS else greedy(s)
        ns, done, phi = env_step(s, a, rng)
        target = phi if done else phi + GAMMA * Psi[ns, greedy(ns)]
        visits[s, a] += 1
        Psi[s, a] += (target - Psi[s, a]) * (ALPHA if ALPHA else 1.0 / (visits[s, a] + 1) ** KAPPA)
        Qw[s, a] = Psi[s, a] @ w
        steps += 1
        t_ep += 1
        s = None if done else ns
        if steps % CHECK == 0:
            g = greedy_all()
            if (sf_state(g, M) @ w)[INIT].mean() < v_opt - DELTA:          # oracle (measurement only)
                last_bad = steps
            nflip = int((g != prev)[NONTERM].sum())
            flips.append(nflip)
            stable = stable + 1 if nflip <= STAB_K else 0
            prev = g
            if stable >= STABLE and visits[allowed & NONTERM[:, None]].min() >= NMIN + n0:
                break
    if steps >= MAX_STEPS:
        PROF.n["learns that hit the step cap"] += 1
    PROF.flips.append(flips)
    PROF.n["env steps"] += steps
    PROF.n["env steps to converge"] += last_bad
    PROF.log.append((PROF.query, last_bad))
    PROF.log_total.append((PROF.query, steps))
    return Psi, steps, last_bad, visits


def learned_policy(w, Psi, allowed, source, learned=None):
    actions = np.where(allowed, Psi @ w, -np.inf).argmax(1)
    psi_s = Psi[ROWS, actions]
    learned = allowed if learned is None else learned
    return Policy(w, actions, psi_s, np.where(learned[..., None], Psi, np.nan), source)


def gpi_init(w, tables):
    T = np.stack(tables)
    j = np.nan_to_num(np.einsum("nsad,d->nsa", T, w), nan=-np.inf).argmax(0)
    return np.take_along_axis(T, j[None, ..., None], 0)[0]


class Learner:
    def __init__(self, rng, M):
        self.rng, self.M = rng, M

    def full_learn(self, w, warm_tables=None, init=None, visits0=None):
        # optimistic init, tight: Q = max(w) everywhere (Psi = 1 in every component gives Q = sum(w), which
        # decaying step sizes with gamma = 0.99 take millions of steps to unlearn)
        if init is None:
            init = np.tile(np.eye(D)[int(np.argmax(w))], (S, A_N, 1)) if not warm_tables else gpi_init(w, warm_tables)
        v_opt = policy_iteration(w, self.M)[1][INIT].mean()
        Psi, steps, conv, _ = sfql(w, init, FULL, self.rng, (v_opt, self.M), n0=N0 if warm_tables else 0,
                                   visits0=visits0)
        PROF.n["full learns"] += 1
        PROF.learns.append(("full", PROF.query, steps, conv))
        return learned_policy(w, Psi, FULL, "q-full")


class StandardQ(Learner):
    def __init__(self, rng, M, warm=False):
        super().__init__(rng, M)
        self.warm, self.learned = warm, []

    def __call__(self, w):
        p = self.full_learn(w, [q.psi_sa for q in self.learned] if self.warm else None)
        self.learned.append(p)
        return p


class CertifyOrGrowQModelFree(Learner):
    """Certify-or-Grow + pool warm start, model-free certificate (notebook 14's 4th arm, stochastic version)."""

    def __init__(self, rng, M):
        super().__init__(rng, M)
        self.pool, self.basis = [], []
        for i in range(D):
            warm = [p.psi_sa for p in self.basis] if BASIS_WARM and self.basis else None
            self.basis.append(self.full_learn(np.eye(D)[i], warm))

    def _allowed(self):
        own = np.stack([p.actions for p in self.basis])
        allowed = np.zeros((S, A_N), dtype=bool)
        allowed[ROWS[None, :].repeat(len(own), 0), own] = True
        PROF.cover.append((PROF.query, len(self.basis), float(allowed[NONTERM].sum(1).mean())))
        return allowed

    def _reuse(self, w):
        """QREUSE=1: OLS's first d queries are the extremes, which the basis already answers (no learn)."""
        for p in (self.basis[:D] if REUSE else []):
            if np.allclose(p.w, w):
                PROF.n["reused"] += 1
                return p
        return None

    def _grow(self, w, Psi, visits, allowed, learned):
        """Full learn of w after a failed certificate; the new policy joins the basis. QFALLBACK = cold: from the
        optimistic init; gpi: GPI warm start over every table learned so far, the failed metapolicy's (its `learned`
        entries) included, as SFOLS + warm start; continue: the metapolicy learn goes on with all actions (its table
        and visit counts on `learned`, GPI over the basis and pool elsewhere)."""
        PROF.n["fallbacks"] += 1
        tables = [p.psi_sa for p in self.basis + self.pool]
        if FALLBACK == "cold":
            p = self.full_learn(w)
        elif FALLBACK == "gpi":
            p = self.full_learn(w, tables + [np.where(learned[..., None], Psi, np.nan)])
        else:
            p = self.full_learn(w, init=np.where(learned[..., None], Psi, gpi_init(w, tables)),
                                visits0=np.where(allowed, visits, 0))
        self.basis.append(p)
        return p

    def __call__(self, w):
        reused = self._reuse(w)
        if reused is not None:
            return reused
        allowed = self._allowed()
        init = np.where(allowed[..., None], gpi_init(w, [p.psi_sa for p in self.basis + self.pool]), 0.0)
        v_opt = policy_iteration(w, self.M, allowed)[1][INIT].mean()              # oracle: restricted optimum
        Psi, steps, conv, visits = sfql(w, init, allowed, self.rng, (v_opt, self.M), n0=N0)
        PROF.learns.append(("meta", PROF.query, steps, conv))
        acts = np.where(allowed, Psi @ w, -np.inf).argmax(1)
        c = self._complete(Psi, allowed, acts)
        PROF.n["env steps"] += c
        PROF.n["env steps to converge"] += c
        PROF.n["completion steps"] += c
        q, x = PROF.log[-1]; PROF.log[-1] = (q, x + c)
        q, x = PROF.log_total[-1]; PROF.log_total[-1] = (q, x + c)
        policy = learned_policy(w, Psi, allowed, "q-meta-mf", learned=FULL)
        Qw = Psi @ w
        gap = (Qw.max(1) - Qw[ROWS, policy.actions]).max()
        # oracle: is the learned metapolicy really optimal for w? (measurement only)
        V = sf_state(policy.actions, self.M) @ w
        true_gap = (q_backup(V, w, self.M).max(1) - V).max()
        PROF.cert.append((PROF.query, float(true_gap), float(policy_iteration(w, self.M)[1][INIT].mean() - V[INIT].mean())))
        PROF.n["true gap <= cert tol"] += true_gap <= CERT_TOL
        if gap > CERT_TOL:
            PROF.n["false fallbacks"] += true_gap <= CERT_TOL
            return self._grow(w, Psi, visits, allowed, FULL)
        PROF.n["certified"] += 1
        PROF.n["false certificates"] += true_gap > CERT_TOL
        if not any(np.allclose(policy.psi0, q.psi0, atol=1e-6) for q in self.pool):
            self.pool.append(policy)
        return policy

    def _complete(self, Psi, allowed, acts):
        steps = 0
        for s in range(S):
            for a in range(A_N):
                if allowed[s, a] or not NONTERM[s]:
                    continue
                acc = np.zeros(D)
                for _ in range(M_COMPLETE):
                    ns, done, phi = env_step(s, a, self.rng)
                    acc += phi if done else phi + GAMMA * Psi[ns, acts[ns]]
                Psi[s, a] = acc / M_COMPLETE
                steps += M_COMPLETE
        return steps


class CertifyOrGrowQ(CertifyOrGrowQModelFree):
    """Certify-or-Grow + pool warm start, certificate with the model (notebook 14's 3rd arm): the exact Bellman gap of
    the learned metapolicy, no completion samples."""

    def __call__(self, w):
        reused = self._reuse(w)
        if reused is not None:
            return reused
        allowed = self._allowed()
        init = np.where(allowed[..., None], gpi_init(w, [p.psi_sa for p in self.basis + self.pool]), 0.0)
        v_opt = policy_iteration(w, self.M, allowed)[1][INIT].mean()
        Psi, steps, conv, visits = sfql(w, init, allowed, self.rng, (v_opt, self.M), n0=N0)
        PROF.learns.append(("meta", PROF.query, steps, conv))
        policy = learned_policy(w, Psi, allowed, "q-meta")
        V = sf_state(policy.actions, self.M) @ w
        gap = (q_backup(V, w, self.M).max(1) - V)[NONTERM].max()
        PROF.cert.append((PROF.query, float(gap), float(policy_iteration(w, self.M)[1][INIT].mean() - V[INIT].mean())))
        if gap > CERT_TOL:
            return self._grow(w, Psi, visits, allowed, allowed)
        PROF.n["certified"] += 1
        if not any(np.allclose(policy.psi0, q.psi0, atol=1e-6) for q in self.pool):
            self.pool.append(policy)
        return policy


ARMS = {"standard": "standard SFOLS", "standard-warm": "standard + GPI warm start",
        "certify": "CoG + warm start", "certify-mf": "CoG + warm start, model-free"}


def _tag(slip):
    cog = f"{'_bw' if BASIS_WARM else ''}{'_reuse' if REUSE else ''}{'' if FALLBACK == 'cold' else '_fb' + FALLBACK}"
    if DOMAIN == "office":
        return f"slip{slip}{cog}"
    return f"wind{WIND}_g{GAMMA}_eps{OLS_EPS}_cert{CERT_TOL}_stab{STAB_K}{cog}"


def run_q(arm, seed, slip, n0=0):
    global PROF, SLIP, N0
    SLIP, N0 = slip, n0
    PROF = Profile()
    PROF.logfile = os.path.join(OUT_DIR, "logs", f"q_ccs_{'' if DOMAIN == 'office' else 'ds_'}{_tag(slip)}{'_n0%d' % N0 if N0 else ''}_{arm}_{seed}.log")
    os.makedirs(os.path.dirname(PROF.logfile), exist_ok=True)
    open(PROF.logfile, "w").close()
    rng = random.Random(seed)
    M = _model()
    t0 = time.perf_counter()
    query = {"standard": lambda: StandardQ(rng, M), "standard-warm": lambda: StandardQ(rng, M, warm=True),
             "certify": lambda: CertifyOrGrowQ(rng, M), "certify-mf": lambda: CertifyOrGrowQModelFree(rng, M)}[arm]()
    ccs, n_queries = ols(query, tol=OLS_EPS)
    true_psi0 = [sf_state(p.actions, M)[INIT].mean(0) for p in ccs]
    return dict(arm=arm, seed=seed, slip=slip, n_queries=n_queries, total=time.perf_counter() - t0,
                n=dict(PROF.n), log=PROF.log, log_total=PROF.log_total, learns=PROF.learns,
                ccs_actions=[p.actions for p in ccs], ccs_psi0=[p.psi0 for p in ccs], true_psi0=true_psi0,
                n_basis=len(getattr(query, "basis", [])), reuse=REUSE, fallback=FALLBACK, basis_cover=PROF.cover,
                cert=PROF.cert)


def _job(args):
    try:
        return run_q(*args)
    except Exception as e:                     # one failed job must not kill the pool
        import traceback
        return dict(arm=args[0], seed=args[1], slip=args[2], error=traceback.format_exc())


def main_run(slip, seeds, n0=0, workers=None):
    global SLIP, N0
    SLIP, N0 = slip, n0
    arms = [a for a in ARMS if a in os.environ.get("ARMS", ",".join(ARMS)).split(",")]
    jobs = [(arm, seed, slip, n0) for seed in range(SEED0, SEED0 + seeds) for arm in arms]
    out = os.path.join(OUT_DIR, f"{'office' if DOMAIN == 'office' else 'ds'}_q_runs_{_tag(slip)}"
                                         f"{'_n0%d' % n0 if n0 else ''}.pkl")
    t0 = time.perf_counter()
    runs = []
    with mp.get_context("fork").Pool(workers or min(len(jobs), max(1, len(os.sched_getaffinity(0)) - 1))) as pool:
        for r in pool.imap_unordered(_job, jobs):
            runs.append(r)
            if "error" in r:
                print(f"  {ARMS[r['arm']]:30s} seed {r['seed']}: FAILED\n{r['error']}", flush=True)
                continue
            print(f"  {ARMS[r['arm']]:30s} seed {r['seed']}: queries {r['n_queries']}, CCS {len(r['ccs_psi0'])}, "
                  f"steps to converge {r['n']['env steps to converge']:,}, all steps {r['n']['env steps']:,}, "
                  f"{r['total']:.0f}s, {dict((k, v) for k, v in r['n'].items() if 'steps' not in k)}", flush=True)
            with open(out, "wb") as fh:
                pickle.dump(runs, fh)
    print(f"{len(jobs)} runs in {time.perf_counter() - t0:.0f}s -> {out}")
    # reference: exact eps-CCS and quality of each learned CCS (true SFs of the learned policies)
    M = _model()
    rng = np.random.default_rng(0)
    Wt = [rng.dirichlet(np.ones(D)) for _ in range(300)]
    vstar = np.array([policy_iteration(w, M)[1][INIT].mean() for w in Wt])
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm and "error" not in r]
        if not rs:
            continue
        gaps = [max(vstar[i] - max(p @ Wt[i] for p in r["true_psi0"]) for i in range(len(Wt))) for r in rs]
        conv = np.median([r["n"]["env steps to converge"] for r in rs])
        tot = np.median([r["n"]["env steps"] for r in rs])
        print(f"{ARMS[arm]:30s} median steps to converge {conv:,.0f}, all steps {tot:,.0f}, CCS size "
              f"{np.median([len(r['ccs_psi0']) for r in rs]):.0f}, worst eps-gap of the learned CCS {max(gaps):.4f}")
    if any("error" in r for r in runs):
        sys.exit(1)                                # failed runs must show up as failed (cluster/sweep.py status)


def main_exact(slip):
    """Exact OLS with the model (no learning): eps-CCS size, and how often Certify-or-Grow's exact metapolicy would be
    certified (gap <= CERT_TOL) with the basis grown as in the learned arm."""
    global SLIP, PROF
    SLIP = slip
    PROF = Profile()
    M = _model()

    def exact(w):
        pi, _ = policy_iteration(w, M)
        Ps = sf_state(pi, M)
        return Policy(w, pi, Ps, None, "exact")
    W = []

    def query(w):
        W.append(w)
        return exact(w)
    t0 = time.perf_counter()
    ccs, nq = ols(query, tol=OLS_EPS)
    basis, fb = [exact(np.eye(D)[i]) for i in range(D)], 0
    for w in W:
        allowed = np.zeros((S, A_N), bool)
        for p in basis:
            allowed[ROWS, p.actions] = True
        pi, V = policy_iteration(w, M, allowed)
        if (q_backup(V, w, M).max(1) - V)[NONTERM].max() > CERT_TOL:
            fb += 1
            basis.append(exact(w))
    print(f"{DOMAIN} {_tag(slip)}: S={S} ({NONTERM.sum()} non-terminal), A={A_N}, d={D}; eps-CCS {len(ccs)} policies "
          f"from {nq} queries ({time.perf_counter() - t0:.0f}s); exact CoG: {fb} fallbacks, basis {len(basis)}")


def main_pilot(slip):
    global SLIP, PROF
    SLIP = slip
    PROF = Profile()
    M = _model()
    rng = random.Random(0)
    for w in (np.eye(D)[0], np.full(D, 1 / D), np.array([0.4, 0.1, 0.3, 0.05, 0.1, 0.05]) if D == 6 else np.array([0.3, 0.7])):
        t0 = time.perf_counter()
        v_opt = policy_iteration(w, M)[1][INIT].mean()
        init = np.tile(np.eye(D)[int(np.argmax(w))], (S, A_N, 1))
        Psi, steps, conv, _ = sfql(w, init, FULL, rng, (v_opt, M))
        p = learned_policy(w, Psi, FULL, "pilot")
        tv = (sf_state(p.actions, M) @ w)[INIT].mean()
        err = np.abs(p.psi0 - sf_state(p.actions, M)[INIT].mean(0)).max()
        fl = np.array(PROF.flips[-1])
        tail = fl[int(len(fl) * 0.8):]
        print(f"  greedy changes per check over the last 20% of the learn: median {np.median(tail):.0f}, 90th pct "
              f"{np.percentile(tail, 90):.0f}, max {tail.max()}, checks with 0 changes {100 * (tail == 0).mean():.0f}%")
        print(f"slip {slip} alpha {'1/(n+1)^%g' % KAPPA if not ALPHA else ALPHA} w={np.round(w, 2)}: steps to "
              f"converge (oracle) {conv:,}, stopped at {steps:,}, final greedy value {tv:.4f} vs opt {v_opt:.4f}, "
              f"psi0 error {err:.1e}, {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "pilot":
        ALPHA = float(sys.argv[3]) if sys.argv[3] != "decay" else None
    if sys.argv[1] == "pilot":
        main_pilot(float(sys.argv[2]))
    elif sys.argv[1] == "exact":
        main_exact(float(sys.argv[2]))
    else:
        main_run(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0,
                 int(sys.argv[5]) if len(sys.argv) > 5 else None)
