"""Model-free tabular learners compiled with numba: the sample-by-sample updates of q_fsa_stochastic.py and q_lazy_fsa.py
(step 1/(n+1)^KAPPA, eps-greedy, exploring starts), about 50x faster than the Python loops.

    sfql    vector SF Q-learning at a weight w, acting and bootstrapping on an `allowed` action mask (B(s) for the
            metapolicy); visit counts are kept so that warm starts can inherit them
    flatq   scalar Q-learning on the product MDP U x S of a task automaton (Kuric semantics), the baseline without SFs

Sampling tables come from an exact MOMDP of certlib.domains: the outcomes of (s, a) are the next non-terminal state, or
entering exit o (feature e_o, the episode ends). The learners run CHECK steps at a time and an oracle, which only
measures, is evaluated in Python in between (policy values by Gauss-Seidel sweeps, warm-started). The model is used for
nothing else.
"""
import numpy as np
from numba import njit

KAPPA, EPS, HORIZON = 0.7, 0.1, 100
EVAL_TOL = 1e-10                       # error of the oracle's policy values


class Sampler:
    """Outcome tables of an MOMDP, (S, A, K) each: cum / prob (cumulative) probabilities, nxt the next state or -1,
    obj the exit entered or -1."""

    def __init__(self, mdp):
        S, A = mdp.S, mdp.A
        outs = [[[(float(p), int(n), -1) for n, p in enumerate(mdp.P[s, a]) if p > 0] +
                 [(float(p), -1, int(o)) for o, p in enumerate(mdp.PHI[s, a]) if p > 0] for a in range(A)]
                for s in range(S)]
        K = max(len(o) for row in outs for o in row)
        self.cum = np.full((S, A, K), 2.0)
        self.prob = np.zeros((S, A, K))
        self.nxt = np.full((S, A, K), -1, np.int64)
        self.obj = np.full((S, A, K), -1, np.int64)
        for s in range(S):
            for a in range(A):
                o = outs[s][a] or [(1.0, s, -1)]            # no outcome: a self-loop without feature
                c = np.cumsum([x[0] for x in o])
                c[-1] = 1.0 + 1e-12
                self.cum[s, a, :len(o)] = c
                self.prob[s, a, :len(o)] = [x[0] for x in o]
                self.nxt[s, a, :len(o)] = [x[1] for x in o]
                self.obj[s, a, :len(o)] = [x[2] for x in o]
        self.starts = np.arange(S, dtype=np.int64)          # exploring starts over every state (exits included)


def action_lists(allowed):
    """(S, A) allowed mask -> per-state action lists padded with -1, and their lengths."""
    S, A = allowed.shape
    al = np.full((S, A), -1, np.int64)
    for s in range(S):
        idx = np.flatnonzero(allowed[s])
        al[s, :len(idx)] = idx
    return al, allowed.sum(1).astype(np.int64)


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def _sample(cum, nxt, obj, s, a):
    u = np.random.random()
    k = 0
    while u >= cum[s, a, k]:
        k += 1
    return nxt[s, a, k], obj[s, a, k]


@njit(cache=True)
def _greedy(Q, al, ac, s):
    b, best = al[s, 0], Q[s, al[s, 0]]
    for i in range(1, ac[s]):
        a = al[s, i]
        if Q[s, a] > best:
            b, best = a, Q[s, a]
    return b


@njit(cache=True)
def sfql_chunk(Psi, n, Qw, w, al, ac, starts, cum, nxt, obj, state, nsteps, eps, kappa, gamma, horizon):
    """nsteps of vector SF Q-learning; state = [s, t] carries the episode over chunks (s = -1: start a new one)."""
    D = Psi.shape[2]
    s, t = state[0], state[1]
    for _ in range(nsteps):
        if s < 0 or t >= horizon:
            s, t = starts[np.random.randint(starts.shape[0])], 0
            a = al[s, np.random.randint(ac[s])]
        elif np.random.random() < eps:
            a = al[s, np.random.randint(ac[s])]
        else:
            a = _greedy(Qw, al, ac, s)
        ns, o = _sample(cum, nxt, obj, s, a)
        n[s, a] += 1.0
        lr = 1.0 / (n[s, a] + 1.0) ** kappa
        q = 0.0
        if o >= 0:
            for d in range(D):
                Psi[s, a, d] += ((1.0 if d == o else 0.0) - Psi[s, a, d]) * lr
                q += Psi[s, a, d] * w[d]
        else:
            b = _greedy(Qw, al, ac, ns)
            for d in range(D):
                Psi[s, a, d] += (gamma * Psi[ns, b, d] - Psi[s, a, d]) * lr
                q += Psi[s, a, d] * w[d]
        Qw[s, a] = q
        t += 1
        s = -1 if o >= 0 else ns
    state[0], state[1] = s, t


@njit(cache=True)
def flatq_chunk(Q, n, tau, exits, nU, S, starts, cum, nxt, obj, state, nsteps, eps, kappa, gamma, horizon):
    """nsteps of Q-learning on the product MDP. Q, n: (nU * S, A); tau[u, o] = next automaton state on entering exit o
    in u (-1: accepting, reward 1); exits[o] = low-level state where the next stage starts (the jump is discounted by
    gamma, as in certlib.lazyfsa). state = [u, s, t]."""
    A = Q.shape[1]
    u, s, t = state[0], state[1], state[2]
    for _ in range(nsteps):
        if s < 0 or t >= horizon:
            u, s, t = np.random.randint(nU), starts[np.random.randint(starts.shape[0])], 0
            a = np.random.randint(A)
        else:
            x = u * S + s
            if np.random.random() < eps:
                a = np.random.randint(A)
            else:
                a, best = 0, Q[x, 0]
                for b in range(1, A):
                    if Q[x, b] > best:
                        a, best = b, Q[x, b]
        ns, o = _sample(cum, nxt, obj, s, a)
        x = u * S + s
        done = False
        if o >= 0:
            v = tau[u, o]
            if v < 0:
                target, done = 1.0, True
            else:
                y = v * S + exits[o]
                m = Q[y, 0]
                for b in range(1, A):
                    m = max(m, Q[y, b])
                target, u, ns = gamma * m, v, exits[o]
        else:
            y = u * S + ns
            m = Q[y, 0]
            for b in range(1, A):
                m = max(m, Q[y, b])
            target = gamma * m
        n[x, a] += 1.0
        Q[x, a] += (target - Q[x, a]) / (n[x, a] + 1.0) ** kappa
        t += 1
        s = -1 if done else ns
    state[0], state[1], state[2] = u, s, t


@njit(cache=True)
def eval_policy(act, w, prob, nxt, obj, gamma, V, tol):
    """Value of the deterministic policy act at weight w by in-place Gauss-Seidel sweeps from V (modified)."""
    S, K = act.shape[0], prob.shape[2]
    r = np.zeros(S)
    for s in range(S):
        for k in range(K):
            if obj[s, act[s], k] >= 0:
                r[s] += prob[s, act[s], k] * w[obj[s, act[s], k]]
    thr = tol * (1.0 - gamma) / gamma
    for _ in range(1_000_000):
        delta = 0.0
        for s in range(S):
            v = r[s]
            for k in range(K):
                j = nxt[s, act[s], k]
                if j >= 0:
                    v += gamma * prob[s, act[s], k] * V[j]
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta <= thr:
            break
    return V


@njit(cache=True)
def eval_product(act, tau, exits, prob, nxt, obj, gamma, V, tol):
    """Value of a product policy act (nU, S) by Gauss-Seidel from V (nU, S), modified in place (Kuric semantics:
    reward 1 on entering an accepting exit; entering exit o otherwise jumps to (tau[u, o], exits[o]), discounted)."""
    nU, S = act.shape
    K = prob.shape[2]
    thr = tol * (1.0 - gamma) / gamma
    for _ in range(1_000_000):
        delta = 0.0
        for u in range(nU):
            for s in range(S):
                a, v = act[u, s], 0.0
                for k in range(K):
                    p = prob[s, a, k]
                    if p == 0.0:
                        continue
                    o, j = obj[s, a, k], nxt[s, a, k]
                    if o >= 0:
                        if tau[u, o] < 0:
                            v += p
                        else:
                            v += gamma * p * V[tau[u, o], exits[o]]
                    else:
                        v += gamma * p * V[u, j]
                delta = max(delta, abs(v - V[u, s]))
                V[u, s] = v
        if delta <= thr:
            break
    return V


# ---------------------------------------------------------------- oracle (measurement only)
class Oracle:
    """Exact values of deterministic policies (Gauss-Seidel, warm-started from the last call) and V*_w by policy
    iteration, cached per weight (and action mask)."""

    def __init__(self, mdp, smp):
        self.mdp, self.smp = mdp, smp
        self.R = np.arange(mdp.S)
        self._V = np.zeros(mdp.S)
        self._vstar = {}

    def value(self, act, w, V0=None):
        V = (self._V if V0 is None else V0).copy()
        V = eval_policy(np.asarray(act, np.int64), np.asarray(w, float), self.smp.prob, self.smp.nxt, self.smp.obj,
                        self.mdp.gamma, V, EVAL_TOL)
        if V0 is None:
            self._V = V
        return V

    def vstar(self, w, allowed=None):
        key = (tuple(np.round(w, 12)), None if allowed is None else allowed.tobytes())
        if key not in self._vstar:
            m = self.mdp
            act = np.zeros(m.S, np.int64) if allowed is None else np.argmax(allowed, 1).astype(np.int64)
            V = np.zeros(m.S)
            for _ in range(1000):
                V = self.value(act, w, V)
                Q = m.PHI @ w + m.gamma * (m._P2 @ V).reshape(m.S, m.A)
                if allowed is not None:
                    Q = np.where(allowed, Q, -np.inf)
                best = Q.argmax(1)
                new = np.where(Q[self.R, act] >= Q[self.R, best] - 1e-12, act, best)
                if np.array_equal(new, act):
                    break
                act = new
            else:
                raise RuntimeError("PI did not converge")
            self._vstar[key] = (act, V)
        return self._vstar[key]


def warmup(smp, gamma):
    """Compile every kernel (call before forking workers, so that they share the compiled code)."""
    S, A, K = smp.prob.shape
    D = int(smp.obj.max()) + 1
    Q, n = np.zeros((S, A)), np.zeros((S, A))
    al, ac = action_lists(np.ones((S, A), bool))
    sfql_chunk(np.zeros((S, A, D)), n.copy(), Q.copy(), np.ones(D), al, ac, smp.starts, smp.cum, smp.nxt, smp.obj,
               np.array([-1, 0], np.int64), 10, EPS, KAPPA, gamma, HORIZON)
    flatq_chunk(Q.copy(), n.copy(), np.full((1, D), -1, np.int64), np.zeros(D, np.int64), 1, S, smp.starts, smp.cum,
                smp.nxt, smp.obj, np.array([0, -1, 0], np.int64), 10, EPS, KAPPA, gamma, HORIZON)
    eval_policy(np.zeros(S, np.int64), np.ones(D), smp.prob, smp.nxt, smp.obj, gamma, np.zeros(S), 1e-3)
    eval_product(np.zeros((1, S), np.int64), np.full((1, D), -1, np.int64), np.zeros(D, np.int64), smp.prob, smp.nxt,
                 smp.obj, gamma, np.zeros((1, S)), 1e-3)


def sfql(smp, w, Psi0, n0, allowed, ok, budget, stable, check, rng_seed, gamma):
    """Vector SFQL from (Psi0, n0) until ok(greedy actions, estimated values) held for `stable` checks in a row
    (oracle), or `budget` steps. Returns Psi, n, greedy actions, steps used, steps to converge (the last failed check)."""
    Psi, n = Psi0.copy(), n0.astype(float).copy()
    Qw = Psi @ w
    al, ac = action_lists(allowed)
    masked = np.where(allowed, 0.0, -np.inf)
    seed(rng_seed)
    state = np.array([-1, 0], np.int64)

    def check_ok():
        Q = Qw + masked
        return ok(Q.argmax(1), Q.max(1))
    good, steps, last_bad = 0, 0, (0 if not check_ok() else -1)
    while steps < budget:
        sfql_chunk(Psi, n, Qw, w, al, ac, smp.starts, smp.cum, smp.nxt, smp.obj, state, check, EPS, KAPPA, gamma,
                   HORIZON)
        steps += check
        if check_ok():
            good += 1
            if good >= stable:
                break
        else:
            good, last_bad = 0, steps
    return Psi, n, (Qw + masked).argmax(1), steps, max(last_bad, 0)
