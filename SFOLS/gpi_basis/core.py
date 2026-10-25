"""Exact tabular machinery for building GPI bases on multi-objective MDPs with successor features.

Conventions
-----------
P[s, a, s']   sub-stochastic transitions: the probability mass of transitions that END the episode
              is removed, so terminal states have all-zero rows and V = (I - gamma P_pi)^{-1} r_pi.
PHI[s, a, :]  expected immediate feature vector of (s, a), including features of terminal transitions.
w             a weight vector (normally in the simplex); scalar reward = PHI . w.
relevant      the states at which GPI has to be (near-)optimal; for SF-FSA-VI these are the start
              state and the exit states (where the high-level planner reads values).
"""
import numpy as np


class TabularMOMDP:
    def __init__(self, P, PHI, gamma, relevant, name=""):
        self.P = np.ascontiguousarray(P, dtype=float)
        self.PHI = np.ascontiguousarray(PHI, dtype=float)
        self.gamma = float(gamma)
        self.S, self.A, _ = self.P.shape
        self.D = self.PHI.shape[2]
        self.relevant = np.asarray(relevant, dtype=int)
        self.start = int(self.relevant[0])
        self.name = name
        self.R = np.arange(self.S)
        self._P2 = self.P.reshape(self.S * self.A, self.S)
        self._I = np.eye(self.S)
        self._layer_masks = None
        self.stats = dict(lin_solves=0)

    # ------------------------------------------------------------------ evaluation
    def _solve(self, act, rhs):
        self.stats["lin_solves"] += 1
        return np.linalg.solve(self._I - self.gamma * self.P[self.R, act], rhs)

    def sf(self, act):
        """State successor features Psi^pi(s) := Psi^pi(s, pi(s)), shape (S, D)."""
        return self._solve(act, self.PHI[self.R, act])

    def value(self, act, w):
        return self._solve(act, self.PHI[self.R, act] @ w)

    def sf_q(self, psi_s):
        """Action successor features Psi^pi(s, a) = PHI(s, a) + gamma E[Psi^pi(s')], shape (S, A, D)."""
        return self.PHI + self.gamma * (self._P2 @ psi_s).reshape(self.S, self.A, self.D)

    def q(self, V, w):
        """Full-action one-step backup of a scalar value V, shape (S, A)."""
        return self.PHI @ w + self.gamma * (self._P2 @ V).reshape(self.S, self.A)

    # ------------------------------------------------------------------ control
    def optimal(self, w, init=None, max_sweeps=10_000):
        """Exact policy iteration over the full action set. Returns (actions, V, sweeps)."""
        pi = np.zeros(self.S, dtype=int) if init is None else np.array(init, dtype=int)
        for n in range(1, max_sweeps + 1):
            V = self.value(pi, w)
            Q = self.q(V, w)
            best = Q.argmax(1)
            new = np.where(Q[self.R, pi] >= Q[self.R, best] - 1e-12, pi, best)
            if np.array_equal(new, pi):
                return pi, V, n
            pi = new
        raise RuntimeError("policy iteration did not converge")

    def meta(self, w, own, init_choice=None, max_sweeps=10_000):
        """Metapolicy: policy iteration restricted to B(s) = {own[j, s]} (own: (k, S) base actions).

        Returns (actions, V, Q_full, sweeps); Q_full is the FULL-action backup of V, so the Bellman
        certificate of the metapolicy is max_s [Q_full(s).max() - V(s)]."""
        own = np.asarray(own)
        j = np.zeros(self.S, dtype=int) if init_choice is None else np.array(init_choice, dtype=int)
        cols = self.R
        for n in range(1, max_sweeps + 1):
            act = own[j, cols]
            V = self.value(act, w)
            Q = self.q(V, w)
            Qj = Q[cols[None, :], own]                        # (k, S): value of each base action
            best = Qj.argmax(0)
            new = np.where(Qj[j, cols] >= Qj[best, cols] - 1e-12, j, best)
            if np.array_equal(new, j):
                return act, V, Q, n
            j = new
        raise RuntimeError("restricted policy iteration did not converge")

    # ------------------------------------------------------------------ certificates
    def layer_masks(self, tail_tol=1e-12):
        """For each relevant state r: the layers L_t = 'states reachable from r in exactly t steps under SOME
        policy' (t < t0), stored as one flat index array with segment offsets, and the set of all states
        reachable from r (used for the tail t >= t0)."""
        if self._layer_masks is None:
            adj = self.P.sum(1) > 0                              # (S, S): s -> s' possible
            T = int(np.ceil(np.log(tail_tol) / np.log(self.gamma)))
            masks = []
            for r in self.relevant:
                M = np.zeros((T, self.S), dtype=bool)
                cur = np.zeros(self.S, dtype=bool); cur[r] = True
                reach = cur.copy()
                for t in range(T):
                    M[t] = cur
                    cur = adj[cur].any(0)
                    reach |= cur
                # the layer map L_t -> L_{t+1} is deterministic, so once two consecutive layers are equal all
                # later ones are too: keep layers up to that point and let the tail term
                # gamma^t0 / (1 - gamma) * max_{L_t0}(gap) cover the rest exactly. Without stationarity, the
                # tail beyond T is bounded with the full reachable set.
                stat = np.flatnonzero((M[:-1] == M[1:]).all(1))
                if len(stat):
                    t0 = int(stat[0])
                    tail_set = M[t0].copy()
                else:
                    t0, tail_set = T, reach
                nonempty = [t for t in range(t0) if M[t].any()]
                idx = [np.flatnonzero(M[t]) for t in nonempty]
                flat = np.concatenate(idx) if idx else np.zeros(0, dtype=int)
                offsets = np.cumsum([0] + [len(i) for i in idx[:-1]]) if idx else np.zeros(0, dtype=int)
                disc = self.gamma ** np.array(nonempty, dtype=float)
                masks.append((flat, offsets, disc, np.flatnonzero(tail_set), self.gamma ** t0 / (1 - self.gamma)))
            self._layer_masks = masks
        return self._layer_masks

    def reach_bound(self, gap):
        """Upper bound on V*(r) - V^pi(r) for every relevant state r, from the Bellman gap vector of pi:
            V* - V^pi = sum_t gamma^t P_{pi*}^t A,  A(s) = Q^pi(s, pi*(s)) - V^pi(s) <= gap(s),
        and from r the optimal policy is at time t somewhere in the layer reachable in exactly t steps.
        gap: (S,) or (n, S). Returns (n_relevant,) or (n, n_relevant)."""
        masks = self.layer_masks()
        g = np.maximum(np.atleast_2d(gap), 0.0).T                # (S, n)
        out = np.empty((g.shape[1], len(masks)))
        for i, (flat, offsets, disc, reach, tail) in enumerate(masks):
            body = disc @ np.maximum.reduceat(g[flat], offsets, axis=0) if len(flat) else 0.0
            out[:, i] = body + tail * g[reach].max(0)
        return out[0] if np.ndim(gap) == 1 else out


# ---------------------------------------------------------------------- policy records
class PolicyRecord:
    """A policy with what GPI and the certificates need: its actions, state SFs and (lazily) action SFs."""
    __slots__ = ("mdp", "actions", "psi_s", "_psi_sa", "key")

    def __init__(self, mdp, actions, psi_s=None):
        self.mdp = mdp
        self.actions = np.asarray(actions, dtype=int)
        self.psi_s = mdp.sf(self.actions) if psi_s is None else psi_s
        self._psi_sa = None
        self.key = self.actions.tobytes()

    @property
    def psi_sa(self):
        if self._psi_sa is None:
            self._psi_sa = self.mdp.sf_q(self.psi_s)
        return self._psi_sa

    def gap(self, w):
        """Bellman gap of this fixed policy at weight w (vector over states, >= 0)."""
        return np.maximum((self.psi_sa @ w).max(1) - self.psi_s @ w, 0.0)


class PolicyCache:
    """Evaluations of policies keyed by their action table (GPI cell policies repeat a lot).
    Records that are in a pool are kept by the pool; the cache itself is cleared when it gets big."""
    def __init__(self, mdp, maxsize=4000):
        self.mdp, self.d, self.misses, self.maxsize = mdp, {}, 0, maxsize

    def get(self, actions):
        k = np.asarray(actions, dtype=int).tobytes()
        rec = self.d.get(k)
        if rec is None:
            if len(self.d) >= self.maxsize:
                self.d.clear()
            rec = PolicyRecord(self.mdp, actions)
            self.d[k] = rec
            self.misses += 1
        return rec


# ---------------------------------------------------------------------- GPI
def tol_argmax(Q, tol=1e-12):
    """argmax over the last axis that breaks numerical ties (within tol) by the lowest index, so that
    floating-point noise between tied actions never changes the selected action."""
    return (Q >= Q.max(-1, keepdims=True) - tol).argmax(-1)


class Pool:
    """The GPI pool: action-SF tables of all stored policies, kept stacked for fast GPI."""
    def __init__(self, mdp):
        self.mdp = mdp
        self.records, self.kinds, self.weights = [], [], []
        self.stack = np.zeros((0, mdp.S, mdp.A, mdp.D))

    def add(self, rec, kind, w):
        self.records.append(rec); self.kinds.append(kind); self.weights.append(np.asarray(w, float))
        self.stack = np.concatenate([self.stack, rec.psi_sa[None]], 0)

    def __len__(self):
        return len(self.records)

    def qhat(self, w, upto=None):
        """GPI estimate max_j Psi_j(s, a).w, shape (S, A)."""
        st = self.stack if upto is None else self.stack[:upto]
        return np.tensordot(st, w, axes=([3], [0])).max(0)

    def actions(self, w, upto=None):
        return tol_argmax(self.qhat(w, upto))
