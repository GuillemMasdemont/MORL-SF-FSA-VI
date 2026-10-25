"""Tabular multi-objective MDP with solve-free operations.

Everything here is built from one primitive, the expected SF backup

    psi(s, a) <- phi(s, a) + gamma * sum_s' P(s' | s, a) psi(s', pi(s')),

which has a sample-based counterpart (a TD update on a transition (s, a, phi, s')). No linear system is
solved anywhere: policies are evaluated by iterating the backup, improved by greedy steps, and certified
by Bellman gaps read off the same action-SF table. The number of backups ("sweeps", one pass over all
state-action pairs) is counted as the cost of every operation.

Conventions: P[s, a, s'] is sub-stochastic (mass of transitions that end the episode is removed),
PHI[s, a] is the expected immediate feature vector, `relevant` are the states where values are read
(the start state and the exits, for SF-FSA-VI).
"""
import numpy as np

NUM_ZERO = 1e-10        # numerical zero for gaps / bounds (values are O(1))


def tol_argmax(Q, tol=1e-12):
    """argmax over the last axis; ties within tol go to the lowest index."""
    return (Q >= Q.max(-1, keepdims=True) - tol).argmax(-1)


class MOMDP:
    def __init__(self, P, PHI, gamma, relevant, name=""):
        self.P = np.ascontiguousarray(P, dtype=float)
        self.PHI = np.ascontiguousarray(PHI, dtype=float)
        self.gamma = float(gamma)
        self.S, self.A, _ = self.P.shape
        self.D = self.PHI.shape[2]
        self.relevant = np.asarray(relevant, dtype=int)
        self.name = name
        self.R = np.arange(self.S)
        self._P2 = self.P.reshape(self.S * self.A, self.S)
        self._layers = None
        self.sweeps = 0                                    # cost counter (expected SF backups)

    # ------------------------------------------------------------------ the primitive
    def backup(self, psi_s):
        """Action SFs from state SFs: PHI + gamma * P psi_s, shape (S, A, D). Counts one sweep."""
        self.sweeps += 1
        return self.PHI + self.gamma * (self._P2 @ psi_s).reshape(self.S, self.A, self.D)

    def backup_scalar(self, V, w):
        """One-step lookahead of a scalar value: r_w(s, a) + gamma * P V, shape (S, A). Counts one sweep."""
        self.sweeps += 1
        return self.PHI @ w + self.gamma * (self._P2 @ V).reshape(self.S, self.A)

    # ------------------------------------------------------------------ prediction
    def evaluate(self, act, psi_sa=None, tol=1e-10, max_sweeps=200_000):
        """Iterative SF evaluation of the deterministic policy `act` (S,), warm-started at psi_sa.
        Stops when the sup-norm change is <= tol (1 - gamma) / gamma, so the error is <= tol."""
        psi = np.zeros((self.S, self.A, self.D)) if psi_sa is None else psi_sa.copy()
        thr = tol * (1 - self.gamma) / self.gamma
        for _ in range(max_sweeps):
            new = self.backup(psi[self.R, act])
            delta = np.abs(new - psi).max()
            psi = new
            if delta <= thr:
                return psi
        raise RuntimeError("evaluation did not converge")

    # ------------------------------------------------------------------ certificates
    def gap(self, psi_sa, act, w):
        """Bellman gap of the policy `act` with action SFs psi_sa at weight w: (S,), >= 0."""
        Q = psi_sa @ w
        return np.maximum(Q.max(1) - Q[self.R, act], 0.0)

    def layers(self, tail_tol=1e-12):
        """For each relevant state r: the layers L_t(r) (states some policy can occupy at time t from r)
        until they become stationary (L_{t0+1} = L_{t0}); only the support of P is used."""
        if self._layers is None:
            adj = self.P.sum(1) > 0
            T = int(np.ceil(np.log(tail_tol) / np.log(self.gamma)))
            out = []
            for r in self.relevant:
                cur = np.zeros(self.S, dtype=bool); cur[r] = True
                Ls, reach, t0 = [], cur.copy(), None
                for t in range(T):
                    Ls.append(cur)
                    nxt = adj[cur].any(0)
                    if (nxt == cur).all():
                        t0 = t
                        break
                    cur = nxt
                    reach |= cur
                if t0 is None:                              # not stationary within T: tail over all reached
                    t0, tail = T, reach
                    body = Ls
                else:
                    tail, body = Ls[t0], Ls[:t0]
                idx = [np.flatnonzero(L) for L in body]
                nonempty = [t for t, i in enumerate(idx) if len(i)]
                flat = np.concatenate([idx[t] for t in nonempty]) if nonempty else np.zeros(0, int)
                offs = np.cumsum([0] + [len(idx[t]) for t in nonempty[:-1]]) if nonempty else np.zeros(0, int)
                disc = self.gamma ** np.array(nonempty, dtype=float)
                out.append((flat, offs, disc, np.flatnonzero(tail), self.gamma ** t0 / (1 - self.gamma)))
            self._layers = out
        return self._layers

    def reach_bound(self, g):
        """B(r) = sum_t gamma^t max_{L_t(r)} g  >=  V*(r) - V^pi(r), for every relevant r. g: (S,)."""
        g = np.maximum(g, 0.0)
        out = np.empty(len(self.relevant))
        for i, (flat, offs, disc, tail, tf) in enumerate(self.layers()):
            body = disc @ np.maximum.reduceat(g[flat], offs) if len(flat) else 0.0
            out[i] = body + tf * (g[tail].max() if len(tail) else 0.0)
        return out

    # ------------------------------------------------------------------ control
    def improve(self, w, act, psi_sa, allowed=None, stop_bound=0.0, max_iters=1000, tol=1e-10):
        """Policy iteration at weight w from (act, psi_sa = its action SFs). Greedy steps range over
        `allowed` (S, A) boolean (None = all actions); the certificate always uses all actions.
        Stops when the policy is stable or when max_r B(r) <= stop_bound (anytime).
        Returns (act, psi_sa, B, iterations)."""
        act = np.array(act, dtype=int)
        for it in range(max_iters):
            B = self.reach_bound(self.gap(psi_sa, act, w))
            if B.max() <= stop_bound:
                return act, psi_sa, B, it
            Q = psi_sa @ w
            if allowed is not None:
                Q = np.where(allowed, Q, -np.inf)
            best = tol_argmax(Q)
            new = np.where(Q[self.R, act] >= Q[self.R, best] - 1e-12, act, best)
            if np.array_equal(new, act):
                return act, psi_sa, B, it
            act = new
            psi_sa = self.evaluate(act, psi_sa, tol=tol)
        raise RuntimeError("policy iteration did not converge")

    def solve(self, w, act0=None):
        """Optimal policy at w by policy iteration from act0 (ground truth / 'learning from scratch')."""
        act = np.zeros(self.S, dtype=int) if act0 is None else np.asarray(act0, int)
        psi = self.evaluate(act)
        act, psi, B, _ = self.improve(w, act, psi, stop_bound=-1.0)
        return act, psi
