"""SF library: action-SF tables of every stored policy (learned or composed), with GPI over them."""
import numpy as np

from .mdp import tol_argmax


class Library:
    def __init__(self, mdp):
        self.mdp = mdp
        self.stack = np.zeros((0, mdp.S, mdp.A, mdp.D))        # action SFs, (n, S, A, D)
        self.acts = []                                           # each policy's actions (S,)
        self.kinds = []                                          # 'learned' | 'gpi' | 'meta'
        self.weights = []                                        # weight it was obtained for

    def __len__(self):
        return len(self.acts)

    def add(self, act, psi_sa, kind, w):
        self.stack = np.concatenate([self.stack, psi_sa[None]], 0)
        self.acts.append(np.asarray(act, int)); self.kinds.append(kind)
        self.weights.append(np.asarray(w, float))

    def state_sfs(self, r):
        """psi^pi(r) of every stored policy: (n, D)."""
        return np.stack([self.stack[j, r, a[r]] for j, a in enumerate(self.acts)])

    def qhat(self, w):
        """GPI estimate max_j psi_j(s, a).w: (S, A)."""
        return np.tensordot(self.stack, w, axes=([3], [0])).max(0)

    def gpi(self, w):
        """GPI's actions at w and the warm-start table psi_{j*(s,a)}(s, a) (the stored policy attaining the
        GPI estimate at each (s, a); its value is a lower bound on the GPI policy's action values)."""
        Q = np.tensordot(self.stack, w, axes=([3], [0]))                   # (n, S, A)
        j = Q.argmax(0)
        warm = np.take_along_axis(self.stack, j[None, :, :, None], 0)[0]   # (S, A, D)
        return tol_argmax(Q.max(0)), warm

    def base_mask(self):
        """B(s): actions taken by the learned policies, (S, A) boolean."""
        m = np.zeros((self.mdp.S, self.mdp.A), dtype=bool)
        for a, k in zip(self.acts, self.kinds):
            if k == "learned":
                m[np.arange(self.mdp.S), a] = True
        return m
