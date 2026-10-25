"""True (exact) quality of a GPI pool on held-out weights, for d-dimensional weight spaces."""
import itertools

import numpy as np


def test_weights(D, groups, rng, n_random=600, n_face=60):
    """Held-out weights: random on the simplex, random on the faces spanned by 1 and 2 symbol groups
    (where FSA weights live), and the extrema. Returns (W, labels)."""
    W, lab = [], []
    for _ in range(n_random):
        W.append(rng.dirichlet(np.ones(D))); lab.append("simplex")
    if groups is not None:
        for k in (1, 2):
            for combo in itertools.combinations(range(len(groups)), k):
                idx = [i for g in combo for i in groups[g]]
                for _ in range(n_face):
                    w = np.zeros(D); w[idx] = rng.dirichlet(np.ones(len(idx)))
                    W.append(w); lab.append(f"face{k}")
    for i in range(D):
        W.append(np.eye(D)[i]); lab.append("extreme")
    return np.array(W), np.array(lab)


def vstar_table(mdp, W):
    """Optimal values at every state for each weight (rows)."""
    out = np.empty((len(W), mdp.S))
    pi = None
    for i, w in enumerate(W):
        pi, V, _ = mdp.optimal(w, init=pi)
        out[i] = V
    return out


def gpi_loss(mdp, pool, W, Vstar, upto=None):
    """Exact loss of executed GPI over the first `upto` pool policies, at the relevant states.
    Returns (n,) = max over relevant states of V*(r) - V^GPI(r) for every weight."""
    rel = mdp.relevant
    out = np.empty(len(W))
    seen = {}
    for i, w in enumerate(W):
        act = pool.actions(w, upto)
        k = act.tobytes()
        psi = seen.get(k)
        if psi is None:
            psi = mdp.sf(act)[rel]
            seen[k] = psi
        out[i] = (Vstar[i, rel] - psi @ w).max()
    return np.maximum(out, 0.0)
