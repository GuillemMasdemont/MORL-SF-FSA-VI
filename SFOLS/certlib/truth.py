"""Ground truth for reporting only (never used by the algorithms): optimal values on weight grids and the
true loss of GPI over a library, both computed with solve-free policy iteration / evaluation."""
import numpy as np

from .mdp import tol_argmax


def vstar(mdp, W, act0=None):
    """V*_w at every state for each weight row of W (policy iteration warm-started along the list)."""
    out, act = np.empty((len(W), mdp.S)), act0
    for i, w in enumerate(W):
        act, psi = mdp.solve(w, act)
        out[i] = psi[mdp.R, act] @ w
    return out


def gpi_loss(mdp, lib, W, VS, cache=None):
    """True loss V*_w(r) - V^{GPI_L}_w(r) at the relevant states for each weight row of W. (n, nr)."""
    cache = {} if cache is None else cache
    out = np.empty((len(W), len(mdp.relevant)))
    s0 = mdp.sweeps
    for i, w in enumerate(W):
        act = tol_argmax(np.tensordot(lib.stack, w, axes=([3], [0])).max(0))
        key = act.tobytes()
        if key not in cache:
            cache[key] = mdp.evaluate(act, tol=1e-12)[mdp.relevant, act[mdp.relevant]]
        out[i] = VS[i, mdp.relevant] - cache[key] @ w
    mdp.sweeps = s0                                    # reporting is not part of the algorithm's cost
    return np.maximum(out, 0.0)
