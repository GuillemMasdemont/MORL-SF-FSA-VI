"""GPI cells along a segment of weight space w(x) = x a + (1 - x) b, x in [0, 1].

Along a segment every estimate Psi_j(s, a).w(x) is a line in x, so at each state GPI's choice changes only
at breakpoints of an upper envelope of lines. Between consecutive breakpoints (over all states) GPI is a
single fixed policy: a 'GPI cell'. Inside a cell the policy's value is linear in x, its Bellman gap is
convex in x, and V* is convex in x, so every loss bound we use is maximised at the cell's endpoints.
"""
import numpy as np

from .core import PolicyRecord, tol_argmax


class Segment:
    def __init__(self, a, b):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.d = self.a - self.b

    def w(self, x):
        return np.multiply.outer(np.asarray(x, float), self.d) + self.b


# ---------------------------------------------------------------------- upper envelopes of lines
def envelope_label_breaks(I, M, labels, lo=0.0, hi=1.0):
    """x in (lo, hi) where the label of the maximising line y = I + M x changes."""
    order = np.lexsort((labels, -I, M))                # slope asc, then intercept desc, then label asc
    I, M, labels = I[order], M[order], labels[order]
    keep = np.ones(len(M), dtype=bool)
    keep[1:] = M[1:] != M[:-1]                         # equal slopes: the first one dominates
    I, M, labels = I[keep], M[keep], labels[keep]
    hull = []
    for k in range(len(M)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            # line i2 is never on top if line k overtakes i1 no later than i2 does
            if (I[i1] - I[k]) * (M[i2] - M[i1]) <= (I[i1] - I[i2]) * (M[k] - M[i1]):
                hull.pop()
            else:
                break
        hull.append(k)
    h = np.array(hull)
    if len(h) == 1:
        return []
    xs = (I[h[:-1]] - I[h[1:]]) / (M[h[1:]] - M[h[:-1]])          # where hull[i+1] takes over
    out = []
    for x, l0, l1 in zip(xs, labels[h[:-1]], labels[h[1:]]):
        if lo < x < hi and l0 != l1:
            out.append(float(x))
    return out


def gpi_breakpoints(stack, seg, states):
    """All x in (0, 1) where GPI (over the pool 'stack' (k, S, A, D)) changes its action at a state."""
    I = stack @ seg.b                                   # (k, S, A)
    M = stack @ seg.d
    k, _, A = I.shape
    labels = np.tile(np.arange(A), k)
    xs = set()
    for s in states:
        xs.update(envelope_label_breaks(I[:, s, :].ravel(), M[:, s, :].ravel(), labels))
    return sorted(xs)


def _merge(xs, tol=1e-13):
    out = [0.0]
    for x in xs:
        if x - out[-1] > tol:
            out.append(x)
    if 1.0 - out[-1] <= tol:
        out[-1] = 1.0
    else:
        out.append(1.0)
    return out


def gpi_cells(pool, seg, cache, states=None):
    """List of (lo, hi, PolicyRecord): the GPI cells of the pool along the segment.

    Every cell is verified: the GPI policy just inside both ends must equal the one at its midpoint
    (otherwise the cell is split); adjacent cells with the same policy are merged."""
    mdp = pool.mdp
    if states is None:                                  # decision states: anything with transitions or features
        states = np.flatnonzero((mdp.P.sum((1, 2)) > 0) | (np.abs(mdp.PHI).sum((1, 2)) > 0))
    edges = _merge(gpi_breakpoints(pool.stack, seg, states))
    I, M = pool.stack @ seg.b, pool.stack @ seg.d

    def act(x):
        return tol_argmax((I + M * x).max(0))

    cells = []
    stackx = [(lo, hi, 0) for lo, hi in zip(edges[:-1], edges[1:])][::-1]
    while stackx:
        lo, hi, depth = stackx.pop()
        mid = 0.5 * (lo + hi)
        a_mid = act(mid)
        d = 1e-7 * (hi - lo)
        if depth < 25 and (not np.array_equal(act(lo + d), a_mid) or not np.array_equal(act(hi - d), a_mid)):
            stackx.append((mid, hi, depth + 1))
            stackx.append((lo, mid, depth + 1))
            continue
        rec = cache.get(a_mid)
        if cells and cells[-1][2].key == rec.key and abs(cells[-1][1] - lo) < 1e-15:
            cells[-1] = (cells[-1][0], hi, rec)
        else:
            cells.append((lo, hi, rec))
    return cells


# ---------------------------------------------------------------------- certification
def chord_upper(known_x, known_V, x):
    """Upper bound on V*(r)(x) from exact values at known weights: V* is convex in x, so it lies below
    the chord between neighbouring known points. known_V: (n_known, n_rel). Returns (n_rel,)."""
    order = np.argsort(known_x)
    kx, kv = np.asarray(known_x)[order], np.asarray(known_V)[order]
    return np.array([np.interp(x, kx, kv[:, r]) for r in range(kv.shape[1])])


def certify(mdp, cells, seg, known_x, known_V, use_reach=True, use_chord=True, eps=0.0, refine=32):
    """Bounds on GPI's loss V*(r) - V^GPI(r) at the relevant states r, everywhere on the segment.

    Per cell (a fixed policy) and relevant state, two upper bounds are convex in x: the reachability-
    discounted Bellman certificate and the chord bound (V* convex, V^pi linear). A convex function is
    maximised at an interval's ends, so on [a, b] the loss is at most min(max_ends(reach), max_ends(chord)).
    Cells whose bound is above eps are re-checked on `refine` equal sub-intervals, which tightens it.

    Returns dict with
      cert     : certified max loss over the whole segment (valid for EVERY x)
      vertices : list of (x, pointwise bound at x) at every evaluated point (query candidates)
      sup_cert : the crude certificate max_s gap(s)/(1-gamma) (all states) at the cell ends."""
    rel, nrel = mdp.relevant, len(mdp.relevant)
    kx = np.asarray(known_x); order = np.argsort(kx)
    kx, kv = kx[order], np.asarray(known_V)[order]
    vert, vchord, cert, sup_cert = {}, {}, 0.0, 0.0

    def points(rec, xs):
        W = seg.w(xs)                                                    # (n, D)
        G = np.maximum((rec.psi_sa @ W.T).max(1) - rec.psi_s @ W.T, 0.0).T   # (n, S) Bellman gaps
        gmax = G.max(1) / (1 - mdp.gamma)
        reach = mdp.reach_bound(G) if use_reach else np.repeat(gmax[:, None], nrel, 1)
        if use_chord:
            ub = np.stack([np.interp(xs, kx, kv[:, r]) for r in range(nrel)], 1)
            chord = np.maximum(ub - W @ rec.psi_s[rel].T, 0.0)
        else:
            chord = np.full((len(xs), nrel), np.inf)
        for x, b, c in zip(xs, np.minimum(reach, chord).max(1), chord.max(1)):
            vert[float(x)] = max(vert.get(float(x), 0.0), float(b))
            vchord[float(x)] = max(vchord.get(float(x), 0.0), float(c))
        return reach, chord, gmax

    for lo, hi, rec in cells:
        xs = np.array([lo, hi])
        reach, chord, gmax = points(rec, xs)
        sup_cert = max(sup_cert, gmax.max())
        bound = float(np.minimum(reach.max(0), chord.max(0)).max())
        if bound > eps and refine > 1 and hi - lo > 1e-12:
            xs = np.linspace(lo, hi, refine + 1)
            reach, chord, _ = points(rec, xs)
            r2 = np.maximum(reach[:-1], reach[1:]); c2 = np.maximum(chord[:-1], chord[1:])
            bound = float(np.minimum(r2, c2).max())
        cert = max(cert, bound)
    return dict(cert=cert, vertices=sorted(vert.items()), vertices_chord=sorted(vchord.items()),
                sup_cert=sup_cert)


# ---------------------------------------------------------------------- true loss (for evaluation)
def true_loss(pool, seg, cache, xgrid, vstar_grid, upto=None):
    """Exact loss of executed GPI on a grid: returns (loss_at_relevant (n,), loss_all_states (n,)).
    vstar_grid: (n, S) optimal values at the grid weights."""
    mdp = pool.mdp
    sub = pool if upto is None else _Prefix(pool, upto)
    cells = gpi_cells(sub, seg, cache)
    los = np.array([c[0] for c in cells])
    idx = np.clip(np.searchsorted(los, xgrid, side="right") - 1, 0, len(cells) - 1)
    rel = mdp.relevant
    lr, la = np.empty(len(xgrid)), np.empty(len(xgrid))
    for ci in np.unique(idx):
        m = idx == ci
        W = seg.w(xgrid[m])                                       # (n_m, D)
        V = W @ cells[ci][2].psi_s.T                              # (n_m, S)
        diff = vstar_grid[m] - V
        lr[m] = diff[:, rel].max(1)
        la[m] = diff.max(1)
    return np.maximum(lr, 0), np.maximum(la, 0)


class _Prefix:
    """View of the first `upto` policies of a pool (pools only grow, so prefixes = history)."""
    def __init__(self, pool, upto):
        self.mdp, self.stack = pool.mdp, pool.stack[:upto]


def vstar_on_grid(mdp, seg, xgrid):
    """Optimal values at every state for each grid weight (policy iteration warm-started along x)."""
    out, pi = np.empty((len(xgrid), mdp.S)), None
    for i, x in enumerate(xgrid):
        pi, V, _ = mdp.optimal(seg.w(x), init=pi)
        out[i] = V
    return out
