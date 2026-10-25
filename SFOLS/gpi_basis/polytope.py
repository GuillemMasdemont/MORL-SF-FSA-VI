"""Corner weights of a linear-value envelope and OLS' LP upper bound, robust to near-degenerate points.

Float pycddlib silently drops vertices when value vectors are nearly degenerate (seen on slippery Office:
65 corners found instead of 748, so OLS stopped with a true gap 15x above its tolerance). Here vertices
are enumerated in exact rational arithmetic, after removing every point that is optimal for no weight.
"""
from fractions import Fraction

import cdd
import numpy as np
from scipy.optimize import linprog


def envelope_points(P, tol=1e-12):
    """Rows of P that are the best value P_i.w for some weight w in the simplex (one LP per row)."""
    P = np.unique(np.round(np.asarray(P, float), 12), axis=0)
    n, m = P.shape
    keep = []
    for i in range(n):
        others = np.delete(P, i, axis=0)
        if len(others) == 0:
            keep.append(i)
            continue
        # max t  s.t.  (P_i - P_j).w >= t  for all j != i,  w in the simplex
        A_ub = np.hstack([-(P[i] - others), np.ones((len(others), 1))])
        res = linprog(np.r_[np.zeros(m), -1.0], A_ub=A_ub, b_ub=np.zeros(len(others)),
                      A_eq=np.r_[np.ones(m), 0.0][None, :], b_eq=[1.0],
                      bounds=[(0, None)] * m + [(None, None)], method="highs")
        if res.success and -res.fun >= -tol:
            keep.append(i)
    return P[keep]


def corner_weights(P, denom=10 ** 9, exact=True):
    """Corner weights of max_i P_i.w over the simplex. exact=True: rational vertex enumeration (always
    complete); exact=False: float pycddlib as in the published SFOLS code (fast, may miss corners when
    points are nearly degenerate; falls back to exact if cdd raises)."""
    P = envelope_points(P)
    n, m = P.shape
    if n == 1:
        return [np.eye(m)[i] for i in range(m)]
    A = np.vstack([np.hstack([P, -np.ones((n, 1))]),
                   np.hstack([np.ones((1, m)), [[0.0]]]), np.hstack([-np.ones((1, m)), [[0.0]]]),
                   np.hstack([-np.eye(m), np.zeros((m, 1))])])
    b = np.concatenate([np.zeros(n), [1.0, -1.0], np.zeros(m)])
    Mx = np.hstack([b[:, None], -A])
    gens = None
    if not exact:
        try:
            mat = cdd.Matrix(Mx, number_type="float")
            mat.rep_type = cdd.RepType.INEQUALITY
            gens = cdd.Polyhedron(mat).get_generators()
        except RuntimeError:
            gens = None
    if gens is None:
        mat = cdd.Matrix([[Fraction(float(x)).limit_denominator(denom) for x in row] for row in Mx],
                         number_type="fraction")
        mat.rep_type = cdd.RepType.INEQUALITY
        gens = cdd.Polyhedron(mat).get_generators()
    out = []
    for i, g in enumerate(list(gens)):
        g = np.array([float(x) for x in g])
        if g[0] != 1 or i in gens.lin_set:
            continue
        w = np.clip(g[1:m + 1], 0, None)
        out.append(w / w.sum())
    return out


def lp_upper(w, W_known, V_known):
    """OLS / Nemecek-Parr upper bound on a convex function V* at w from exact values at known weights:
    min sum_i l_i V*(w_i)  s.t.  sum_i l_i w_i = w, l >= 0   (= max w.v s.t. w_i.v <= V*(w_i)).
    V_known: (n,) or (n, k) -> one bound per column."""
    W_known = np.asarray(W_known, float)
    V_known = np.asarray(V_known, float)
    cols = V_known[:, None] if V_known.ndim == 1 else V_known
    out = np.empty(cols.shape[1])
    for c in range(cols.shape[1]):
        res = linprog(cols[:, c], A_eq=W_known.T, b_eq=np.asarray(w, float),
                      bounds=[(0, None)] * len(W_known), method="highs")
        out[c] = res.fun if res.success else np.inf
    return out[0] if V_known.ndim == 1 else out
