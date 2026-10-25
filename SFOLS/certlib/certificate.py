"""Global certificate for GPI over an SF library on a face of the weight simplex.

For every relevant state r and every weight w of the face F:
    lower   E_r(w) = max_{pi in L, a} psi^pi(r, a).w              (the GPI estimate at r, which the
                                                                    planner reads; GPI's value is >= it)
    upper   U_r(w) = min { sum_q l_q u_q(r) : sum_q l_q w_q = w, l >= 0 }
            from query records (w_q, u_q) with u_q(r) >= V*_{w_q}(r)  (V* is convex in w)
    gap     U_r(w) - E_r(w)  >=  V*_w(r) - V^{GPI_L}_w(r).
On each region where E_r is affine, U_r - E_r is convex, so the largest gap over F is attained at a corner
of E_r (a vertex of the regions). Gaps only decrease as records and policies are added (U decreases, E
increases), so corners are kept in a max-heap and re-scored lazily.
"""
import heapq
import itertools

import numpy as np
from scipy.optimize import linprog

from gpi_basis.polytope import corner_weights
from gpi_basis.segment import envelope_label_breaks


def _float_corners(P):
    """Corner weights of max_i P_i.w over the simplex (k = P.shape[1] >= 3), by qhull halfspace intersection in
    reduced coordinates x = (w_1..w_{k-1}), w_k = 1 - sum x (no equality constraint), with the epigraph capped
    above. Sanity check: the k simplex vertices are always corners; if any is missing, or qhull fails, the
    corners are recomputed in exact rational arithmetic."""
    from scipy.spatial import HalfspaceIntersection, QhullError
    n, k = P.shape
    cap = P.max() + 1.0
    A = np.hstack([P[:, :-1] - P[:, -1:], -np.ones((n, 1))])          # v >= p.w  <=>  A [x, v] + p_k <= 0
    H = [np.hstack([A, P[:, -1:]]),
         np.hstack([-np.eye(k - 1), np.zeros((k - 1, 1)), np.zeros((k - 1, 1))]),            # x >= 0
         np.r_[np.ones(k - 1), 0.0, -1.0][None, :],                                        # sum x <= 1
         np.r_[np.zeros(k - 1), 1.0, -cap][None, :]]                                       # v <= cap
    H = np.vstack(H)
    interior = np.r_[np.full(k - 1, 1.0 / k), cap - 0.5]
    try:
        pts = HalfspaceIntersection(H, interior, qhull_options="QJ").intersections
    except (QhullError, ValueError):
        return [np.asarray(c) for c in corner_weights(P, exact=True)]
    pts = pts[pts[:, -1] < cap - 1e-9]
    W = np.hstack([pts[:, :-1], 1 - pts[:, :-1].sum(1, keepdims=True)])
    W = np.clip(W, 0, None)
    W = W / W.sum(1, keepdims=True)
    W = np.unique(np.round(W, 10), axis=0)
    have = {tuple(np.round(w, 10)) for w in W}
    if not all(tuple(np.eye(k)[i]) in have for i in range(k)):
        return [np.asarray(c) for c in corner_weights(P, exact=True)]
    return list(W)


def _region_vertices(p, Q, denom=10 ** 12):
    """Vertices of R_p = {w in the simplex : (p - q).w >= 0 for all rows q of Q} (the region where the new
    point p is on top of the old envelope), by exact rational vertex enumeration (the region is small)."""
    import cdd
    from fractions import Fraction
    k = len(p)
    F = lambda x: Fraction(float(x)).limit_denominator(denom)
    rows = [[0] + [F(v) for v in (p - q)] for q in Q]
    rows += [[0] + [1 if j == i else 0 for j in range(k)] for i in range(k)]
    rows += [[-1] + [1] * k]
    mat = cdd.Matrix(rows, number_type="fraction")
    mat.rep_type = cdd.RepType.INEQUALITY
    mat.lin_set = frozenset([len(rows) - 1])
    gens = cdd.Polyhedron(mat).get_generators()
    out = []
    for i in range(gens.row_size):
        g = gens[i]
        if g[0] != 1 or i in gens.lin_set:
            continue
        w = np.array([float(x) for x in g[1:]])
        w = np.clip(w, 0, None)
        out.append(w / w.sum())
    return out


def _region_vertices_float(p, Q, tie=1e-9):
    """Same as _region_vertices with qhull: halfspace intersection in reduced coordinates from the region's
    Chebyshev centre (one LP). Returns None if the region has no interior (p is nowhere strictly on top)."""
    from scipy.spatial import HalfspaceIntersection, QhullError
    k = len(p)
    D = p[None, :] - Q                                         # (p - q).w >= 0
    A = -(D[:, :-1] - D[:, -1:])                               # A x + b <= 0 with w = (x, 1 - sum x)
    b = -D[:, -1]
    A = np.vstack([A, -np.eye(k - 1), np.ones((1, k - 1))])
    b = np.r_[b, np.zeros(k - 1), -1.0]
    nrm = np.linalg.norm(A, axis=1)
    keep = nrm > 1e-14                                         # drop void constraints (q = p)
    A, b, nrm = A[keep], b[keep], nrm[keep]
    res = linprog(np.r_[np.zeros(k - 1), -1.0], A_ub=np.hstack([A, nrm[:, None]]), b_ub=-b,
                  bounds=[(None, None)] * (k - 1) + [(0, 1)], method="highs")
    if not res.success or res.x[-1] <= 1e-10:
        return None
    try:
        pts = HalfspaceIntersection(np.hstack([A, b[:, None]]), res.x[:-1], qhull_options="QJ").intersections
    except (QhullError, ValueError):
        return _region_vertices(p, Q)
    W = np.hstack([pts, 1 - pts.sum(1, keepdims=True)])
    W = np.clip(W, 0, None)
    W = W / W.sum(1, keepdims=True)
    return list(np.unique(np.round(W, 11), axis=0))


def _lower_hull_interp(xs, us, x):
    """Lower convex envelope of the points (xs, us) evaluated at x (1-D version of the LP bound)."""
    o = np.lexsort((us, xs))
    xs, us = np.asarray(xs)[o], np.asarray(us)[o]
    keep = np.r_[True, xs[1:] != xs[:-1]]                 # same x: keep the smallest u
    xs, us = xs[keep], us[keep]
    hull = []
    for i in range(len(xs)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (us[b] - us[a]) * (xs[i] - xs[a]) >= (us[i] - us[a]) * (xs[b] - xs[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    h = np.array(hull)
    return np.interp(x, xs[h], us[h])


class Certificate:
    def __init__(self, mdp, face, exact_final=True, use_hull=True):
        self.mdp = mdp
        self.F = np.asarray(face, int)                    # coordinates spanning the face
        self.k = len(self.F)
        self.nr = len(mdp.relevant)
        self.W, self.U = [], []                           # records: weights (D,), upper values (nr,)
        self.P = [np.zeros((0, mdp.D)) for _ in range(self.nr)]
        self.heap, self.cnt = [], itertools.count()
        self.corners = [set() for _ in range(self.nr)]
        self.cpos = [{} for _ in range(self.nr)]          # corner key -> full-precision weight
        self.regions = [[] for _ in range(self.nr)]       # vertices of each stored point's linearity region
        self.exact_final = exact_final
        self._plane_cache = [(-1, None, np.inf) for _ in range(self.nr)]
        self.use_hull = use_hull                         # hull of records (fast, ~1e-10 accurate) or LP

    # ------------------------------------------------------------------ geometry
    def embed(self, wf):
        w = np.zeros(self.mdp.D); w[self.F] = wf
        return w

    def _corners(self, r, exact=False):
        P = self.P[r][:, self.F]
        if self.k == 1:
            return [self.embed(np.ones(1))]
        if self.k == 2:
            I, M = P[:, 1], P[:, 0] - P[:, 1]             # w(x) = x e_F0 + (1 - x) e_F1
            xs = [0.0, 1.0] + envelope_label_breaks(I, M, np.arange(len(P)))
            return [self.embed(np.array([x, 1 - x])) for x in sorted(set(xs))]
        if exact:
            return [self.embed(c) for c in corner_weights(P, exact=True)]
        return [self.embed(c) for c in _float_corners(P)]

    def _face_records(self, r):
        """Records inside the face, merged when they sit at the same weight (smallest upper value kept)."""
        Wq = np.asarray(self.W); Uq = np.asarray(self.U)[:, r]
        inside = np.abs(Wq[:, np.setdiff1d(np.arange(self.mdp.D), self.F)]).sum(1) < 1e-12
        Wq, Uq = Wq[inside][:, self.F], Uq[inside]
        keys = np.round(Wq, 9)
        order = np.lexsort(np.vstack([Uq] + [keys[:, i] for i in range(keys.shape[1])][::-1]))
        Wq, Uq, keys = Wq[order], Uq[order], keys[order]
        first = np.r_[True, (np.abs(np.diff(keys, axis=0)).sum(1) > 0)]
        return Wq[first], Uq[first]

    def _planes(self, r):
        """Lower facets of the convex hull of the merged records (face coordinates, value): U_r is the
        maximum of their affine functions. Ill-conditioned (near-vertical) facets are dropped and the result
        is capped by the largest record value; the final certificate is re-checked with the exact LP."""
        if self._plane_cache[r][0] == len(self.W):
            return self._plane_cache[r][1]
        from scipy.spatial import ConvexHull, QhullError
        Wq, Uq = self._face_records(r)
        pts = np.hstack([Wq[:, :-1], Uq[:, None]])
        planes = None
        try:
            eq = ConvexHull(pts).equations                     # unit normals: n.x + c <= 0 inside
            low = eq[eq[:, -2] < -1e-6]
            planes = np.hstack([-low[:, :-2], -low[:, -1:]]) / low[:, -2:-1]     # u = a.x + b
        except (QhullError, ValueError):
            planes = None
        self._plane_cache[r] = (len(self.W), planes, Uq.max() if len(Uq) else np.inf)
        return planes

    def upper_lp(self, w, r):
        Wq, Uq = self._face_records(r)
        res = linprog(Uq, A_eq=Wq.T, b_eq=w[self.F], bounds=[(0, None)] * len(Wq), method="highs")
        return float(res.fun) if res.success else np.inf

    def upper(self, w, r):
        if self.k >= 3 and self.use_hull:
            pl = self._planes(r)
            if pl is not None and len(pl):
                return float(min((pl[:, :-1] @ w[self.F[:-1]] + pl[:, -1]).max(), self._plane_cache[r][2]))
        Wq = np.asarray(self.W); Uq = np.asarray(self.U)[:, r]
        inside = np.abs(Wq[:, np.setdiff1d(np.arange(self.mdp.D), self.F)]).sum(1) < 1e-12
        Wq, Uq = Wq[inside], Uq[inside]
        if self.k == 2:
            return float(_lower_hull_interp(Wq[:, self.F[0]], Uq, w[self.F[0]]))
        res = linprog(Uq, A_eq=Wq[:, self.F].T, b_eq=w[self.F], bounds=[(0, None)] * len(Wq), method="highs")
        return float(res.fun) if res.success else np.inf

    def lower(self, w, r):
        return float((self.P[r] @ w).max())

    def gap(self, w, r):
        return self.upper(w, r) - self.lower(w, r)

    # ------------------------------------------------------------------ updates
    def add_record(self, w, u):
        self.W.append(np.asarray(w, float)); self.U.append(np.asarray(u, float))

    def add_policy(self, psi_ra, exact=False):
        """psi_ra: (nr, A, D) action SFs of a new library policy at the relevant states (each action at r
        gives one point of the envelope E_r). Recomputes the corners of every envelope it raises."""
        for r in range(self.nr):
            new = psi_ra[r]
            if self.k >= 3 and len(self.P[r]):
                self._add_full(r, new)
                continue
            if len(self.P[r]):
                if not self._raises(r, new):
                    continue
            self.P[r] = np.vstack([self.P[r], new])
            self._push_corners(r, self._corners(r, exact=exact))

    def _add_full(self, r, new, tie=1e-12):
        """Insert the points of `new` that lie above E_r somewhere (exact test at the current corners:
        p.w - E_r(w) is affine on every linearity region) and update the linearity regions. A stored point's
        region changes only if a new point is above it at one of that region's vertices (affine difference,
        so this is exact); only those regions and the new points' regions are recomputed. Points whose
        region has no interior are dropped (they never strictly attain the maximum)."""
        pos = self.cpos[r]
        Cm = np.array(list(pos.values()))
        E = (Cm @ self.P[r].T).max(1)
        raise_ = [p for p in new if (Cm @ p - E).max() > tie]
        if not raise_:
            return
        F = self.F
        regs = self.regions[r]
        if len(regs) != len(self.P[r]):                         # first use: build every region
            regs = [None] * len(self.P[r])
        affected = set()
        for i, q in enumerate(self.P[r]):
            if regs[i] is None:
                affected.add(i); continue
            V = regs[i]
            if any(((V[:, F] @ (p[F] - q[F])).max() > tie) for p in raise_):
                affected.add(i)
        n0 = len(self.P[r])
        self.P[r] = np.vstack([self.P[r]] + [np.asarray(raise_)])
        regs = list(regs) + [None] * len(raise_)
        affected |= set(range(n0, len(self.P[r])))
        P = self.P[r][:, F]
        for i in affected:
            reg = _region_vertices_float(P[i], np.delete(P, i, axis=0))
            regs[i] = np.array([self.embed(c) for c in reg]) if reg is not None and len(reg) else None
        alive = [i for i in range(len(P)) if regs[i] is not None]
        self.P[r] = self.P[r][alive]
        self.regions[r] = [regs[i] for i in alive]
        corners = {}
        for V in self.regions[r]:
            for w in V:
                corners[tuple(np.round(w, 10))] = w
        for key in corners:
            if key not in pos:
                heapq.heappush(self.heap, (-np.inf, next(self.cnt), r, key))
        self.cpos[r] = corners
        self.corners[r] = set(corners)

    def _raises(self, r, new):
        """Is some point of `new` above E_r somewhere on the face? (LP per point; exact for k <= 2.)"""
        if self.k <= 2:
            cs = self._corners(r)
            # E_r is convex and piecewise affine, new points are affine: above somewhere iff above at a corner
            return any(p @ c > self.lower(c, r) + 1e-13 for p in new for c in cs)
        P = self.P[r][:, self.F]
        for p in new[:, self.F]:
            # max t  s.t.  (p - P_j).w >= t for all j, w in the face simplex
            A_ub = np.hstack([-(p - P), np.ones((len(P), 1))])
            res = linprog(np.r_[np.zeros(self.k), -1.0], A_ub=A_ub, b_ub=np.zeros(len(P)),
                          A_eq=np.r_[np.ones(self.k), 0.0][None, :], b_eq=[1.0],
                          bounds=[(0, None)] * self.k + [(None, None)], method="highs")
            if res.success and -res.fun > 1e-13:
                return True
        return False

    def _push_corners(self, r, cs):
        cur = set()
        self.cpos[r] = {}
        for c in cs:
            key = tuple(np.round(c, 10))
            cur.add(key)
            self.cpos[r][key] = np.asarray(c, float)
            if key not in self.corners[r]:
                heapq.heappush(self.heap, (-np.inf, next(self.cnt), r, key))
        self.corners[r] = cur

    # ------------------------------------------------------------------ queries
    def worst(self):
        """Pop-and-rescore until the top entry is exact; returns (gap, r, w) of the largest current gap
        (or None if no corner is left). The entry stays in the heap."""
        while self.heap:
            negp, _, r, key = self.heap[0]
            if key not in self.corners[r]:
                heapq.heappop(self.heap); continue
            w = np.array(key)
            g = self.gap(w, r)
            if -negp <= g + 1e-15:                         # already exact
                return g, r, w
            heapq.heapreplace(self.heap, (-g, next(self.cnt), r, key))
        return None

    def drop(self, r, w):
        """Remove a corner (after it has been queried; gaps there are now <= the accepted bound)."""
        key = tuple(np.round(w, 10))
        self.corners[r].discard(key)
        self.cpos[r].pop(key, None)

    def full_corners(self, r):
        """All corners of E_r recomputed from scratch: the vertices of every point's linearity region."""
        P = self.P[r][:, self.F]
        out = {}
        for i in range(len(P)):
            reg = _region_vertices_float(P[i], np.delete(P, i, axis=0)) if len(P) > 1 else [np.eye(self.k)[j] for j in range(self.k)]
            for c in reg or []:
                w = self.embed(c)
                out[tuple(np.round(w, 10))] = w
        return out

    def verify_lp(self, eps, margin=1e-7):
        """Exact re-check of the stopping decision with the LP upper bound. The hull bound is within 1e-8 of
        the LP (validated), so only corners whose hull gap is within `margin` of eps can be above eps; only
        those are re-checked. Corners above eps are re-pushed with their exact gap (the loop continues)."""
        bad = []
        if self.k < 3 or not self.use_hull:
            return bad
        for r in range(self.nr):
            for key, w in self.cpos[r].items():
                if self.gap(w, r) <= eps - margin:
                    continue
                g = self.upper_lp(w, r) - self.lower(w, r)
                if g > eps:
                    bad.append((r, w))
                    heapq.heappush(self.heap, (-g, next(self.cnt), r, key))
        return bad

    def verify_full(self, eps):
        """Final check for k >= 3: recompute every envelope's corners from scratch, add the ones the
        incremental bookkeeping does not have, and return those whose gap exceeds eps."""
        bad = []
        for r in range(self.nr):
            for key, w in self.full_corners(r).items():
                if key not in self.cpos[r]:
                    self.cpos[r][key] = w
                    self.corners[r].add(key)
                    heapq.heappush(self.heap, (-np.inf, next(self.cnt), r, key))
                    if self.gap(w, r) > eps:
                        bad.append((r, w))
        return bad

    def verify_exact(self, eps):
        """Final check for k >= 3: exact (rational) corner enumeration of every envelope; returns the
        corners whose gap exceeds eps (pushes them so that the loop continues)."""
        bad = []
        for r in range(self.nr):
            cs = self._corners(r, exact=True)
            missing = [c for c in cs if tuple(np.round(c, 10)) not in self.corners[r]]
            for c in missing:
                if self.gap(c, r) > eps:
                    bad.append((r, c))
            if missing:
                self.corners[r] |= {tuple(np.round(c, 10)) for c in missing}
                for c in missing:
                    heapq.heappush(self.heap, (-np.inf, next(self.cnt), r, tuple(np.round(c, 10))))
        return bad

    def max_gap_exhaustive(self, exact=True):
        """Largest gap over every current corner of every envelope (for reporting)."""
        best = (0.0, None, None)
        for r in range(self.nr):
            for c in self._corners(r, exact=exact):
                g = self.gap(c, r)
                if g > best[0]:
                    best = (g, r, c)
        return best
