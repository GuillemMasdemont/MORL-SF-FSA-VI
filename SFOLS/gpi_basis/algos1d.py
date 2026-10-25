"""Basis-construction algorithms along a segment of weight space (d = 2, or a 1-D face of a simplex).

All of them build the same kind of object -- a GPI pool -- and differ in WHERE they query and HOW they
obtain the policy at the queried weight:

  where:  'sfols'    corner weights of the best-single-policy envelope, ranked by OLS' optimistic gap
          'gpils'    the same corners, ranked by how much GPI already beats the envelope (Alegre et al. 2023)
          'targeted' vertices of GPI's own cells, ranked by a certified bound on GPI's loss (ours)
  how:    'full'     exact policy iteration over the full action set (= learning a new policy from scratch)
          'cog'      Certify-or-Grow: the metapolicy over the base policies' actions; only if its Bellman
                     certificate fails, a full solve (warm-started at the metapolicy's greedy policy)
"""
import heapq
import itertools
import time

import numpy as np

from .core import Pool, PolicyCache, PolicyRecord
from .segment import certify, chord_upper, envelope_label_breaks, gpi_cells

META_TOL = 1e-9          # Bellman-gap tolerance to call a metapolicy optimal (values are O(1))


class Growth:
    """Pool + base + exact values at queried weights, with full cost accounting."""

    def __init__(self, mdp, seg, cache=None):
        self.mdp, self.seg = mdp, seg
        self.cache = cache if cache is not None else PolicyCache(mdp)
        self.pool = Pool(mdp)
        self.base = []                       # action tables of fully solved policies (metapolicy's B(s))
        self.known_x, self.known_V = [], []  # exact V*(relevant) at queried weights (for chord bounds)
        self.c = dict(queries=0, full=0, meta_ok=0, meta_fail=0, sweeps_full=0, sweeps_meta=0)
        self.log = []
        self.t0 = time.perf_counter()
        self.lin0 = mdp.stats["lin_solves"]

    def _record(self, x, kind):
        self.log.append(dict(self.c, x=x, kind=kind, pool=len(self.pool),
                             lin_solves=self.mdp.stats["lin_solves"] - self.lin0,
                             time=time.perf_counter() - self.t0))

    def _add(self, x, actions, V, kind):
        rec = self.cache.get(actions)
        self.pool.add(rec, kind, self.seg.w(x))
        self.known_x.append(float(x)); self.known_V.append(V[self.mdp.relevant])
        return rec

    def full(self, x, init=None):
        w = self.seg.w(x)
        if init is None and len(self.pool):
            init = self.pool.actions(w)                     # warm start at GPI
        pi, V, n = self.mdp.optimal(w, init=init)
        self.c["full"] += 1; self.c["sweeps_full"] += n
        self.base.append(pi)
        return self._add(x, pi, V, "full")

    def grow(self, x, how):
        self.c["queries"] += 1
        if how == "full" or not self.base:
            rec = self.full(x)
            self._record(x, "full")
            return rec
        w = self.seg.w(x)
        own = np.stack(self.base)
        qh = self.pool.qhat(w)
        init = qh[np.arange(self.mdp.S)[None, :], own].argmax(0)    # base action GPI ranks highest
        act, V, Q, n = self.mdp.meta(w, own, init_choice=init)
        self.c["sweeps_meta"] += n
        if (Q.max(1) - V).max() <= META_TOL:
            self.c["meta_ok"] += 1
            rec = self._add(x, act, V, "meta")
            self._record(x, "meta")
            return rec
        self.c["meta_fail"] += 1
        rec = self.full(x, init=Q.argmax(1))
        self._record(x, "full")
        return rec


def _mu(mdp, rec):
    return rec.psi_s[mdp.relevant].mean(0)


def _corners(lines, seg):
    """Corner weights (in x) of the upper envelope of linear values psi . w(x)."""
    P = np.asarray(lines)
    I, M = P @ seg.b, P @ seg.d
    return [0.0] + envelope_label_breaks(I, M, np.arange(len(P))) + [1.0]


def run_sfols(mdp, seg, how="full", tol=1e-6, max_queries=10_000, cache=None):
    """SFOLS (Alegre et al. 2022) along the segment; with how='cog' this is Certify-or-Grow (our paper)."""
    G = Growth(mdp, seg, cache)
    ccs, explored = [], {}
    cnt = itertools.count()
    heap = [(-np.inf, next(cnt), 1.0), (-np.inf, next(cnt), 0.0)]
    env = lambda x: max(p @ seg.w(x) for p in ccs)
    while heap and G.c["queries"] < max_queries:
        _, _, x = heapq.heappop(heap)
        if x in explored:
            continue
        rec = G.grow(x, how)
        psi = _mu(mdp, rec)
        explored[x] = psi @ seg.w(x)                      # exact optimal value (solved or certified)
        if ccs and psi @ seg.w(x) <= env(x) + 1e-9:
            continue
        ccs.append(psi)
        heap = [h for h in heap if h[0] == -np.inf]
        kx, kv = list(explored), [[v] for v in explored.values()]
        for xc in _corners(ccs, seg):
            if xc in explored:
                continue
            d = chord_upper(kx, kv, xc)[0] - env(xc)          # = OLS' LP optimistic improvement in 1-D
            if d > tol:
                heapq.heappush(heap, (-d, next(cnt), xc))
        heapq.heapify(heap)
    G.ccs = ccs
    return G


def run_gpils(mdp, seg, how="full", max_queries=10_000, cache=None):
    """GPI-LS (Alegre et al. 2023): corners of the current value set, picked by v^GPI_w - max_pi v^pi_w."""
    G = Growth(mdp, seg, cache)
    ccs, visited = [], set()
    for x in (1.0, 0.0):
        rec = G.grow(x, how); visited.add(x); ccs.append(_mu(mdp, rec))
    while G.c["queries"] < max_queries:
        cand = [x for x in _corners(ccs, seg) if x not in visited]
        if not cand:
            break
        env = lambda x: max(p @ seg.w(x) for p in ccs)
        prio = []
        for x in cand:
            rec = G.cache.get(G.pool.actions(seg.w(x)))
            prio.append(_mu(mdp, rec) @ seg.w(x) - env(x))
        x = cand[int(np.argmax(prio))]
        visited.add(x)
        rec = G.grow(x, how)
        psi = _mu(mdp, rec)
        if psi @ seg.w(x) > env(x) + 1e-9:
            ccs.append(psi)
    return G


def run_targeted(mdp, seg, how="cog", eps=1e-6, max_queries=10_000, batch=1, use_reach=True,
                 use_chord=True, cache=None, verbose=False, select="bound"):
    """Ours: query where GPI's certified loss bound is largest; stop when GPI is certified eps-optimal
    at the relevant states for EVERY weight on the segment."""
    G = Growth(mdp, seg, cache)
    for x in (1.0, 0.0):
        G.grow(x, "full")
    G.cert_log = []
    while True:
        cells = gpi_cells(G.pool, seg, G.cache)
        cr = certify(mdp, cells, seg, G.known_x, G.known_V, use_reach=use_reach, use_chord=use_chord, eps=eps)
        G.cert_log.append(dict(pool=len(G.pool), cert=cr["cert"], sup_cert=cr["sup_cert"],
                               cells=len(cells), **G.c))
        if verbose:
            print(f"  pool {len(G.pool):4d} full {G.c['full']:3d} cells {len(cells):4d} "
                  f"cert {cr['cert']:.2e} sup {cr['sup_cert']:.2e}", flush=True)
        if cr["cert"] <= eps or G.c["queries"] >= max_queries:
            break
        known = set(G.known_x)
        use_bound = select == "bound" or (select == "alt" and len(G.cert_log) % 2 == 0)
        pts = cr["vertices"] if use_bound else cr["vertices_chord"]
        cand = [(b, x) for x, b in pts if x not in known]
        for b, x in sorted(cand, reverse=True)[:batch]:
            G.grow(x, how)
    G.final_cert = cr
    return G
