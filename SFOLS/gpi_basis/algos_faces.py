"""Basis construction on a UNION of segments of the weight simplex, with one shared GPI pool.

Typical use: the three symbol edges {C1,C2}, {M1,M2}, {O1,O2} of Office, which contain every weight an FSA
task with out-degree 1 can produce. Every segment gets its own exact GPI cells and certificate; the pool
(and the metapolicy's base) is shared, so a policy found for one edge is used on all of them.
"""
import heapq
import itertools

import numpy as np

from .algosnd import GrowthND
from .segment import Segment, certify, chord_upper, envelope_label_breaks, gpi_cells


def on_segment(seg, w, tol=1e-9):
    w = np.asarray(w, float)
    x = float(np.dot(w - seg.b, seg.d) / np.dot(seg.d, seg.d))
    if -tol <= x <= 1 + tol and np.allclose(seg.w(x), w, atol=tol):
        return min(max(x, 0.0), 1.0)
    return None


def known_on(G, seg):
    kx, kv = [], []
    for w, v in zip(G.known_W, G.known_V):
        x = on_segment(seg, w)
        if x is not None:
            kx.append(x); kv.append(v)
    return kx, kv


def endpoints(segs):
    out = []
    for s in segs:
        for e in (s.a, s.b):
            if not any(np.allclose(e, o) for o in out):
                out.append(e.copy())
    return out


def certify_all(G, segs, eps):
    """Certificate on every segment; returns (max cert, list of (bound, chord bound, w) candidates)."""
    cert, cand = 0.0, []
    for seg in segs:
        kx, kv = known_on(G, seg)
        cells = gpi_cells(G.pool, seg, G.cache)
        cr = certify(G.mdp, cells, seg, kx, kv, eps=eps)
        cert = max(cert, cr["cert"])
        chord = dict(cr["vertices_chord"])
        ks = set(kx)
        for x, b in cr["vertices"]:
            if x not in ks:
                cand.append((b, chord.get(x, b), seg.w(x)))
    return cert, cand


def run_targeted_multi(mdp, segs, how="cog", eps=1e-6, select="chord", max_queries=10_000, verbose=False):
    G = GrowthND(mdp)
    for w in endpoints(segs):
        G.grow(w, "full")
    G.cert_log = []
    while True:
        cert, cand = certify_all(G, segs, eps)
        G.cert_log.append(dict(pool=len(G.pool), cert=cert, **G.c))
        if verbose:
            print(f"  pool {len(G.pool):4d} full {G.c['full']:3d} cert {cert:.2e}", flush=True)
        if cert <= eps or G.c["queries"] >= max_queries or not cand:
            break
        i = 0 if select == "bound" else 1
        best = max(cand, key=lambda c: c[i])
        G.grow(best[2], how)
    G.final_cert = cert
    return G


def run_ols_multi(mdp, segs, how="full", tol=1e-6, priority="envelope", max_queries=10_000):
    """OLS over the union of segments with a global queue.
    priority='envelope': SFOLS (optimistic chord bound minus the best single policy);
    priority='gpi'     : the same corners ranked by the chord bound minus GPI's realized value;
    priority='gpils'   : GPI-LS (GPI's realized value minus the best single policy)."""
    G = GrowthND(mdp)
    rel = mdp.relevant
    mu = lambda rec: rec.psi_s[rel].mean(0)
    for w in endpoints(segs):
        G.grow(w, how)
    explored = set(tuple(np.round(w, 9)) for w in G.known_W)
    cnt = itertools.count()

    def corners_and_prio():
        out = []
        for seg in segs:
            # the CCS of this segment: policies in the pool, by their mu-value lines on the segment
            P = np.array([mu(r) for r in G.pool.records])
            I, M = P @ seg.b, P @ seg.d
            xs = [0.0] + envelope_label_breaks(I, M, np.arange(len(P))) + [1.0]
            kx, kv = known_on(G, seg)
            kmu = [np.mean(v) for v in kv]
            for x in xs:
                w = seg.w(x)
                if tuple(np.round(w, 9)) in explored:
                    continue
                env = (P @ w).max()
                ub = chord_upper(kx, [[v] for v in kmu], x)[0]
                if priority == "envelope":
                    p = ub - env
                else:
                    vg = mu(G.cache.get(G.pool.actions(w))) @ w
                    p = ub - vg if priority == "gpi" else vg - env
                out.append((p, x, w))
        return out

    while G.c["queries"] < max_queries:
        cand = corners_and_prio()
        if priority != "gpils":
            cand = [c for c in cand if c[0] > tol]
        if not cand:
            break
        p, x, w = max(cand, key=lambda c: c[0])
        explored.add(tuple(np.round(w, 9)))
        G.grow(w, how)
    return G
