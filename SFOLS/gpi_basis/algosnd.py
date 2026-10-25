"""Basis-construction algorithms on the full weight simplex (any d).

Same three query rules as in algos1d (sfols / gpils / targeted) and the same two ways of obtaining the
policy at a queried weight (full solve / Certify-or-Grow). In d > 2 GPI's cells are not enumerated;
the targeted rule scores a candidate set (corners of the pool's value envelope, random weights, and
points on the low-dimensional faces) by the same pointwise loss bound, so it is a heuristic search for
GPI's worst weight -- its quality is measured on held-out weights, not certified.
"""
import heapq
import itertools
import time

import numpy as np

from .core import Pool, PolicyCache
from .polytope import corner_weights, lp_upper

META_TOL = 1e-9


class GrowthND:
    def __init__(self, mdp, cache=None):
        self.mdp = mdp
        self.cache = cache if cache is not None else PolicyCache(mdp)
        self.pool = Pool(mdp)
        self.base = []
        self.known_W, self.known_V = [], []       # exact V*(relevant) at queried weights
        self.c = dict(queries=0, full=0, meta_ok=0, meta_fail=0, sweeps_full=0, sweeps_meta=0)
        self.log = []
        self.t0 = time.perf_counter()
        self.lin0 = mdp.stats["lin_solves"]

    def _record(self, w, kind):
        self.log.append(dict(self.c, w=np.asarray(w).copy(), kind=kind, pool=len(self.pool),
                             lin_solves=self.mdp.stats["lin_solves"] - self.lin0,
                             time=time.perf_counter() - self.t0))

    def _add(self, w, actions, V, kind):
        rec = self.cache.get(actions)
        self.pool.add(rec, kind, w)
        self.known_W.append(np.asarray(w, float)); self.known_V.append(V[self.mdp.relevant])
        return rec

    def full(self, w, init=None):
        if init is None and len(self.pool):
            init = self.pool.actions(w)
        pi, V, n = self.mdp.optimal(w, init=init)
        self.c["full"] += 1; self.c["sweeps_full"] += n
        self.base.append(pi)
        return self._add(w, pi, V, "full")

    def grow(self, w, how):
        self.c["queries"] += 1
        if how == "full" or not self.base:
            rec = self.full(w); self._record(w, "full"); return rec
        own = np.stack(self.base)
        qh = self.pool.qhat(w)
        init = qh[np.arange(self.mdp.S)[None, :], own].argmax(0)
        act, V, Q, n = self.mdp.meta(w, own, init_choice=init)
        self.c["sweeps_meta"] += n
        if (Q.max(1) - V).max() <= META_TOL:
            self.c["meta_ok"] += 1
            rec = self._add(w, act, V, "meta"); self._record(w, "meta"); return rec
        self.c["meta_fail"] += 1
        rec = self.full(w, init=Q.argmax(1)); self._record(w, "full"); return rec


def _mu(mdp, rec):
    return rec.psi_s[mdp.relevant].mean(0)


def run_sfols(mdp, how="full", tol=1e-6, max_queries=10_000, cache=None, verbose=False, exact=False,
              priority="envelope"):
    """SFOLS (exact corner enumeration); how='cog' gives Certify-or-Grow (same queries, cheaper answers)."""
    D = mdp.D
    G = GrowthND(mdp, cache)
    ccs, W_exp, V_exp = [], [], []

    def lower(w):
        """What the priority compares the optimistic bound with: the best single policy (SFOLS) or the
        realized value of GPI over the whole pool (GPI-aware ranking of the same corners)."""
        if priority == "envelope":
            return max(p @ w for p in ccs)
        return _mu(mdp, G.cache.get(G.pool.actions(w))) @ w
    cnt = itertools.count()
    heap = [(-np.inf, next(cnt), tuple(np.eye(D)[i])) for i in range(D)]
    seen = set()
    key = lambda w: tuple(np.round(w, 9))
    while heap and G.c["queries"] < max_queries:
        negp, _, wt = heapq.heappop(heap)
        w = np.array(wt)
        if key(w) in seen:
            continue
        if np.isfinite(negp):                      # lazy re-scoring, as in OLS
            d = lp_upper(w, W_exp, V_exp) - lower(w)
            if d <= tol:
                seen.add(key(w)); continue
            if heap and d < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-d, next(cnt), wt)); continue
        seen.add(key(w))
        rec = G.grow(w, how)
        psi = _mu(mdp, rec)
        W_exp.append(w); V_exp.append(psi @ w)
        if ccs and psi @ w <= max(p @ w for p in ccs) + 1e-9:
            continue
        ccs.append(psi)
        if len(ccs) < 2:
            continue
        queued = {key(np.array(h[2])) for h in heap}
        for wc in corner_weights(np.array(ccs), exact=exact):
            k = key(wc)
            if k in seen or k in queued:
                continue
            d = lp_upper(wc, W_exp, V_exp) - lower(wc)
            if d > tol:
                heapq.heappush(heap, (-d, next(cnt), tuple(wc)))
            else:
                seen.add(k)
        if verbose and G.c["queries"] % 10 == 0:
            print(f"  sfols queries {G.c['queries']} ccs {len(ccs)} queue {len(heap)}", flush=True)
    G.ccs = ccs
    return G


def run_gpils(mdp, how="full", max_queries=10_000, cache=None, exact=False):
    """GPI-LS (Alegre et al. 2023): corners of the value set picked by v^GPI_w - max_pi v^pi_w."""
    D = mdp.D
    G = GrowthND(mdp, cache)
    ccs, visited = [], set()
    key = lambda w: tuple(np.round(w, 9))
    for i in range(D):
        w = np.eye(D)[i]
        rec = G.grow(w, how); visited.add(key(w))
        psi = _mu(mdp, rec)
        if not ccs or psi @ w > max(p @ w for p in ccs) + 1e-9:
            ccs.append(psi)
    while G.c["queries"] < max_queries:
        cand = [w for w in corner_weights(np.array(ccs), exact=exact) if key(w) not in visited]
        if not cand:
            break
        prio = []
        for w in cand:
            rec = G.cache.get(G.pool.actions(w))
            prio.append(_mu(mdp, rec) @ w - max(p @ w for p in ccs))
        w = cand[int(np.argmax(prio))]
        visited.add(key(w))
        rec = G.grow(w, how)
        psi = _mu(mdp, rec)
        if psi @ w > max(p @ w for p in ccs) + 1e-9:
            ccs.append(psi)
    return G


def face_samples(D, groups, n_per, rng):
    """Random weights on the faces spanned by unions of 1 or 2 symbol groups (where FSA weights live)."""
    out = []
    for k in (1, 2):
        for combo in itertools.combinations(range(len(groups)), k):
            idx = [i for g in combo for i in groups[g]]
            for _ in range(n_per):
                w = np.zeros(D); w[idx] = rng.dirichlet(np.ones(len(idx))); out.append(w)
    return out


def score_candidates(G, cand, top_lp=40):
    """Pointwise bound on GPI's loss at the relevant states: max_r min(reach_r, LP_r - V^GPI_r).
    The reachability certificate is computed for every candidate, the LP bound only for the top ones."""
    mdp = G.mdp
    rel = mdp.relevant
    recs = [G.cache.get(G.pool.actions(w)) for w in cand]
    reach = np.array([mdp.reach_bound(r.gap(w)) for r, w in zip(recs, cand)])        # (n, n_rel)
    vpi = np.array([r.psi_s[rel] @ w for r, w in zip(recs, cand)])
    score = reach.max(1)
    order = np.argsort(-score)[:top_lp]
    for i in order:
        ub = lp_upper(cand[i], G.known_W, G.known_V)
        score[i] = np.minimum(reach[i], np.maximum(ub - vpi[i], 0.0)).max()
    return score


def run_targeted(mdp, how="cog", eps=1e-6, max_queries=10_000, batch=1, n_random=300, n_face=20,
                 groups=None, seed=0, cache=None, verbose=False, pool_corners=True):
    """Ours in d dims: score candidates by a loss bound on GPI and query the worst."""
    D = mdp.D
    rng = np.random.default_rng(seed)
    G = GrowthND(mdp, cache)
    for i in range(D):
        G.grow(np.eye(D)[i], "full")
    G.score_log = []
    carry = []
    while G.c["queries"] < max_queries:
        cand = [rng.dirichlet(np.ones(D)) for _ in range(n_random)] + carry
        if groups is not None:
            cand += face_samples(D, groups, n_face, rng)
        if pool_corners and len(G.pool) >= 2:
            cand += [w for w in corner_weights(np.array([_mu(mdp, r) for r in G.pool.records]), exact=False)]
        known = {tuple(np.round(w, 9)) for w in G.known_W}
        cand = [w for w in cand if tuple(np.round(w, 9)) not in known]
        score = score_candidates(G, cand)
        order = np.argsort(-score)
        G.score_log.append(dict(pool=len(G.pool), best=float(score[order[0]]), **G.c))
        if verbose:
            print(f"  pool {len(G.pool):4d} full {G.c['full']:3d} best bound {score[order[0]]:.2e}", flush=True)
        if score[order[0]] <= eps:
            break
        for i in order[:batch]:
            G.grow(cand[i], how)
        carry = [cand[i] for i in order[batch:batch + 50]]
    return G


def run_probe(mdp, max_queries=10_000, n_random=200, n_face=10, groups=None, seed=0, tol=1e-9, cache=None):
    """Lower-bound-driven growth: at each candidate weight compare GPI's realized value with the
    metapolicy's (restricted PI over the base, warm-started at GPI). V_meta <= V*, so V_meta - V^GPI is a
    GUARANTEED loss of GPI. Add the metapolicy at the candidate with the largest guaranteed loss (it is
    certified-optimal there if its Bellman gap is 0; otherwise it is still a better policy than GPI).
    If no candidate shows a positive guaranteed loss, query the candidate with the largest upper bound
    with Certify-or-Grow (exploration: the base may lack actions there)."""
    D = mdp.D
    rng = np.random.default_rng(seed)
    G = GrowthND(mdp, cache)
    for i in range(D):
        G.grow(np.eye(D)[i], "full")
    rel = mdp.relevant
    G.c.update(probes=0, meta_uncertified=0)
    while G.c["queries"] < max_queries:
        cand = [rng.dirichlet(np.ones(D)) for _ in range(n_random)]
        if groups is not None:
            cand += face_samples(D, groups, n_face, rng)
        own = np.stack(G.base)
        best, best_w, best_meta = 0.0, None, None
        for w in cand:
            gpi = G.cache.get(G.pool.actions(w))
            qh = G.pool.qhat(w)
            init = qh[np.arange(mdp.S)[None, :], own].argmax(0)
            act, V, Q, n = mdp.meta(w, own, init_choice=init)
            G.c["probes"] += 1; G.c["sweeps_meta"] += n
            gain = float((V[rel] - gpi.psi_s[rel] @ w).max())
            if gain > best:
                best, best_w, best_meta = gain, w, (act, V, Q)
        G.c["queries"] += 1
        if best > tol:
            act, V, Q = best_meta
            gap = (Q.max(1) - V).max()
            kind = "meta" if gap <= META_TOL else "meta_unc"
            rec = G.cache.get(act)
            G.pool.add(rec, kind, best_w)
            if gap <= META_TOL:
                G.c["meta_ok"] += 1
                G.known_W.append(best_w); G.known_V.append(V[rel])
            else:
                G.c["meta_uncertified"] += 1
            G._record(best_w, kind)
        else:
            score = score_candidates(G, cand)
            G.c["queries"] -= 1                     # grow() counts the query itself
            G.grow(cand[int(np.argmax(score))], "cog")
    return G
