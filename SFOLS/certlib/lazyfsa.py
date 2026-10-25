"""Lazy SF-FSA-VI: Kuric et al.'s high-level value iteration over a small base that grows only where the planner
needs it (FSA-driven Certify-or-Grow), with an exact stopping certificate.

Semantics (Kuric et al., ICAPS 2024): entering exit j in automaton state u moves the automaton to tau(u, j): 'T' (task
done, value 1), another state v, or u itself when the exit's symbol has no edge from u (self-loop), so the high-level
problem is genuinely iterative. Assumption: the jump is discounted by gamma (entering an exit takes a step), which
makes the high-level operator a gamma-contraction also when an exit can be re-entered at once:
    w_j(u) = 1 if tau(u, j) = T,   else   gamma * V_{w(tau(u, j))}(exit_j).

Algorithm (one task; the base L persists across tasks):
    repeat
        W <- Kuric's iteration over L (GPI estimates at the exits) to its fixed point     # zero-shot, no low-level solve
        for every automaton state u:
            if the GPI estimate over L is a Bellman fixed point at w(u): continue      # certified: it equals Q*
            pi <- metapolicy over L at w(u); if its gap is not zero: pi <- pi_hyb, then policy iteration
            add pi (its SFs) to L                                                          # one small refinement
    until no automaton state needed a new policy
The certificate is sound: the GPI estimate Q^ = max_j Q^{pi_j}_w is a lower bound with T Q^ >= Q^, so T Q^ = Q^
forces Q^ = Q*_w. At the stop every w(u) is certified, so GPI over L is exact at the exits, T_L W = T* W = W, and
W = W* (unique fixed point): the executed GPI policy is optimal. Every low-level solve happens at a weight the task
needs. In deterministic domains the d extremes make Q^ = Q* (Theorem 1), so nothing is ever added.

usage: PYTHONPATH=SFOLS python -m certlib.lazyfsa <domain> [n_random_tasks]     domain: office1f2 | office3f2 | windy
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gpi_basis.fsa import symbols

from .experiments import get, library_from, load, save, tasks_for
from .hybrid import Cost, evaluate, gpi, meta_hyb, policy_iteration
from .library import Library
from .mdp import tol_argmax

TOL = 1e-10                     # certificate tolerance (values are O(1))


class Task:
    def __init__(self, mdp, task):
        SYM = symbols(mdp)
        sym_of = {e: s for s, es in SYM.items() for e in es}
        self.U = list(task)
        self.exits = np.array([mdp.exit_index[o] for o in mdp.objects])
        self.tau = {u: [task[u].get(sym_of[o], u) for o in mdp.objects] for u in self.U}

    def weights(self, Vex, g):
        """W(u) from the values Vex[v] (E,) at the exits under each state's own weight."""
        return {u: np.array([1.0 if v == "T" else g * Vex[v][j] for j, v in enumerate(self.tau[u])]) for u in self.U}


# ---------------------------------------------------------------- product MDP (measurement only)
def _product(mdp, T, acts):
    """(I - M, r) of the product MDP over U x S under the per-state policies acts[u]: V = r + M V."""
    S, g, U = mdp.S, mdp.gamma, T.U
    rows, cols, vals, r = [], [], [], np.zeros(len(U) * S)
    for ui, u in enumerate(U):
        a = acts[u]
        P, PHI = mdp.P[mdp.R, a], mdp.PHI[mdp.R, a]
        i, j = np.nonzero(P)
        rows += list(ui * S + i); cols += list(ui * S + j); vals += list(g * P[i, j])
        for e, v in enumerate(T.tau[u]):
            col = PHI[:, e]
            nz = np.flatnonzero(col)
            if v == "T":
                r[ui * S + nz] += col[nz]
            else:
                vi = U.index(v)
                rows += list(ui * S + nz); cols += [vi * S + T.exits[e]] * len(nz); vals += list(g * col[nz])
    M = sp.csr_matrix((vals, (rows, cols)), shape=(len(U) * S,) * 2)
    return sp.identity(len(U) * S, format="csr") - M, r


def product_value(mdp, T, acts):
    A_, r = _product(mdp, T, acts)
    V = spla.spsolve(A_.tocsc(), r)
    return {u: V[i * mdp.S:(i + 1) * mdp.S] for i, u in enumerate(T.U)}


def product_q(mdp, T, V):
    g = mdp.gamma
    Vex = {u: V[u][T.exits] for u in T.U}
    W = T.weights(Vex, g)
    return {u: mdp.PHI @ W[u] + g * (mdp._P2 @ V[u]).reshape(mdp.S, mdp.A) for u in T.U}, W


def product_optimum(mdp, T):
    acts = {u: np.zeros(mdp.S, int) for u in T.U}
    for _ in range(500):
        V = product_value(mdp, T, acts)
        Q, W = product_q(mdp, T, V)
        new = {}
        for u in T.U:
            best = tol_argmax(Q[u])
            new[u] = np.where(Q[u][mdp.R, acts[u]] >= Q[u][mdp.R, best] - 1e-12, acts[u], best)
        if all(np.array_equal(new[u], acts[u]) for u in T.U):
            return acts, V, W
        acts = new
    raise RuntimeError("product PI did not converge")


def product_vi_cost(mdp, T, tol=1e-3):
    """Baseline without SFs: value iteration on the product MDP, then policy iteration (iterative evaluation) to
    zero gap. Cost in full sweeps of the low-level MDP (a product sweep is |U| of them, an evaluation sweep 1/A)."""
    V = {u: np.zeros(mdp.S) for u in T.U}
    n = 0
    while True:
        Q, _ = product_q(mdp, T, V)
        n += len(T.U)
        new = {u: Q[u].max(1) for u in T.U}
        d = max(np.abs(new[u] - V[u]).max() for u in T.U)
        V = new
        if d <= tol:
            break
    acts = {u: tol_argmax(Q[u]) for u in T.U}
    A_, r = None, None
    while True:
        # iterative evaluation of the product policy, warm-started at V
        A_, r = _product(mdp, T, acts)
        M = sp.identity(A_.shape[0], format="csr") - A_
        x = np.concatenate([V[u] for u in T.U])
        thr = 1e-11 * (1 - mdp.gamma) / mdp.gamma
        while True:
            xn = r + M @ x
            n += len(T.U) / mdp.A
            dd, x = np.abs(xn - x).max(), xn
            if dd <= thr:
                break
        V = {u: x[i * mdp.S:(i + 1) * mdp.S] for i, u in enumerate(T.U)}
        Q, _ = product_q(mdp, T, V)
        n += len(T.U)
        new = {}
        for u in T.U:
            best = tol_argmax(Q[u])
            new[u] = np.where(Q[u][mdp.R, acts[u]] >= Q[u][mdp.R, best] - 1e-12, acts[u], best)
        if all(np.array_equal(new[u], acts[u]) for u in T.U):
            return n
        acts = new


# ---------------------------------------------------------------- Kuric's iteration over a base L
def kuric_vi(mdp, T, L, W=None, tol=1e-13, max_it=200_000):
    Sx = L.stack[:, T.exits]                                     # (k, E, A, D): SFs at the exits only
    W = {u: np.zeros(mdp.D) for u in T.U} if W is None else dict(W)
    for it in range(1, max_it + 1):
        Vex = {u: np.einsum("keap,p->kea", Sx, W[u]).max(axis=(0, 2)) for u in T.U}
        Wn = T.weights(Vex, mdp.gamma)
        d = max(np.abs(Wn[u] - W[u]).max() for u in T.U)
        W = Wn
        if d <= tol:
            return W, it
    raise RuntimeError("high-level iteration did not converge")


def gpi_policies(L, W):
    return {u: tol_argmax(np.tensordot(L.stack, w, axes=([3], [0])).max(0)) for u, w in W.items()}


def certified_by(mdp, L, w):
    """Is the GPI estimate over L at w a fixed point of the Bellman optimality operator (then it is Q*_w)?"""
    Qh = np.tensordot(L.stack, w, axes=([3], [0])).max(0)                          # (S, A)
    TQ = mdp.PHI @ w + mdp.gamma * (mdp._P2 @ Qh.max(1)).reshape(mdp.S, mdp.A)
    return (TQ - Qh).max() <= TOL * max(w.max(), 1e-12)


def sf_table(mdp, act, psi0, tol=1e-12):
    """Action SFs of act: state SFs by iteration (warm-started), then one backup. Cost in full sweeps."""
    r, P = mdp.PHI[mdp.R, act], mdp.P[mdp.R, act]
    thr = tol * (1 - mdp.gamma) / mdp.gamma
    psi, n = psi0.copy(), 0
    while True:
        new = r + mdp.gamma * P @ psi
        n += 1
        d, psi = np.abs(new - psi).max(), new
        if d <= thr:
            break
    psa = mdp.PHI + mdp.gamma * (mdp._P2 @ psi).reshape(mdp.S, mdp.A, mdp.D)
    return psa, n * mdp.D / mdp.A + mdp.D


def grow(mdp, L, w, how, cost):
    """A policy optimal at w. how = 'cog': metapolicy over L, then pi_hyb, then policy iteration;
    how = 'pi': GPI over L, then policy iteration. Returns (act, kind, improvement steps)."""
    stack, acts = L.stack, L.acts
    if how == "cog":
        mask = np.zeros((mdp.S, mdp.A), bool)
        for a in acts:
            mask[mdp.R, a] = True
        mh = meta_hyb(mdp, stack, acts, mask, w, cost)
        am, Q = mh["meta"][0], mh["Q"]
        if (Q.max(1) - Q[mdp.R, am]).max() <= TOL * max(w.max(), 1e-12):
            return am, "metapolicy", 0
        act, V = mh["hyb"]
        kind = "hybrid"
    else:
        act, Vh = gpi(stack, w)
        V = evaluate(mdp, act, w, Vh, cost)
        kind = "gpi"
    act, V, _, steps = policy_iteration(mdp, w, act, V, cost)
    return act, (kind if steps == 0 else "policy iteration"), steps


def lazy(mdp, T, L, how, max_rounds=100):
    """Lazy SF-FSA-VI on one task; L is extended in place. Returns W and a log."""
    cost, sfc, checks = Cost(mdp.A), 0.0, 0
    log = dict(rounds=0, hl_iters=0, hits=0, grown=[], diff=[], steps=0)
    W = None
    while True:
        W, it = kuric_vi(mdp, T, L, W)
        log["rounds"] += 1
        log["hl_iters"] += it
        grew = 0
        for u in T.U:
            w = W[u]
            ok = certified_by(mdp, L, w)
            checks += 1 + len(L) / mdp.A                 # one lookahead + forming the GPI table
            if ok:
                log["hits"] += 1
                continue
            act, kind, steps = grow(mdp, L, w, how, cost)
            near = min(int((act != a).sum()) for a in L.acts)
            _, warm = L.gpi(w)
            psa, c = sf_table(mdp, act, warm[mdp.R, act])
            sfc += c
            L.add(act, psa, kind, w)
            log["grown"].append(kind)
            log["diff"].append(near)
            log["steps"] += steps
            grew += 1
        if grew == 0 or log["rounds"] >= max_rounds:
            break
    log.update(sweeps=cost.sweeps, sf_sweeps=sfc, checks=checks, total=cost.sweeps + sfc + checks,
               certified=grew == 0)
    return W, log


def extremes(mdp, run):
    return library_from(mdp, run, mdp.D)


RUNS = {"windy": ("sfols", "1e-5"), "office0f2": ("cl", "1e-6")}


def prefix_needed(mdp, T, full, vopt, ks=(6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600)):
    """Smallest prefix of the SFOLS (OLS-order) CCS for which Kuric's plan is optimal on this task (executed relative
    regret <= 1e-9; exact exit weights are not enough: the policy inside an automaton state can still be wrong)."""
    s0 = int(mdp.relevant[0])
    for k in ks:
        if k > len(full):
            break
        L = Library(mdp)
        L.stack, L.acts = full.stack[:k], full.acts[:k]
        W, _ = kuric_vi(mdp, T, L)
        if vopt - product_value(mdp, T, gpi_policies(L, W))["u0"][s0] <= 1e-9 * vopt:
            return k
    return np.inf


def cmd_prefix(domain, n_random=30):
    """Recompute the OLS-prefix baseline of a saved run with the executed-regret criterion."""
    mdp, _ = get(domain)
    n_random = int(n_random)
    run = load("run", domain, *RUNS.get(domain, ("sfols", "2e-3")), 0.0)
    full = library_from(mdp, run)
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    tasks = tasks[:3 + n_random] if domain.startswith("office") else tasks[:n_random]
    d = load("lazyfsa", domain, n_random)
    for r, task in zip(d["rows"], tasks):
        r["prefix"] = prefix_needed(mdp, Task(mdp, task), full, r["opt"])
    save(d, "lazyfsa", domain, n_random)
    report(domain, d["rows"], d["ccs_size"])


def copy_lib(L):
    M = Library(L.mdp)
    M.stack, M.acts, M.kinds, M.weights = L.stack.copy(), list(L.acts), list(L.kinds), list(L.weights)
    return M


def main(domain, n_random=30):
    n_random = int(n_random)
    mdp, _ = get(domain)
    run = load("run", domain, *RUNS.get(domain, ("sfols", "2e-3")), 0.0)
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    tasks = tasks[:3 + n_random] if domain.startswith("office") else tasks[:n_random]
    s0 = int(mdp.relevant[0])
    base = extremes(mdp, run)
    ccs = library_from(mdp, run)
    print(f"{domain}: {len(tasks)} tasks (Kuric semantics: self-loops on exits without an edge), base = {len(base)} "
          f"extremes, full eps-CCS from SFOLS = {len(ccs)} policies", flush=True)
    stream = {"cog": copy_lib(base), "pi": copy_lib(base)}
    rows = []
    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        T = Task(mdp, task)
        acts_opt, Vopt, Wopt = product_optimum(mdp, T)
        vopt = Vopt["u0"][s0]
        r = dict(i=i, n_states=len(T.U), opt=float(vopt))
        for name, L in (("extremes", base), ("ccs", ccs)):
            W, it = kuric_vi(mdp, T, L)
            r[f"reg_{name}"] = float(vopt - product_value(mdp, T, gpi_policies(L, W))["u0"][s0])
            r[f"it_{name}"] = it
        for how in ("cog", "pi"):
            for mode, L in (("fresh", copy_lib(base)), ("stream", stream[how])):
                n0 = len(L)
                W, log = lazy(mdp, T, L, how)
                reg = float(vopt - product_value(mdp, T, gpi_policies(L, W))["u0"][s0])
                werr = max(np.abs(W[u] - Wopt[u]).max() for u in T.U)
                r[f"{how}_{mode}"] = dict(log, reg=reg, werr=werr, added=len(L) - n0, size=len(L))
        r["vi_cost"] = product_vi_cost(mdp, T)
        r["prefix"] = prefix_needed(mdp, T, ccs, vopt)
        rows.append(r)
        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            print(f"  {i + 1}/{len(tasks)} tasks ({time.perf_counter() - t0:.0f}s); stream base: CoG {len(stream['cog'])}, "
                  f"PI {len(stream['pi'])} policies", flush=True)
    save(dict(rows=rows, ccs_size=len(ccs)), "lazyfsa", domain, n_random)
    report(domain, rows, len(ccs))


def report(domain, rows, ccs_size):
    rel = lambda key: max(max(r[key], 0) / r["opt"] for r in rows)
    print(f"\n{domain}: {len(rows)} tasks. Max relative regret: Kuric + 6 extremes {rel('reg_extremes'):.1e}, "
          f"Kuric + eps-CCS ({ccs_size} policies) {rel('reg_ccs'):.1e}")
    print(f"  product VI from scratch (no SFs): {np.mean([r['vi_cost'] for r in rows]):.0f} sweeps per task")
    pre = np.array([r["prefix"] for r in rows], float)
    print(f"  OLS-order CCS prefix that makes Kuric exact: median {np.median(pre):.0f}, max {pre.max():.0f}, "
          f"never within the eps-CCS on {int(np.isinf(pre).sum())} of {len(rows)} tasks")
    for key in ("cog_fresh", "pi_fresh", "cog_stream", "pi_stream"):
        L = [r[key] for r in rows]
        kinds = [k for l in L for k in l["grown"]]
        nk = {k: kinds.count(k) for k in ("metapolicy", "hybrid", "gpi", "policy iteration")}
        print(f"  {key:11s} max rel regret {max(max(l['reg'], 0) / r['opt'] for l, r in zip(L, rows)):.1e}, "
              f"W error {max(l['werr'] for l in L):.0e}, all certified {all(l['certified'] for l in L)} | per task: "
              f"rounds {np.mean([l['rounds'] for l in L]):.2f}, policies added {np.mean([l['added'] for l in L]):.2f}, "
              f"cost {np.mean([l['total'] for l in L]):.0f} sweeps (solves {np.mean([l['sweeps'] for l in L]):.0f} + SF "
              f"tables {np.mean([l['sf_sweeps'] for l in L]):.0f} + checks {np.mean([l['checks'] for l in L]):.0f}) | "
              f"added policies by kind {nk}; states changed vs nearest stored policy: median "
              f"{np.median([d for l in L for d in l['diff']]) if kinds else 0:.0f}; final base {L[-1]['size']}", flush=True)


if __name__ == "__main__":
    cmd_prefix(*sys.argv[2:]) if sys.argv[1] == "prefix" else main(*sys.argv[1:])
