"""Per-task certified SF-FSA-VI: GPI over a small base, then a few value-iteration sweeps per automaton state
that certify (or repair) it. Solve-free: scalar Bellman backups only.

For an automaton state u with weight w = w(u), three monotone scalar sequences (one backup each per step):
  U  <- T* U    from U0 = V^ + max(rho+)/(1-g)   (super-solution: U >= V*_w, decreasing)
  Lg <- T^G Lg  from V^                           (V^ = GPI estimate is a sub-solution of the GPI policy:
                                                    Lg <= V^{GPI}_w, increasing)
  Lr <- T* Lr   from V^                           (sub-solution of T*: Lr <= V^{greedy(Lr)}_w, increasing)
Stop at u when, on the states where u is entered (R_u),
  U - Lg <= eps_u                    -> execute GPI (certified as is), or
  U - Lr <= eps_u and Lg converged   -> execute the greedy policy of the last T* backup of Lr (repaired).
Composition (one-hot exit features, sum_i psi_i <= 1, so V*_w is 1-Lipschitz in w): the planner propagates
the lower values (achieved by the executed policies), and the task-level upper bound is
  Ubar_u(s) = U_u(s) + max_{exits e of edges u->v} (Ubar_v(e) - L_v(e)),
so the certified regret at the start is Ubar_u0(start) - L_u0(start) <= sum over a path of eps_u.
'scratch' mode is the same planner without a base: Lr from 0, U from max(w) (both valid since rewards are
exit features in [0, max w] and episodes end at exits); it is the baseline the base must beat.
"""
import time

import numpy as np

from gpi_basis.fsa import _order, symbols

from .mdp import tol_argmax


def depth(task):
    d = {}
    for u in _order(task):
        d[u] = 1 + max([d[v] for v in task[u].values() if v != "T"], default=0)
    return d["u0"]


def entry_states(mdp, task):
    """R_u: states where automaton state u can start (the start for u0, the exits of edges into u)."""
    ex, SYM = mdp.exit_index, symbols(mdp)
    R = {u: set() for u in task}
    R["u0"].add(int(mdp.relevant[0]))
    for u, edges in task.items():
        for sym, v in edges.items():
            if v != "T":
                R[v].update(ex[e] for e in SYM[sym])
    return {u: np.array(sorted(s)) for u, s in R.items()}


def certify_state(mdp, w, stack, Ru, eps_u, scratch=False, max_iters=50000):
    """Two-sided value iteration at weight w. Returns (act, L, U, kind, scalar sweeps)."""
    g, n = mdp.gamma, 0

    def T(V):
        nonlocal n
        n += 1
        return mdp.PHI @ w + g * (mdp._P2 @ V).reshape(mdp.S, mdp.A)

    if scratch:
        Lr, U = np.zeros(mdp.S), np.full(mdp.S, w.max())
        Lg, gact = None, None
    else:
        Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
        Vh = Qh.max(1)
        gact = tol_argmax(Qh)
        Q = T(Vh)
        # pointwise min of super-solutions is a super-solution: the residual shift, the trivial max(w), and
        # sum_i w_i V*_{e_i} (a conic combination of the extremes' optimal values, when the base holds them)
        U = np.minimum(Vh + np.maximum(Q.max(1) - Vh, 0.0).max() / (1 - g), w.max())
        if len(stack) >= mdp.D:
            ext = np.stack([stack[i][:, :, i].max(1) for i in range(mdp.D)])      # V*_{e_i}, base = extremes
            U = np.minimum(U, w @ ext)
        Lg, Lr = Vh.copy(), Vh.copy()
    ract, conv = None, 1e-3 * eps_u * (1 - g) / g
    for _ in range(max_iters):
        if Lg is not None and (U[Ru] - Lg[Ru]).max() <= eps_u:
            return gact, Lg, U, "gpi", n
        if ract is not None and (U[Ru] - Lr[Ru]).max() <= eps_u and (Lg is None or dLg <= conv):
            return ract, Lr, U, "repaired" if Lg is not None else "scratch", n
        U = np.minimum(U, T(U).max(1))
        Q = T(Lr)
        ract, Lr = tol_argmax(Q), np.maximum(Lr, Q.max(1))
        if Lg is not None:
            new = np.maximum(Lg, T(Lg)[mdp.R, gact])
            dLg, Lg = np.abs(new - Lg).max(), new
    raise RuntimeError("certification did not converge")


def plan_task(mdp, task, stack, eps, scratch=False):
    """Certified plan for one task. Returns dict(acts, bound, lower, sweeps, kinds, seconds)."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    R = entry_states(mdp, task)
    eps_u = eps / depth(task)
    L, Ubar, acts, kinds, sweeps = {}, {}, {}, {}, 0
    t0 = time.perf_counter()
    for u in _order(task):
        w, slack = np.zeros(mdp.D), 0.0
        for sym, v in task[u].items():
            for e in SYM[sym]:
                if v == "T":
                    w[objs.index(e)] = 1.0
                else:
                    w[objs.index(e)] = L[v][ex[e]]
                    slack = max(slack, Ubar[v][ex[e]] - L[v][ex[e]])
        acts[u], L[u], U, kinds[u], n = certify_state(mdp, w, stack, R[u], eps_u, scratch)
        Ubar[u] = U + slack
        sweeps += n
    s = int(mdp.relevant[0])
    return dict(acts=acts, bound=float(Ubar["u0"][s] - L["u0"][s]), lower=float(L["u0"][s]),
                sweeps=sweeps, kinds=kinds, seconds=time.perf_counter() - t0)


def executed_value(mdp, task, acts, cache):
    """True value at the start of executing acts[u] in every automaton state (reporting only)."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    psi = {}
    for u, a in acts.items():
        k = a.tobytes()
        if k not in cache:
            cache[k] = mdp.evaluate(a, tol=1e-12)[mdp.R, a]
        psi[u] = cache[k]
    Wt = {}
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else psi[v][ex[e]] @ Wt[v]
        Wt[u] = w
    return float(psi["u0"][int(mdp.relevant[0])] @ Wt["u0"])


def main(domain, n_random, eps):
    """python -m certlib.pertask <domain> <n_random_tasks> <eps>: zero-shot GPI vs certified per-task
    planning over the d extreme policies vs the same certified planner from scratch (no base)."""
    import os
    from .experiments import fname, get, load, save, tasks_for
    from .fsa import library_value, optimal_value
    from .library import Library
    mdp, _ = get(domain)
    n_random, eps = int(n_random), float(eps)
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    n_paper = 3 if domain.startswith("office") else 0
    tasks = tasks[:n_paper + n_random]
    opt_all = load("taskopt", domain) if os.path.exists(fname("taskopt", domain)) else None
    cache = {}
    opt = [opt_all[i] if opt_all is not None else optimal_value(mdp, t, cache) for i, t in enumerate(tasks)]
    s0, t0 = mdp.sweeps, time.perf_counter()
    lib = Library(mdp)
    for i in range(mdp.D):
        w = np.eye(mdp.D)[i]
        a, psi = mdp.solve(w)
        lib.add(a, psi, "learned", w)
    base_sweeps, base_sec = mdp.sweeps - s0, time.perf_counter() - t0
    rows = []
    for k, t in enumerate(tasks):
        zs = library_value(mdp, t, lib, cache)
        cg = plan_task(mdp, t, lib.stack, eps)
        sc = plan_task(mdp, t, lib.stack, eps, scratch=True)
        r = dict(opt=opt[k], zero_shot=opt[k] - zs, depth=depth(t), states=len(t),
                 cert_regret=opt[k] - executed_value(mdp, t, cg["acts"], cache), cert_bound=cg["bound"],
                 cert_sweeps=cg["sweeps"], cert_sec=cg["seconds"], kinds=list(cg["kinds"].values()),
                 scr_regret=opt[k] - executed_value(mdp, t, sc["acts"], cache), scr_bound=sc["bound"],
                 scr_sweeps=sc["sweeps"], scr_sec=sc["seconds"])
        rows.append(r)
        print(f"  task {k:3d} |U|={r['states']} H={r['depth']} opt {r['opt']:.4f} | zero-shot regret "
              f"{r['zero_shot']:.1e} | certified: regret {r['cert_regret']:.1e} bound {r['cert_bound']:.1e} "
              f"sweeps {r['cert_sweeps']:4d} {r['kinds']} | scratch: regret {r['scr_regret']:.1e} "
              f"sweeps {r['scr_sweeps']:5d}", flush=True)
    a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "kinds"}
    kinds = [x for r in rows for x in r["kinds"]]
    print(f"\n{domain}: {len(tasks)} tasks, eps {eps:g}, base = {mdp.D} extremes "
          f"({base_sweeps} SF sweeps, {base_sec:.1f}s, once)")
    print(f"  zero-shot GPI       max regret {a['zero_shot'].max():.2e}  mean {a['zero_shot'].mean():.2e}")
    print(f"  certified (base)    max regret {a['cert_regret'].max():.2e}  max bound {a['cert_bound'].max():.2e}  "
          f"scalar sweeps/task mean {a['cert_sweeps'].mean():.0f} max {a['cert_sweeps'].max()}  "
          f"time/task mean {a['cert_sec'].mean():.3f}s  states: gpi {kinds.count('gpi')} repaired "
          f"{kinds.count('repaired')}")
    print(f"  certified (scratch) max regret {a['scr_regret'].max():.2e}  max bound {a['scr_bound'].max():.2e}  "
          f"scalar sweeps/task mean {a['scr_sweeps'].mean():.0f} max {a['scr_sweeps'].max()}  "
          f"time/task mean {a['scr_sec'].mean():.3f}s")
    print(f"  bound valid on every task: {bool(np.all(a['cert_regret'] <= a['cert_bound'] + 1e-9))} (base), "
          f"{bool(np.all(a['scr_regret'] <= a['scr_bound'] + 1e-9))} (scratch)")
    save(dict(rows=rows, base_sweeps=base_sweeps, eps=eps), "pertask", domain, eps, len(tasks))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:4])


def plan_evaluated(mdp, task, stack, tol=1e-6):
    """Execute GPI over `stack`; the planner's exit values are GPI's ACTUAL values, from iterative evaluation of
    the GPI policy started at the GPI estimate (a sub-solution, so the iterates increase to V^GPI): prediction
    only, no control step. Returns (acts, scalar sweeps)."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    g, thr = mdp.gamma, tol * (1 - mdp.gamma) / mdp.gamma
    V, acts, n = {}, {}, 0
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else V[v][ex[e]]
        Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
        act, L = tol_argmax(Qh), Qh.max(1)
        r = (mdp.PHI @ w)[mdp.R, act]
        P = mdp.P[mdp.R, act]
        while True:
            new = r + g * P @ L
            n += 1
            d, L = np.abs(new - L).max(), new
            if d <= thr:
                break
        V[u], acts[u] = L, act
    return acts, n


def plan_gap(mdp, task, stack, eps, tol=1e-10, allowed=None):
    """The user's scheme. In reverse topological order, at w(u) (built from the executed policies' actual values):
    evaluate GPI's policy (prediction), read its Bellman gap g = max_a Q - V, and certify it when the reach bound
    B(r) = sum_t g^t max_{L_t(r)} g <= eps_u at the entry states; otherwise learn (policy iteration over all
    actions from GPI's policy) until B <= eps_u. Task-level bound: Ubar_u = V_u + B_u + max slack downstream.
    Returns dict(acts, bound, kinds, sweeps (scalar), learn_steps)."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    g, thr = mdp.gamma, tol * (1 - mdp.gamma) / mdp.gamma
    R = entry_states(mdp, task)
    ridx = {int(s): i for i, s in enumerate(mdp.relevant)}
    eps_u = eps / depth(task)
    V, Ubar, acts, kinds, n, steps = {}, {}, {}, {}, 0, 0

    def evaluate(act, w, L):
        nonlocal n
        r, P = (mdp.PHI @ w)[mdp.R, act], mdp.P[mdp.R, act]
        while True:
            new = r + g * P @ L
            n += 1
            d, L = np.abs(new - L).max(), new
            if d <= thr:
                return L

    for u in _order(task):
        w, slack = np.zeros(mdp.D), 0.0
        for sym, v in task[u].items():
            for e in SYM[sym]:
                if v == "T":
                    w[objs.index(e)] = 1.0
                else:
                    w[objs.index(e)] = V[v][ex[e]]
                    slack = max(slack, Ubar[v][ex[e]] - V[v][ex[e]])
        if len(stack):
            Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
            act = tol_argmax(Qh)
            Vp = evaluate(act, w, Qh.max(1))                # GPI estimate is a sub-solution: warm start
        else:                                               # no base: learning from an arbitrary policy
            act = np.zeros(mdp.S, dtype=int)
            Vp = evaluate(act, w, np.zeros(mdp.S))
        kinds[u] = "gpi"
        while True:
            Q = mdp.PHI @ w + g * (mdp._P2 @ Vp).reshape(mdp.S, mdp.A)
            n += 1
            B = mdp.reach_bound(np.maximum(Q.max(1) - Vp, 0.0))
            Bu = np.array([B[ridx[int(s)]] for s in R[u]])
            if Bu.max() <= eps_u:
                break
            if allowed is not None and kinds[u] == "gpi":     # metapolicy: PI restricted to the base's actions
                kinds[u] = "meta"
                while True:
                    Qa = np.where(allowed, Q, -np.inf)
                    best = tol_argmax(Qa)
                    new = np.where(Qa[mdp.R, act] >= Qa[mdp.R, best] - 1e-12, act, best)
                    if np.array_equal(new, act):
                        break
                    act = new
                    Vp = evaluate(act, w, Vp)
                    Q = mdp.PHI @ w + g * (mdp._P2 @ Vp).reshape(mdp.S, mdp.A)
                    n += 1
                    steps += 1
                continue                                      # re-check the gap over ALL actions
            kinds[u] = "learned"
            best = tol_argmax(Q)
            act = np.where(Q[mdp.R, act] >= Q[mdp.R, best] - 1e-12, act, best)
            Vp = evaluate(act, w, Vp)
            steps += 1
        Bfull = np.zeros(mdp.S); Bfull[mdp.relevant] = B
        V[u], acts[u], Ubar[u] = Vp, act, Vp + Bfull + slack
    s = int(mdp.relevant[0])
    return dict(acts=acts, bound=float(Ubar["u0"][s] - V["u0"][s]), kinds=kinds, sweeps=n, learn_steps=steps)
