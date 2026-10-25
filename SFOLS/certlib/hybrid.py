"""GPI, metapolicy and hybrid policy over a base, and per-task certified planning built on them (scalar, solve-free).

usage: PYTHONPATH=SFOLS python -m certlib.hybrid curve <domain> <k,k,...>     fraction of held-out weights where
                                                                            GPI / metapolicy / pi_hyb are optimal
       PYTHONPATH=SFOLS python -m certlib.hybrid tasks <domain> <k>         per-task cost of reaching the optimum

Definitions at a weight w, base L (action SFs of k policies), B(s) = actions the base policies take at s:
  GPI       pi_G(s) = argmax_a max_j psi_j(s, a).w
  election  pi_E(s) = pi_j*(s)(s), j*(s) = argmax_j V_j(s)       (start of the metapolicy, actions in B(s))
  meta      policy iteration restricted to B(s) from pi_E: the optimal policy of the MDP restricted to B
  hyb       pi_H(s) = argmax_{a in A} Q_meta(s, a): one improvement step over ALL actions from the metapolicy
Every evaluation is iterative and warm-started at a sub-solution of the evaluated policy (GPI estimate, max_j V_j,
V_meta), so the iterates increase monotonically. Cost unit: one sweep = S*A one-step backups (a greedy step costs 1,
a policy-evaluation sweep 1/A). 'optimal' = zero Bellman gap at every state (policy iteration is stable).
"""
import sys

import numpy as np

from gpi_basis.fsa import _order, symbols

from .experiments import get, library_from, load, save, tasks_for
from .fsa import library_value
from .mdp import tol_argmax
from .pertask import executed_value


class Cost:
    def __init__(self, A):
        self.A, self.evals, self.greedy, self.restricted = A, 0, 0, 0     # restricted: lookaheads over B(s) only

    @property
    def sweeps(self):
        return self.greedy + self.evals / self.A


def evaluate(mdp, act, w, V, cost, tol=1e-11):
    r, P = (mdp.PHI @ w)[mdp.R, act], mdp.P[mdp.R, act]
    thr = tol * (1 - mdp.gamma) / mdp.gamma
    while True:
        new = r + mdp.gamma * P @ V
        cost.evals += 1
        d, V = np.abs(new - V).max(), new
        if d <= thr:
            return V


def lookahead(mdp, V, w, cost):
    cost.greedy += 1
    return mdp.PHI @ w + mdp.gamma * (mdp._P2 @ V).reshape(mdp.S, mdp.A)


def policy_iteration(mdp, w, act, V, cost, allowed=None, max_steps=None):
    """From (act, V = its value). Returns (act, V, Q, steps). Stops when stable (or after max_steps)."""
    steps = 0
    while True:
        Q = lookahead(mdp, V, w, cost)
        Qa = Q if allowed is None else np.where(allowed, Q, -np.inf)
        best = tol_argmax(Qa)
        new = np.where(Qa[mdp.R, act] >= Qa[mdp.R, best] - 1e-12, act, best)
        if np.array_equal(new, act) or (max_steps is not None and steps >= max_steps):
            return act, V, Q, steps
        act = new
        V = evaluate(mdp, act, w, V, cost)
        steps += 1


def base_tables(stack, acts, w):
    Qj = np.tensordot(stack, w, axes=([3], [0]))                          # (k, S, A)
    Vj = np.take_along_axis(Qj, np.stack(acts)[:, :, None], 2)[..., 0]    # (k, S)
    return Qj, Vj


def gpi(stack, w):
    Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
    return tol_argmax(Qh), Qh.max(1)


def meta_hyb(mdp, stack, acts, mask, w, cost):
    """Returns dict with the metapolicy and the hybrid policy (actions, values) at w."""
    Qj, Vj = base_tables(stack, acts, w)
    j = Vj.argmax(0)
    act = np.stack(acts)[j, mdp.R]
    V = evaluate(mdp, act, w, Vj.max(0), cost)                            # max_j V_j: sub-solution of pi_E
    g0 = cost.greedy
    act, V, Q, _ = policy_iteration(mdp, w, act, V, cost, allowed=mask)
    cost.restricted += cost.greedy - g0
    hact = np.where(Q.max(1) > Q[mdp.R, act] + 1e-12, Q.argmax(1), act)  # ties keep the metapolicy's action
    hV = evaluate(mdp, hact, w, V, cost)                                  # V_meta: sub-solution of pi_H
    return dict(meta=(act, V), hyb=(hact, hV), Q=Q)                       # Q: full lookahead of V_meta


# ---------------------------------------------------------------- Figure 2: fraction of optimal weights
def cmd_curve(domain, ks):
    mdp, _ = get(domain)
    run = load("run", domain, "sfols", {"ds": "1e-10", "windy": "1e-5"}.get(domain, "2e-3"), 0.0)
    T = load("truth", domain)
    W, VS = T["W"], T["VS"]
    if len(W) > 400:
        idx = np.linspace(0, len(W) - 1, 201).astype(int)
        W, VS = W[idx], VS[idx]
    ks = [int(k) for k in ks.split(",")]
    full = library_from(mdp, run, max(ks))
    R, out = mdp.relevant, []
    print(f"{domain}: {len(W)} held-out weights, base = first k policies of SFOLS (OLS order); "
          f"% of weights where the policy is optimal at every relevant state", flush=True)
    for k in ks:
        stack, acts = full.stack[:k], full.acts[:k]
        mask = np.zeros((mdp.S, mdp.A), bool)
        for a in acts:
            mask[mdp.R, a] = True
        opt = {"gpi": 0, "meta": 0, "hyb": 0}
        worst = {"gpi": 0.0, "meta": 0.0, "hyb": 0.0}
        extra = []
        for i, w in enumerate(W):
            c = Cost(mdp.A)
            gact, Vh = gpi(stack, w)
            vals = {"gpi": evaluate(mdp, gact, w, Vh, c)}
            mh = meta_hyb(mdp, stack, acts, mask, w, c)
            vals["meta"], vals["hyb"] = mh["meta"][1], mh["hyb"][1]
            for key, V in vals.items():
                loss = (VS[i, R] - V[R]).max() / max(VS[i, R].max(), 1e-12)
                opt[key] += loss <= 1e-8
                worst[key] = max(worst[key], loss)
            _, _, _, st = policy_iteration(mdp, w, mh["hyb"][0], mh["hyb"][1], Cost(mdp.A))
            extra.append(st)
        row = dict(k=k, **{f"opt_{a}": 100 * b / len(W) for a, b in opt.items()},
                   **{f"worst_{a}": v for a, v in worst.items()}, extra_mean=float(np.mean(extra)),
                   extra_max=int(np.max(extra)))
        out.append(row)
        print(f"  k={k:3d}: optimal GPI {row['opt_gpi']:5.1f}%  meta {row['opt_meta']:5.1f}%  hyb {row['opt_hyb']:5.1f}%"
              f" | worst relative loss GPI {worst['gpi']:.1e} meta {worst['meta']:.1e} hyb {worst['hyb']:.1e}"
              f" | PI steps from hyb to optimal: mean {row['extra_mean']:.2f} max {row['extra_max']}", flush=True)
    save(out, "hybcurve", domain)


# ---------------------------------------------------------------- new FSA tasks: cost of the certified optimum
def value_iteration(mdp, w, cost, tol):
    """VI from 0 until the sup change is <= tol; returns the greedy policy and the last iterate (no SFs)."""
    V = np.zeros(mdp.S)
    while True:
        Q = lookahead(mdp, V, w, cost)
        new = Q.max(1)
        d, V = np.abs(new - V).max(), new
        if d <= tol:
            return tol_argmax(Q), V


def solve_state(mdp, w, how, stack, acts, mask, cost, certify=True):
    """Policy and exact values at w by one of: scratch / vi<tol> / gpi / elect / hyb, followed (certify=True) by
    policy iteration to zero Bellman gap, i.e. the certified optimum. certify=False returns the uncertified policy.
    elect = best base policy per state (pi_E), the start of the metapolicy, without its restricted phase."""
    if how == "scratch":
        act, V = np.zeros(mdp.S, int), np.zeros(mdp.S)                    # rewards >= 0: 0 is a sub-solution
        V = evaluate(mdp, act, w, V, cost)
    elif how.startswith("vi"):
        act, V = value_iteration(mdp, w, cost, float(how[2:]))
        V = evaluate(mdp, act, w, V, cost)
    elif how == "gpi":
        act, Vh = gpi(stack, w)
        V = evaluate(mdp, act, w, Vh, cost)
    elif how == "elect":
        Qj, Vj = base_tables(stack, acts, w)
        act = np.stack(acts)[Vj.argmax(0), mdp.R]
        V = evaluate(mdp, act, w, Vj.max(0), cost)                        # max_j V_j: sub-solution of pi_E
    elif how == "meta":
        act, V = meta_hyb(mdp, stack, acts, mask, w, cost)["meta"]
    elif how == "ladder":
        return solve_ladder(mdp, w, stack, acts, mask, cost)
    else:
        act, V = meta_hyb(mdp, stack, acts, mask, w, cost)["hyb"]
    if not certify:
        return act, V, 0
    act, V, _, steps = policy_iteration(mdp, w, act, V, cost)
    return act, V, steps


def stable(mdp, Q, act):
    best = tol_argmax(Q)
    return np.array_equal(np.where(Q[mdp.R, act] >= Q[mdp.R, best] - 1e-12, act, best), act)


def solve_ladder(mdp, w, stack, acts, mask, cost):
    """Algorithm 1, one automaton state: GPI (certified by its Bellman gap) -> metapolicy and pi_hyb (certified) ->
    policy iteration over all actions from pi_hyb. Returns (act, V, rung) with rung 0 / 1 / 2 = where it stopped."""
    act, Vh = gpi(stack, w)
    V = evaluate(mdp, act, w, Vh, cost)
    if stable(mdp, lookahead(mdp, V, w, cost), act):
        return act, V, 0
    act, V = meta_hyb(mdp, stack, acts, mask, w, cost)["hyb"]
    act, V, _, steps = policy_iteration(mdp, w, act, V, cost)
    return act, V, 1 if steps == 0 else 2


def plan(mdp, task, how, stack, acts, mask, certified=True, cost=None):
    """Reverse topological order; exit values passed upstream are exact values of the executed policies."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    V, out, steps = {}, {}, 0
    cost = Cost(mdp.A) if cost is None else cost
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else V[v][ex[e]]
        act, val, st = solve_state(mdp, w, how, stack, acts, mask, cost, certified)
        out[u], V[u] = act, val
        steps += st if how != "ladder" else 0
        if how == "ladder":
            cost.rungs = getattr(cost, "rungs", []) + [st]
    return out, cost.sweeps, steps


def cmd_tasks(domain, k):
    mdp, _ = get(domain)
    k = int(k)
    run = load("run", domain, "sfols", {"windy": "1e-5"}.get(domain, "2e-3"), 0.0)
    lib = library_from(mdp, run, k)
    stack, acts = lib.stack, lib.acts
    mask = np.zeros((mdp.S, mdp.A), bool)
    for a in acts:
        mask[mdp.R, a] = True
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    opt = np.array(load("taskopt", domain))
    cache, res = {}, {}
    zs = np.array([opt[i] - library_value(mdp, t, lib, cache) for i, t in enumerate(tasks)])
    res["zero-shot GPI (estimate planner)"] = dict(regret=zs, sweeps=np.zeros(len(tasks)), steps=np.zeros(len(tasks)))
    for how, cert in (("gpi", False), ("hyb", False), ("scratch", True), ("gpi", True), ("hyb", True)):
        reg, sw, st = [], [], []
        for i, t in enumerate(tasks):
            a, c, s = plan(mdp, t, how, stack, acts, mask, certified=cert)
            reg.append(opt[i] - executed_value(mdp, t, a, cache)); sw.append(c); st.append(s)
        name = {("gpi", False): "GPI, evaluated (no PI)", ("hyb", False): "hybrid, evaluated (no PI)",
                ("scratch", True): "PI from scratch", ("gpi", True): "GPI + PI to zero gap",
                ("hyb", True): "hybrid + PI to zero gap"}[(how, cert)]
        res[name] = dict(regret=np.array(reg), sweeps=np.array(sw), steps=np.array(st))
    print(f"{domain}: {len(tasks)} tasks, base = first {k} SFOLS policies")
    for name, r in res.items():
        print(f"  {name:34s} max regret {r['regret'].max():.1e} mean {r['regret'].mean():.1e} | sweeps/task mean "
              f"{r['sweeps'].mean():6.1f} | improvement steps/task mean {r['steps'].mean():.2f}", flush=True)
    save(res, "hybtasks", domain, k)


# ---------------------------------------------------------------- warm starts for certified PI over all actions
def cmd_warm(domain, k, hows="scratch,vi1e-4,vi1e-6,vi1e-8,gpi,elect,hyb"):
    """Every method ends with full-action policy iteration to zero Bellman gap (certified optimum); they differ
    only in the start. Reports per task: sweeps (greedy + evals/A), full lookaheads, restricted lookaheads."""
    mdp, _ = get(domain)
    k = int(k)
    run = load("run", domain, "sfols", {"windy": "1e-5"}.get(domain, "2e-3"), 0.0)
    lib = library_from(mdp, run, k)
    stack, acts = lib.stack, lib.acts
    mask = np.zeros((mdp.S, mdp.A), bool)
    for a in acts:
        mask[mdp.R, a] = True
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    opt = np.array(load("taskopt", domain))
    cache, res = {}, {}
    print(f"{domain}: {len(tasks)} tasks, base = first {k} SFOLS policies; all rows end at zero Bellman gap", flush=True)
    for how in hows.split(","):
        rows = []
        for i, t in enumerate(tasks):
            c = Cost(mdp.A)
            a, _, st = plan(mdp, t, how, stack, acts, mask, certified=True, cost=c)
            rows.append((opt[i] - executed_value(mdp, t, a, cache), c.sweeps, c.greedy - c.restricted,
                         c.restricted, c.evals / mdp.A, st))
        r = np.array(rows)
        res[how] = dict(regret=r[:, 0], sweeps=r[:, 1], full=r[:, 2], restricted=r[:, 3], evals=r[:, 4], steps=r[:, 5])
        print(f"  {how:8s} max regret {r[:, 0].max():.1e} | sweeps/task {r[:, 1].mean():6.1f} = full lookaheads "
              f"{r[:, 2].mean():5.1f} + restricted {r[:, 3].mean():5.1f} + eval {r[:, 4].mean():6.1f} | PI steps after "
              f"start {r[:, 5].mean():.2f}", flush=True)
    save(res, "hybwarm", domain, k)


# ---------------------------------------------------------------- Algorithm 1 on the FSA tasks, per task type
VI_TOL = {"office1f2": "vi1e-3", "office3f2": "vi1e-3", "windy": "vi1e-2"}     # the cheapest VI tolerance found


def task_type(domain, i, task):
    if domain == "windy":
        return "A or B, then corridor" if len(task["u0"]) == 2 else "A (or B), then corridor"
    return ["sequential", "disjunction", "composite"][i] if i < 3 else "random"


def cmd_fsa(domain, k=6):
    """Per task: regret of the uncertified GPI / metapolicy / pi_hyb plans, cost of the certified optimum by VI from
    scratch, GPI + PI and Algorithm 1 (ladder), and the rung where each automaton state of the ladder stopped."""
    mdp, _ = get(domain)
    k = int(k)
    run = load("run", domain, "sfols", {"windy": "1e-5"}.get(domain, "2e-3"), 0.0)
    lib = library_from(mdp, run, k)
    stack, acts = lib.stack, lib.acts
    mask = np.zeros((mdp.S, mdp.A), bool)
    for a in acts:
        mask[mdp.R, a] = True
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    opt = np.array(load("taskopt", domain))
    cache, rows = {}, []
    for i, t in enumerate(tasks):
        r = dict(i=i, type=task_type(domain, i, t), n_states=len(t), opt=float(opt[i]))
        r["zs"] = float(opt[i] - library_value(mdp, t, lib, cache))
        for how in ("gpi", "meta", "hyb"):
            a, _, _ = plan(mdp, t, how, stack, acts, mask, certified=False)
            r[f"reg_{how}"] = float(opt[i] - executed_value(mdp, t, a, cache))
        for how in (VI_TOL[domain], "gpi", "ladder", "scratch"):
            c = Cost(mdp.A)
            a, sw, st = plan(mdp, t, how, stack, acts, mask, certified=True, cost=c)
            key = how if not how.startswith("vi") else "vi"
            r[f"sw_{key}"], r[f"full_{key}"] = sw, c.greedy - c.restricted
            r[f"cert_reg_{key}"] = float(opt[i] - executed_value(mdp, t, a, cache))
            if how == "ladder":
                r["rungs"] = c.rungs
        rows.append(r)
    save(rows, "hybfsa", domain, k)
    report_fsa(domain, rows)


def report_fsa(domain, rows):
    types = list(dict.fromkeys(r["type"] for r in rows))
    print(f"{domain}: {len(rows)} tasks, base = {6} extremes. Regret (mean / max) and sweeps per task")
    for ty in types:
        rs = [r for r in rows if r["type"] == ty]
        f = lambda key: f"{np.mean([r[key] for r in rs]):.1e}/{np.max([r[key] for r in rs]):.1e}"
        rung = np.concatenate([r["rungs"] for r in rs])
        print(f"  {ty:24s} n={len(rs):3d} | zero-shot {f('zs')} GPI {f('reg_gpi')} meta {f('reg_meta')} hyb {f('reg_hyb')}"
              f" | sweeps VI {np.mean([r['sw_vi'] for r in rs]):6.1f} GPI+PI {np.mean([r['sw_gpi'] for r in rs]):6.1f}"
              f" ladder {np.mean([r['sw_ladder'] for r in rs]):6.1f} | rungs GPI/hyb/PI "
              f"{(rung == 0).mean():.2f}/{(rung == 1).mean():.2f}/{(rung == 2).mean():.2f}"
              f" | max cert regret {max(r['cert_reg_ladder'] for r in rs):.0e}", flush=True)


if __name__ == "__main__":
    {"curve": cmd_curve, "tasks": cmd_tasks, "warm": cmd_warm, "fsa": cmd_fsa}[sys.argv[1]](*sys.argv[2:])
