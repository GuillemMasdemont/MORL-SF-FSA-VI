"""Zero-shot SF-FSA-VI with an SF library (Kuric et al.'s planner), and its true value (reporting only).

A task is {u: {symbol: next_state or 'T'}} (acyclic). Planning: in reverse topological order,
w_e(u) = 1 if exit e's symbol ends the task, the planner's value of exit e under w(v) if it leads to v,
0 otherwise. The planner's value is the GPI estimate max_{j,a} psi_j(e, a).w (as in the paper); the policy
executed in u is GPI over the library at w(u).
"""
import numpy as np

from gpi_basis.fsa import PAPER_TASKS, _order, plan, random_tasks, symbols, windy_tasks  # noqa: F401

from .mdp import tol_argmax


def _true_value(mdp, task, W, policy_fn, cache):
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)

    def sf(act):
        k = act.tobytes()
        if k not in cache:
            cache[k] = mdp.evaluate(act, tol=1e-12)[mdp.R, act]
        return cache[k]
    psi = {u: sf(policy_fn(W[u])) for u in task}
    Wt = {}
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else psi[v][ex[e]] @ Wt[v]
        Wt[u] = w
    start = mdp.relevant[0]
    return float(psi["u0"][start] @ Wt["u0"])


def optimal_value(mdp, task, cache):
    sols = {}

    def solve(w):
        k = tuple(np.round(w, 12))
        if k not in sols:
            sols[k] = mdp.solve(w)
        return sols[k]
    W = plan(mdp, task, lambda w: (solve(w)[1][mdp.R, solve(w)[0]] @ w) if np.any(w) else np.zeros(mdp.S))
    return _true_value(mdp, task, W, lambda w: solve(w)[0] if np.any(w) else np.zeros(mdp.S, int), cache)


def library_value(mdp, task, lib, cache, upto=None):
    stack = lib.stack if upto is None else lib.stack[:upto]
    qhat = lambda w: np.tensordot(stack, w, axes=([3], [0])).max(0)
    W = plan(mdp, task, lambda w: qhat(w).max(1))
    return _true_value(mdp, task, W, lambda w: tol_argmax(qhat(w)), cache)
