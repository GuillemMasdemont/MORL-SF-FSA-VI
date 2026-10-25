"""Demand-driven, certified SF-FSA-VI: grow the GPI pool only at the weights a task actually demands.

For a task, plan with GPI over the current pool (exit values = TRUE values of the GPI policies, one
evaluation each), then check every automaton state's weight w(u) with the Bellman certificate of the GPI
policy at w(u). If GPI is certified optimal at every w(u), the planned weights satisfy the true optimality
equations, so the composed policy is optimal (the planning operator is a contraction with a unique fixed
point). Otherwise, Certify-or-Grow at the uncertified weights (metapolicy first, full solve if needed) adds
an exactly optimal policy there, and we replan. The pool and the metapolicy's base carry over between tasks.
"""
import numpy as np

from .fsa import plan, true_start_value


def gpi_policy(G, w):
    return G.cache.get(G.pool.actions(w))


def solve_task_certified(mdp, task, G, tol=1e-9, max_rounds=50):
    """Returns (true start value, stats). G: a GrowthND whose pool already holds the corner policies."""
    stats = dict(rounds=0, grows=0)
    for rnd in range(max_rounds):
        stats["rounds"] = rnd + 1
        W = plan(mdp, task, lambda w: gpi_policy(G, w).psi_s @ w if np.any(w) else np.zeros(mdp.S))
        bad = []
        for u, w in W.items():
            if not np.any(w):
                continue
            if gpi_policy(G, w).gap(w).max() > tol:
                bad.append(u)
        if not bad:
            break
        for u in bad:
            G.grow(W[u], "cog")
            stats["grows"] += 1
    value = true_start_value(mdp, task, W, lambda w: G.pool.actions(w))
    return value, stats
