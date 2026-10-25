"""Zero-shot SF-FSA-VI over a Certify-or-Grow eps-CCS, with and without the evaluation trick.

usage: PYTHONPATH=SFOLS python -m certlib.zeroshot <domain> <run_eps> <eps,...> <n,...> <seeds>

For each eps, the library is the Certify-or-Grow run's state when Gamma <= eps is first certified. On every task:
  estimate   Kuric et al.'s planner: exit values = GPI estimate max_{j,a} psi_j(e, a).w(v)
  evaluated  the same GPI policies, but the exit values passed upstream are the GPI policy's ACTUAL values,
             learned by TD on a replay buffer of n transitions per (s, a). TD on a fixed buffer converges to the
             policy's value on the buffer's empirical model, which is what is computed here ('exact' = true P).
Reported: true regret of the executed policies, and the planner's error in predicting the task's value.
"""
import sys

import numpy as np

from gpi_basis.fsa import _order, symbols

from .domains import sampled
from .experiments import get, library_from, load, save, tasks_for
from .mdp import tol_argmax
from .pertask import executed_value
from .report import stop_state


def plan(mdp, task, stack, model=None, tol=1e-8):
    """Weights, GPI actions per automaton state, and the planner's predicted start value.
    model=None: estimate planner; otherwise exit values from evaluating GPI's policy on `model`."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    V, acts = {}, {}
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else V[v][ex[e]]
        Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
        acts[u], V[u] = tol_argmax(Qh), Qh.max(1)
        if model is not None:                             # TD fixed point on the buffer, warm-started at V^
            r = (model.PHI @ w)[mdp.R, acts[u]]
            P = model.P[mdp.R, acts[u]]
            thr = tol * (1 - mdp.gamma) / mdp.gamma
            L = V[u]
            while True:
                new = r + mdp.gamma * P @ L
                d, L = np.abs(new - L).max(), new
                if d <= thr:
                    break
            V[u] = L
    return acts, float(V["u0"][int(mdp.relevant[0])])


def main(domain, run_eps, eps_list, n_list, seeds):
    mdp, _ = get(domain)
    run = load("run", domain, "cog", run_eps, 0.0)
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    opt = np.array(load("taskopt", domain))
    full = library_from(mdp, run)
    libs = [("6 extremes", 6, 6)]
    for e in [float(x) for x in eps_list.split(",")]:
        st = stop_state(run, e)
        libs.append((f"eps {e:g}", st["library"], st["learned"]))
    ns = [int(x) for x in n_list.split(",")]
    models = {n: [sampled(mdp, n, np.random.default_rng(100 + s)) for s in range(int(seeds))] for n in ns}
    cache, out = {}, []
    print(f"{domain}: {len(tasks)} tasks; regret = optimal - true value of the executed policies; "
          f"pred = |planner's predicted start value - true value|", flush=True)
    for name, size, learned in libs:
        stack = full.stack[:size]
        row = dict(library=name, size=size, learned=learned)
        for key, mod in [("estimate", [None]), ("exact", [mdp])] + [(f"n={n}", models[n]) for n in ns]:
            reg, pred = [], []
            for m in mod:
                for k, t in enumerate(tasks):
                    acts, p = plan(mdp, t, stack, m)
                    val = executed_value(mdp, t, acts, cache)
                    reg.append(opt[k] - val); pred.append(abs(p - val))
            row[key] = dict(max_regret=float(np.max(reg)), mean_regret=float(np.mean(reg)),
                            max_pred=float(np.max(pred)), mean_pred=float(np.mean(pred)))
            print(f"  {name:12s} ({size:3d} SFs, {learned:2d} trained) {key:9s} regret max {np.max(reg):.1e} "
                  f"mean {np.mean(reg):.1e} | pred max {np.max(pred):.1e} mean {np.mean(pred):.1e}", flush=True)
        out.append(row)
    save(out, "zeroshot", domain, n_list.replace(",", "-"))


def plan_td(mdp, task, stack, k, alpha, m, rng):
    """GPI planner whose exit values are refined by k synchronous sampled TD sweeps of the GPI policy, warm-started
    at the GPI estimate: each sweep draws m fresh outcomes of (s, GPI(s)) for every state s and moves V(s) a step
    alpha toward their mean target. k = 0 is Kuric's zero-shot planner. Returns (acts, predicted start value)."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    S, D, g = mdp.S, mdp.D, mdp.gamma
    V, acts = {}, {}
    for u in _order(task):
        w = np.zeros(D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else V[v][ex[e]]
        Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
        a = tol_argmax(Qh)
        L = Qh.max(1)
        if k:
            P, PHI = mdp.P[mdp.R, a], mdp.PHI[mdp.R, a]
            cdf = np.cumsum(np.concatenate([P, PHI, np.maximum(1 - P.sum(1) - PHI.sum(1), 0)[:, None]], 1), 1)
            pay = np.concatenate([np.zeros(S), w, [0.0]])           # reward of each outcome
            boot = np.concatenate([np.ones(S), np.zeros(D + 1)])    # does the outcome continue?
            for _ in range(k):
                idx = (rng.random((S, m, 1)) > cdf[:, None, :]).sum(-1)          # (S, m) sampled outcomes
                idx = np.minimum(idx, S + D)
                Vext = np.concatenate([L, np.zeros(D + 1)])
                target = (pay[idx] + g * boot[idx] * Vext[idx]).mean(1)
                L = L + alpha * (target - L)
        V[u], acts[u] = L, a
    return acts, float(V["u0"][int(mdp.relevant[0])])


def refine(domain, run_eps, lib_eps, seeds):
    """Sampled-TD refinement grid (k sweeps, step alpha, m samples per state per sweep) vs zero-shot and vs the
    same k sweeps without noise (expected backups, alpha = 1)."""
    mdp, _ = get(domain)
    run = load("run", domain, "cog", run_eps, 0.0)
    tasks = tasks_for(domain, mdp, np.random.default_rng(1))
    opt = np.array(load("taskopt", domain))
    full = library_from(mdp, run)
    libs = [("6 extremes", 6)] + [(f"eps {float(e):g}", stop_state(run, float(e))["library"])
                                  for e in lib_eps.split(",")]
    cache, out = {}, []

    def score(fn, nseeds):
        reg, pred = [], []
        for s in range(nseeds):
            rng = np.random.default_rng(1000 + s)
            for j, t in enumerate(tasks):
                acts, p = fn(t, rng)
                val = executed_value(mdp, t, acts, cache)
                reg.append(opt[j] - val); pred.append(abs(p - val))
        return float(np.max(reg)), float(np.mean(reg)), float(np.max(pred)), float(np.mean(pred))

    for name, size in libs:
        stack = full.stack[:size]
        rows = [("zero-shot", 0, None, None, score(lambda t, r: plan_td(mdp, t, stack, 0, 0, 0, r), 1))]
        for k in (1, 5, 20):
            rows.append((f"exact k={k}", k, 1.0, "inf",
                         score(lambda t, r, k=k: plan_exact_k(mdp, t, stack, k), 1)))
            for alpha in (0.1, 0.5, 1.0):
                for m in (1, 10):
                    rows.append((f"TD k={k} a={alpha} m={m}", k, alpha, m,
                                 score(lambda t, r, k=k, al=alpha, m=m: plan_td(mdp, t, stack, k, al, m, r), int(seeds))))
        print(f"\n{domain}, {name} ({size} SFs): max regret / mean regret / max |pred - true| / mean |pred - true|")
        for lab, k, al, m, (a, b, c, d) in rows:
            print(f"  {lab:24s} {a:.1e}  {b:.1e}  {c:.1e}  {d:.1e}", flush=True)
        out.append(dict(library=name, size=size, rows=rows))
    save(out, "refine", domain, lib_eps.replace(",", "-"))


def plan_exact_k(mdp, task, stack, k):
    """k expected backups of the GPI policy from the GPI estimate (no noise): monotone, each a lower bound."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    V, acts = {}, {}
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else V[v][ex[e]]
        Qh = np.tensordot(stack, w, axes=([3], [0])).max(0)
        a = tol_argmax(Qh)
        L = Qh.max(1)
        r, P = (mdp.PHI @ w)[mdp.R, a], mdp.P[mdp.R, a]
        for _ in range(k):
            L = r + mdp.gamma * P @ L
        V[u], acts[u] = L, a
    return acts, float(V["u0"][int(mdp.relevant[0])])


if __name__ == "__main__":
    if sys.argv[1] == "refine":
        refine(*sys.argv[2:6])
    else:
        main(*sys.argv[1:6])
