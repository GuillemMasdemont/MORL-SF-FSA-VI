"""Model-free lazy SF-FSA-VI on slippery Office (Kuric semantics, certlib/lazyfsa.py): every policy of the base is
LEARNED by SF Q-learning; the planner is Kuric et al.'s iteration over the learned SF tables.

    base    the 6 extremes, SFQL from an optimistic start (QBASE steps each), visit counts kept
    repeat  W <- Kuric's iteration over the learned tables
            for every automaton state u whose GPI policy at w(u) is not certified:
                cog: learn the metapolicy (SFQL acting and bootstrapping on B(s), the base actions), started at the GPI
                     table with inherited counts; if it is not certified, keep learning with all actions
                pi : learn with all actions from the GPI table with inherited counts (no metapolicy)
                add the learned table to the base
    until every automaton state is certified
Assumption (stated): certification is an oracle check that the GPI policy is within DELTA of optimal at the relevant
states (start and exits); the learners stop when their greedy policy has been within DELTA of their own optimum
(restricted for the metapolicy) for STABLE checks. The model is used for nothing else.

usage (okb env, from SFOLS/): python q_lazy_fsa.py [n_random_tasks] [workers]
"""
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np

os.environ.setdefault("QSLIP", "0.05")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q_fsa_stochastic as QF                                    # noqa: E402  (mdp, step, value, vstar, sf_learn)
from certlib.experiments import tasks_for                       # noqa: E402
from certlib.lazyfsa import Task, gpi_policies, kuric_vi, product_optimum, product_value   # noqa: E402
from certlib.library import Library                              # noqa: E402

mdp = QF.mdp
S, A, D, G = mdp.S, mdp.A, mdp.D, mdp.gamma
ROWS, REL = np.arange(S), mdp.relevant
DELTA, STABLE, BUDGET = 1e-3, int(os.environ.get("QSTABLE", 100)), int(float(os.environ.get("QBUDGET", 3e6)))
OUT = os.path.join(HERE, "results", f"lazy_fsa_mf_{'slip%s' % QF.SLIP if QF.DOM == 'office' else QF.DOM}.pkl")


def learn(w, Psi0, n0, allowed, vref, seed):
    """Vector SFQL at w on `allowed`, step 1/(n+1)^KAPPA with n starting at n0. Stops when the greedy policy has been
    within DELTA of vref at the relevant states for STABLE checks (oracle) or after BUDGET steps."""
    rng = random.Random(seed)
    Psi, n = Psi0.copy(), n0.astype(float).copy()
    Qw = Psi @ w
    masked = np.where(allowed, 0.0, -np.inf)
    allowed_l = [np.flatnonzero(allowed[s]).tolist() for s in range(S)]
    E = np.eye(D)

    def ok():
        return (vref - QF.value((Qw + masked).argmax(1), w)[REL]).max() <= DELTA

    good, steps, s, t, last_bad = 0, 0, None, QF.HORIZON, (0 if not ok() else -1)
    while steps < BUDGET:
        if s is None or t >= QF.HORIZON:
            s, t = rng.choice(QF.ALL), 0
            a = rng.choice(allowed_l[s])
        else:
            a = rng.choice(allowed_l[s]) if rng.random() < QF.EPS else int((Qw[s] + masked[s]).argmax())
        ns, o = QF.step(s, a, rng)
        target = E[o] if o >= 0 else G * Psi[ns, int((Qw[ns] + masked[ns]).argmax())]
        n[s, a] += 1
        Psi[s, a] += (target - Psi[s, a]) / (n[s, a] + 1) ** QF.KAPPA
        Qw[s, a] = Psi[s, a] @ w
        steps += 1
        t += 1
        s = None if o >= 0 else ns
        if steps % QF.CHECK == 0:
            if ok():
                good += 1
                if good >= STABLE:
                    break
            else:
                good, last_bad = 0, steps
    act = (Qw + masked).argmax(1)
    return Psi, n, act, steps, max(last_bad, 0)


def certified(L, w):
    """Oracle certificate (assumption): GPI over L at w is within DELTA of optimal at the relevant states."""
    act = np.tensordot(L.stack, w, axes=([3], [0])).max(0).argmax(1)
    return (QF.vstar(w)[1][REL] - QF.value(act, w)[REL]).max() <= DELTA


def grow(L, counts, w, how, seed):
    Qj = np.tensordot(L.stack, w, axes=([3], [0]))
    j = Qj.argmax(0)
    Psi0 = np.take_along_axis(L.stack, j[None, :, :, None], 0)[0]
    n0 = np.take_along_axis(np.stack(counts), j[None], 0)[0]
    full = np.ones((S, A), bool)
    vfull = QF.vstar(w)[1][REL]
    steps = conv = 0
    kind = "full"
    if how == "cog":
        allowed = np.zeros((S, A), bool)
        for a in L.acts:
            allowed[ROWS, a] = True
        vres = QF.vstar(w, allowed)[1][REL]
        Psi0, n0, act, st, cv = learn(w, Psi0, n0, allowed, vres, seed)
        steps, conv = st, cv
        if (vfull - QF.value(act, w)[REL]).max() <= DELTA:
            return Psi0, n0, act, steps, conv, "metapolicy"
        seed += 7919
    Psi, n, act, st, cv = learn(w, Psi0, n0, full, vfull, seed)
    return Psi, n, act, steps + st, steps + cv, kind


def lazy_mf(T, L, counts, how, seed, max_rounds=8):
    W, log = None, dict(rounds=0, steps=0, conv=0, kinds=[])
    for _ in range(max_rounds):
        W, _ = kuric_vi(mdp, T, L, W)
        log["rounds"] += 1
        grew = 0
        for u in T.U:
            if certified(L, W[u]):
                continue
            Psi, n, act, st, cv, kind = grow(L, counts, W[u], how, seed)
            seed += 1
            L.add(act, Psi, kind, W[u])
            counts.append(n)
            log["steps"] += st
            log["conv"] += cv
            log["kinds"].append(kind)
            grew += 1
        if not grew:
            break
    log["certified"] = grew == 0
    return W, log


# ---------------------------------------------------------------- jobs
TASKS, BASE, CCS = [], None, None


def _base_job(args):
    i, seed = args
    w = np.eye(D)[i]
    Psi, n = QF.sf_learn(w, seed)
    return i, Psi, n


def _evaluate(T, L, W):
    s0 = int(mdp.relevant[0])
    return float(product_value(mdp, T, gpi_policies(L, W))["u0"][s0])


def _stream_job(args):
    how, seed = args
    L = Library(mdp)
    L.stack, L.acts, L.kinds, L.weights = BASE["stack"].copy(), list(BASE["acts"]), ["base"] * D, list(np.eye(D))
    counts = [c.copy() for c in BASE["counts"]]
    rows = []
    for i, task in enumerate(TASKS):
        T = Task(mdp, task)
        n0 = len(L)
        t0 = time.perf_counter()
        W, log = lazy_mf(T, L, counts, how, seed + 1000 * i)
        rows.append(dict(log, i=i, value=_evaluate(T, L, W), added=len(L) - n0, size=len(L),
                         seconds=time.perf_counter() - t0))
    return how, rows


def main(n_random=10, workers=8):
    global TASKS, BASE
    n_random, workers = int(n_random), int(workers)
    TASKS = (tasks_for("office", mdp, np.random.default_rng(1))[:3 + n_random] if QF.DOM == "office"
             else tasks_for("windy", mdp, np.random.default_rng(1))[:n_random])
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(min(workers, D)) as pool:
        out = pool.map(_base_job, [(i, 4000 + i) for i in range(D)])
    out.sort()
    stack = np.stack([o[1] for o in out])
    acts = [np.argmax(stack[i] @ np.eye(D)[i], 1) for i in range(D)]
    BASE = dict(stack=stack, counts=[o[2] for o in out], acts=acts)
    print(f"{QF.DOM} (slip {QF.SLIP}): base of {D} extremes learned model-free ({D} x {QF.BASE_STEPS:,} steps, "
          f"{time.perf_counter() - t0:.0f}s); {len(TASKS)} tasks", flush=True)
    s0 = int(mdp.relevant[0])
    opt, zs = [], []
    Lb = Library(mdp)
    Lb.stack, Lb.acts = stack, acts
    for task in TASKS:
        T = Task(mdp, task)
        _, Vopt, _ = product_optimum(mdp, T)
        opt.append(float(Vopt["u0"][s0]))
        W, _ = kuric_vi(mdp, T, Lb)
        zs.append(_evaluate(T, Lb, W))
    with mp.get_context("fork").Pool(2) as pool:
        res = dict(pool.map(_stream_job, [("cog", 11), ("pi", 11)]))
    with open(OUT, "wb") as fh:
        pickle.dump(dict(opt=opt, zero_shot=zs, res=res, base_steps=D * QF.BASE_STEPS), fh)
    report(opt, zs, res)


def report(opt, zs, res):
    opt, zs = np.array(opt), np.array(zs)
    print(f"\nKuric + {D} learned extremes (zero-shot): max relative regret {((opt - zs) / opt).max():.1e}, "
          f"mean {((opt - zs) / opt).mean():.1e}")
    for how, rows in res.items():
        reg = np.array([(o - r["value"]) / o for o, r in zip(opt, rows)])
        kinds = [k for r in rows for k in r["kinds"]]
        print(f"  lazy-{how:3s} (base carried over the {len(rows)} tasks): max relative regret {reg.max():.1e}, mean "
              f"{reg.mean():.1e}; certified {sum(r['certified'] for r in rows)}/{len(rows)}; per task: rounds "
              f"{np.mean([r['rounds'] for r in rows]):.2f}, learned policies {np.mean([r['added'] for r in rows]):.2f}, "
              f"env steps {np.mean([r['steps'] for r in rows]) / 1e6:.2f}M (to converge "
              f"{np.mean([r['conv'] for r in rows]) / 1e6:.2f}M); learned in total {len(kinds)} "
              f"({kinds.count('metapolicy')} as metapolicies); final base {rows[-1]['size']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
