"""Model-free Lazy SF-FSA-VI against the previous algorithms, on Kuric et al.'s planning problem (Kuric semantics,
certlib/lazyfsa.py) with LEARNED successor features. Every method learns from samples with the same learners (mf.py)
and pays for every environment step; the model is only used to measure (oracle), as in q_fsa_stochastic.py.

  base      SFQL at the d extreme weights from a tight optimistic start, visit counts kept (shared by the SF methods)
  lazy-cog  Lazy SF-FSA-VI (paper Alg. 4): Kuric's iteration over the stored tables; for every automaton state whose GPI
            policy at w(u) is not certified, learn the metapolicy (SFQL acting and bootstrapping on B(s)) started at
            the GPI table with inherited counts, and keep learning on all actions if it is not certified; add it
  lazy-ql   the same, but every new policy is learned by SFQL over all actions (no metapolicy)
  sfols     Kuric et al.'s SF-FSA-VI: SFOLS from the same extremes (OLS corner weights, every query learned at the GPI
            table with inherited counts) and, after every query, Kuric's plan over all the learned tables
  flat      Q-learning on the product MDP U x S of each task from scratch (Kuric et al.'s Flat-Q)
Lazy runs on every task from the extremes (fresh, tasks in parallel) and over the task sequence keeping the added
policies (stream), at most QROUNDS rounds per task.

Assumption (stated): the certificate is an oracle check that, at w(u), the GPI policy over the stored tables and the GPI
estimate (which Kuric's iteration reads at the exits) are within delta of V*_{w(u)} at the relevant states (start and
exits): the oracle version of Algorithm 4's certificate Q^GPI = Q*. Every learner stops when its greedy policy and its
estimated values have been within delta of its own optimum for QSTABLE checks in a row (oracle), or after its budget.
delta = QDELTA * max V*_w at the relevant states for the policies, QDELTAEST * max V*_w for the estimates. SF tables start at zero, so estimates are biased low and GPI stays
(approximately) a lower bound, as with exact SFs. Costs are environment steps; "to converge" counts each learn until
its last failed check.

usage (from SFOLS/): python q_lazy_fsa.py [workers]
env: QDOM office|windy|ds, QSLIP (office p), QSEED, QNTASK, QDELTA, QBASE, QBUDGET, QFLAT (budgets), QSTABLE, QCHECK,
     QROUNDS, QOLSEPS, QOLSMAX, QPARTS (subset of sfols,lazy,flat), QINIT (zero|opt), QESTIMATE (1|0), QDELTAEST,
     QFLATQ0, QINHERIT (1|0)
"""
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mf                                                                    # noqa: E402
from certlib import domains                                                  # noqa: E402
from certlib.experiments import tasks_for                                    # noqa: E402
from certlib.lazyfsa import Task, kuric_vi, product_optimum                  # noqa: E402
from certlib.library import Library                                          # noqa: E402
from certlib.mdp import tol_argmax                                           # noqa: E402

DOM = os.environ.get("QDOM", "office")
SLIP = float(os.environ.get("QSLIP", 0.1))
SEED = int(os.environ.get("QSEED", 0))
NTASK = int(os.environ.get("QNTASK", 33))
DELTA = float(os.environ.get("QDELTA", 1e-3))                   # tolerance of the policies
DELTA_EST = float(os.environ.get("QDELTAEST", DELTA))           # tolerance of the estimated values
BASE = int(float(os.environ.get("QBASE", 2e7)))
BUDGET = int(float(os.environ.get("QBUDGET", 2e7)))
FLAT = int(float(os.environ.get("QFLAT", 1e8)))
STABLE = int(os.environ.get("QSTABLE", 100))
CHECK = int(os.environ.get("QCHECK", 1000))
ROUNDS = int(os.environ.get("QROUNDS", 5))
OLS_EPS = float(os.environ.get("QOLSEPS", 1e-3))
OLS_MAX = int(os.environ.get("QOLSMAX", 150))
PARTS = os.environ.get("QPARTS", "sfols,lazy,flat").split(",")
INIT = os.environ.get("QINIT", "zero")                # extremes: zero (pessimistic: GPI stays a lower bound) | opt
ESTIMATE = os.environ.get("QESTIMATE", "1") == "1"   # checks also require the estimated values (0: the policy only)
FLAT_Q0 = float(os.environ.get("QFLATQ0", 0.0))      # initial Q of flat Q-learning
INHERIT = os.environ.get("QINHERIT", "1") == "1"     # warm starts keep the visit counts of the entries they copy
TAG = f"{DOM}{SLIP if DOM == 'office' else ''}_s{SEED}"
OUT = os.path.join(os.environ.get("SF_OUT") or os.path.join(HERE, "results"), f"lazy_mf_{TAG}.pkl")


def domain():
    if DOM == "office":
        mdp = domains.office(p=SLIP)
        return mdp, tasks_for("office", mdp, np.random.default_rng(1))[:NTASK]
    if DOM == "windy":
        mdp = domains.windy_two_stage()
        return mdp, tasks_for("windy", mdp, np.random.default_rng(1))[:NTASK]
    if DOM == "ds":                                          # Kuric et al.'s double slit: "reach blue or red"
        mdp = domains.double_slit(3, 0.9)
        mdp.objects = ["O1", "O2"]
        mdp.exit_index = {o: [i for i, c in enumerate(mdp.coords) if c == pos][0]
                          for o, pos in zip(mdp.objects, _ds_exits())}
        mdp.symbols = {o: (o,) for o in mdp.objects}
        return mdp, [{"u0": {"O1": "T", "O2": "T"}}]
    raise ValueError(DOM)


def _ds_exits():
    from envs.grid_envs import DoubleSlit
    M = DoubleSlit.MAP
    return [tuple(int(x) for x in np.argwhere(M == o)[0]) for o in ("O1", "O2")]


MDP, TASK_DEFS = domain()
SMP, ORC = mf.Sampler(MDP), None
S, A, D, G = MDP.S, MDP.A, MDP.D, MDP.gamma
REL, S0 = MDP.relevant, int(MDP.relevant[0])
FULL = np.ones((S, A), bool)
TASKS = [Task(MDP, t) for t in TASK_DEFS]
for T in TASKS:
    T.tau_a = np.array([[-1 if v == "T" else T.U.index(v) for v in T.tau[u]] for u in T.U], np.int64)
    T.exits_a = T.exits.astype(np.int64)
    T.u0 = T.U.index("u0")
OPT, BASE_LIB = [], None


def oracle():
    global ORC
    if ORC is None:
        ORC = mf.Oracle(MDP, SMP)
    return ORC


def tol_of(w, allowed=None, delta=None):
    v = oracle().vstar(w, allowed)[1][REL]
    return v, (DELTA if delta is None else delta) * max(float(v.max()), 1e-12)


def ok_fn(w, allowed=None, estimate=True):
    """Oracle check at the relevant states: the greedy policy is within delta of the optimum (restricted to `allowed`)
    and, if `estimate`, so is its estimated value, which Kuric's iteration reads at the exits."""
    v, tol = tol_of(w, allowed)
    tol_est = DELTA_EST * max(float(v.max()), 1e-12)
    last = {"act": None, "val": None}

    def ok(act, vest):
        if last["act"] is None or not np.array_equal(act, last["act"]):
            last["act"], last["val"] = act.copy(), oracle().value(act, w)[REL]
        return (v - last["val"]).max() <= tol and (not (estimate and ESTIMATE) or np.abs(vest[REL] - v).max() <= tol_est)
    return ok


# ---------------------------------------------------------------- planning and evaluation (measurement)
def plan_value(ti, L, W, V0=None):
    """True value at (u0, s0) of executing GPI over L at W(u) in every automaton state."""
    T = TASKS[ti]
    acts = np.stack([tol_argmax(np.tensordot(L.stack, W[u], axes=([3], [0])).max(0)) for u in T.U]).astype(np.int64)
    V = mf.eval_product(acts, T.tau_a, T.exits_a, SMP.prob, SMP.nxt, SMP.obj, G,
                        np.zeros((len(T.U), S)) if V0 is None else V0, mf.EVAL_TOL)
    return float(V[T.u0, S0]), V


def rel_regret(ti, value):
    return max(OPT[ti] - value, 0.0) / OPT[ti]


def certified(L, w):
    """Oracle certificate (assumption): the GPI policy over L at w and the GPI estimate are within delta of V*_w at the
    relevant states (with exact SFs: the GPI estimate equals Q*_w, the certificate of Algorithm 4)."""
    Q = np.tensordot(L.stack, w, axes=([3], [0])).max(0)
    return ok_fn(w)(tol_argmax(Q), Q.max(1))


def warm_start(L, counts, w):
    """GPI table over L at w and the visit counts of the entries it copies (inherited counts)."""
    j = np.tensordot(L.stack, w, axes=([3], [0])).argmax(0)
    Psi0 = np.take_along_axis(L.stack, j[None, :, :, None], 0)[0]
    n0 = np.take_along_axis(np.stack(counts), j[None], 0)[0] if INHERIT else np.zeros((S, A))
    return Psi0, n0


def learn(w, Psi0, n0, allowed, ok, seed):
    return mf.sfql(SMP, w, Psi0, n0, allowed, ok, BUDGET, STABLE, CHECK, seed, G)


def grow(L, counts, w, how, seed):
    """A new policy for w (SFQL from the GPI table with inherited counts). 'ql': over all actions. 'cog' (model-free
    Certify-or-Grow): learn the metapolicy (on B(s)) until its policy has converged; if it is optimal, keep it and refine
    its estimates on B(s), otherwise continue over all actions."""
    Psi0, n0 = warm_start(L, counts, w)
    if how == "cog":
        allowed = np.zeros((S, A), bool)
        for a in L.acts:
            allowed[np.arange(S), a] = True
        Psi, n, act, st, cv = learn(w, Psi0, n0, allowed, ok_fn(w, allowed, estimate=False), seed)
        if ok_fn(w, estimate=False)(act, None):                                  # the metapolicy is optimal
            Psi, n, act, st2, cv2 = learn(w, Psi, n, allowed, ok_fn(w, allowed), seed + 7919)
            return Psi, n, act, st + st2, st + cv2, "metapolicy"
        Psi, n, act, st2, cv2 = learn(w, Psi, n, FULL, ok_fn(w), seed + 7919)
        return Psi, n, act, st + st2, st + cv2, "after metapolicy"
    Psi, n, act, st, cv = learn(w, Psi0, n0, FULL, ok_fn(w), seed)
    return Psi, n, act, st, cv, "q-learning"


def copy_lib(L):
    M = Library(MDP)
    M.stack, M.acts, M.kinds, M.weights = L.stack.copy(), list(L.acts), list(L.kinds), list(L.weights)
    return M


# ---------------------------------------------------------------- Lazy SF-FSA-VI (one task)
def lazy_task(ti, L, counts, how, seed):
    T, t0 = TASKS[ti], time.perf_counter()
    W, _ = kuric_vi(MDP, T, L, tol=1e-12)
    val, V = plan_value(ti, L, W)
    log = dict(task=ti, rounds=0, checks=0, passed=0, grows=[], zero_shot=rel_regret(ti, val),
               trace=[(0, 0, rel_regret(ti, val), len(L))])
    cum_st = cum_cv = 0
    grew = 0
    for r in range(ROUNDS):
        log["rounds"] += 1
        grew = 0
        for u in T.U:
            log["checks"] += 1
            if certified(L, W[u]):
                log["passed"] += 1
                continue
            Psi, n, act, st, cv, kind = grow(L, counts, W[u], how, seed)
            seed += 1
            near = min(int((act != a).sum()) for a in L.acts)
            L.add(act, Psi, kind, W[u])
            counts.append(n)
            cum_st, cum_cv = cum_st + st, cum_cv + cv
            Wn, _ = kuric_vi(MDP, T, L, W, tol=1e-12)                # replanned value after this policy (trace only)
            val, V = plan_value(ti, L, Wn, V)
            log["grows"].append(dict(u=u, round=r, kind=kind, steps=st, conv=cv, diff=near, regret=rel_regret(ti, val)))
            log["trace"].append((cum_cv, cum_st, rel_regret(ti, val), len(L)))
            grew += 1
        if not grew:
            break
        W, _ = kuric_vi(MDP, T, L, W, tol=1e-12)
    val, _ = plan_value(ti, L, W)
    log.update(certified=grew == 0, steps=cum_st, conv=cum_cv, regret=rel_regret(ti, val), size=len(L),
               seconds=time.perf_counter() - t0)
    return log


def _line(tag, log):
    kinds = [g["kind"] for g in log["grows"]]
    return (f"  {tag:14s} task {log['task']:3d}: zero-shot regret {log['zero_shot']:.1e} -> {log['regret']:.1e}, "
            f"rounds {log['rounds']}, checks {log['passed']}/{log['checks']} certified, grown {len(kinds)} "
            f"{dict((k, kinds.count(k)) for k in sorted(set(kinds)))}, steps {log['steps'] / 1e6:.2f}M (to converge "
            f"{log['conv'] / 1e6:.2f}M), base {log['size']}, {log['seconds']:.0f}s")


# ---------------------------------------------------------------- jobs
def _base_job(i):
    w = np.eye(D)[i]
    Psi0 = np.tile(w, (S, A, 1)) if INIT == "opt" else np.zeros((S, A, D))
    Psi, n, act, st, cv = mf.sfql(SMP, w, Psi0, np.zeros((S, A)), FULL, ok_fn(w), BASE, STABLE, CHECK,
                                  SEED * 100_003 + i, G)
    v, _ = tol_of(w)
    return i, Psi, n, act, st, cv, float((v - oracle().value(act, w)[REL]).max())


def job_lazy(mode, how, ti=None):
    L, counts = copy_lib(BASE_LIB["lib"]), [c.copy() for c in BASE_LIB["counts"]]
    if mode == "fresh":
        log = lazy_task(ti, L, counts, how, SEED * 1_000_003 + 1000 * ti + (0 if how == "cog" else 500))
        return ("lazy", mode, how, ti), log
    rows = []
    for i in range(len(TASKS)):
        rows.append(lazy_task(i, L, counts, how, SEED * 1_000_003 + 1000 * i + 777))
        print(_line(f"stream-{how}", rows[-1]), flush=True)
    return ("lazy", mode, how, None), rows


def job_flat(ti):
    T, t0 = TASKS[ti], time.perf_counter()
    nU = len(T.U)
    Q, n = np.full((nU * S, A), FLAT_Q0), np.zeros((nU * S, A))
    mf.seed(SEED * 7_000_003 + ti)
    state = np.array([0, -1, 0], np.int64)
    V = np.zeros((nU, S))
    tol = DELTA * OPT[ti]
    steps, good, last_bad, trace, next_rec = 0, 0, 0, [], CHECK
    while steps < FLAT:
        mf.flatq_chunk(Q, n, T.tau_a, T.exits_a, nU, S, SMP.starts, SMP.cum, SMP.nxt, SMP.obj, state, CHECK, mf.EPS,
                       mf.KAPPA, G, mf.HORIZON)
        steps += CHECK
        acts = Q.reshape(nU, S, A).argmax(2).astype(np.int64)
        V = mf.eval_product(acts, T.tau_a, T.exits_a, SMP.prob, SMP.nxt, SMP.obj, G, V, mf.EVAL_TOL)
        reg = max(OPT[ti] - V[T.u0, S0], 0.0)
        if steps >= next_rec:                                    # learning curve on a geometric grid of steps
            trace.append((steps, reg / OPT[ti]))
            next_rec = steps * 1.05
        if reg <= tol:
            good += 1
            if good >= STABLE:
                break
        else:
            good, last_bad = 0, steps
    trace.append((steps, reg / OPT[ti]))
    return ("flat", ti), dict(task=ti, steps=steps, conv=last_bad, regret=reg / OPT[ti], trace=trace,
                              converged=good >= STABLE, seconds=time.perf_counter() - t0)


def job_sfols():
    """SFOLS from the extremes (OLS over psi averaged on the relevant states), Kuric's plan after every query."""
    from q_ccs_stochastic import corner_weights, estimate_improvement, v_smp
    import heapq
    import itertools
    L, counts = copy_lib(BASE_LIB["lib"]), [c.copy() for c in BASE_LIB["counts"]]
    psi0 = lambda Psi, act: Psi[REL, act[REL]].mean(0)
    ccs = [psi0(L.stack[i], L.acts[i]) for i in range(D)]
    W_exp = [np.eye(D)[i] for i in range(D)]
    counter = itertools.count()
    visited = {tuple(np.round(w, 8)) for w in W_exp}
    rejected, heap = set(), []

    def push_corners():
        nonlocal heap
        known = visited | rejected | {tuple(np.round(np.array(h[2]), 8)) for h in heap}
        for wc in corner_weights(ccs):
            key = tuple(np.round(wc, 8))
            if key in known:
                continue
            known.add(key)
            delta = estimate_improvement(wc, ccs, W_exp)
            if delta > OLS_EPS:
                heapq.heappush(heap, (-delta, next(counter), tuple(wc)))
            else:
                rejected.add(key)

    def evaluate(Ws):
        regs = []
        for ti, T in enumerate(TASKS):
            Ws[ti], _ = kuric_vi(MDP, T, L, Ws[ti], tol=1e-12)
            regs.append(rel_regret(ti, plan_value(ti, L, Ws[ti])[0]))
        return regs
    Ws = [None] * len(TASKS)
    trace = [(0, 0, evaluate(Ws), len(L))]
    push_corners()
    cum_st = cum_cv = nq = 0
    t0 = time.perf_counter()
    while heap and nq < OLS_MAX:
        neg_p, _, wt = heapq.heappop(heap)
        w = np.array(wt)
        delta = estimate_improvement(w, ccs, W_exp)
        if delta < OLS_EPS:
            rejected.add(tuple(np.round(w, 8)))
            continue
        if heap and delta < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-delta, next(counter), wt))
            continue
        visited.add(tuple(np.round(w, 8)))
        nq += 1
        Psi0, n0 = warm_start(L, counts, w)
        Psi, n, act, st, cv = learn(w, Psi0, n0, FULL, ok_fn(w), SEED * 3_000_017 + nq)
        L.add(act, Psi, "sfols", w)
        counts.append(n)
        cum_st, cum_cv = cum_st + st, cum_cv + cv
        W_exp.append(w)
        regs = evaluate(Ws)
        trace.append((cum_cv, cum_st, regs, len(L)))
        p = psi0(Psi, act)
        if nq % 10 == 0:
            print(f"  sfols query {nq}: CCS {len(ccs)}, steps {cum_st / 1e6:.1f}M (to converge {cum_cv / 1e6:.1f}M), "
                  f"tasks within 1e-3: {np.mean(np.array(regs) <= 1e-3):.0%}, max regret {max(regs):.1e} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if p @ w <= v_smp(ccs, w) + 1e-9:
            continue
        P_old = np.array(ccs)
        ccs.append(p)
        heap = [h for h in heap if not (p @ np.array(h[2]) > (P_old @ np.array(h[2])).max() + 1e-9)]
        heapq.heapify(heap)
        push_corners()
    return ("sfols",), dict(trace=trace, queries=nq, ccs=len(ccs), open=len(heap), steps=cum_st, conv=cum_cv)


def _job(args):
    try:
        return {"lazy": job_lazy, "flat": job_flat, "sfols": job_sfols}[args[0]](*args[1:])
    except Exception:                               # one failed job must not kill the pool
        import traceback
        return ("error",) + tuple(args), traceback.format_exc()


def save(res):
    with open(OUT, "wb") as fh:
        pickle.dump(res, fh)


def main(workers=8):
    global OPT, BASE_LIB
    workers = int(workers)
    t0 = time.perf_counter()
    for T in TASKS:
        _, Vopt, _ = product_optimum(MDP, T)
        OPT.append(float(Vopt["u0"][S0]))
    print(f"{TAG}: S={S}, A={A}, d={D}, gamma={G}, {len(TASKS)} tasks (optimal values {min(OPT):.3f}-{max(OPT):.3f}); "
          f"delta {DELTA:g} (estimates {DELTA_EST:g}, relative), budgets base {BASE:,} / learn {BUDGET:,} / flat {FLAT:,}, stable {STABLE} "
          f"checks of {CHECK}", flush=True)
    mf.warmup(SMP, G)                                                            # compile before forking
    with mp.get_context("fork").Pool(min(workers, D)) as pool:
        out = sorted(pool.map(_base_job, range(D)))
    L = Library(MDP)
    for i, Psi, n, act, st, cv, loss in out:
        L.add(act, Psi, "base", np.eye(D)[i])
    BASE_LIB = dict(lib=L, counts=[o[2] for o in out])
    base = dict(steps=[o[4] for o in out], conv=[o[5] for o in out], loss=[o[6] for o in out])
    zs = [rel_regret(ti, plan_value(ti, L, kuric_vi(MDP, T, L, tol=1e-12)[0])[0]) for ti, T in enumerate(TASKS)]
    print(f"base: {D} extremes, steps {sum(base['steps']) / 1e6:.1f}M (to converge {sum(base['conv']) / 1e6:.1f}M), "
          f"max loss {max(base['loss']):.1e}; zero-shot Kuric over them: max relative regret {max(zs):.1e}, "
          f"{sum(z <= DELTA for z in zs)}/{len(zs)} tasks within {DELTA:g} ({time.perf_counter() - t0:.0f}s)", flush=True)
    res = dict(config=dict(dom=DOM, slip=SLIP, seed=SEED, delta=DELTA, delta_est=DELTA_EST, base=BASE, budget=BUDGET,
                           flat=FLAT, init=INIT, inherit=INHERIT,
                           estimate=ESTIMATE,
                           flat_q0=FLAT_Q0,
                           stable=STABLE, check=CHECK, rounds=ROUNDS, ols_eps=OLS_EPS, ols_max=OLS_MAX),
               tasks=TASK_DEFS, opt=OPT, base=base, zero_shot=zs, lazy={}, flat={}, sfols=None, errors=[])
    jobs = ([("sfols",)] if "sfols" in PARTS else []) + \
           ([("lazy", "stream", h) for h in ("cog", "ql")] + [("lazy", "fresh", h, i) for h in ("cog", "ql")
                                                                for i in range(len(TASKS))] if "lazy" in PARTS else []) + \
           ([("flat", i) for i in range(len(TASKS))] if "flat" in PARTS else [])
    with mp.get_context("fork").Pool(workers) as pool:
        for key, val in pool.imap_unordered(_job, jobs):
            if key[0] == "error":
                res["errors"].append((key[1:], val))
                print(f"  FAILED {key[1:]}\n{val}", flush=True)
            elif key[0] == "sfols":
                res["sfols"] = val
                print(f"  sfols: {val['queries']} queries, CCS {val['ccs']}, steps {val['steps'] / 1e6:.1f}M (to "
                      f"converge {val['conv'] / 1e6:.1f}M), final: {np.mean(np.array(val['trace'][-1][2]) <= DELTA):.0%}"
                      f" of tasks within {DELTA:g}", flush=True)
            elif key[0] == "flat":
                res["flat"][key[1]] = val
                print(f"  flat           task {key[1]:3d}: steps {val['steps'] / 1e6:.2f}M (to converge "
                      f"{val['conv'] / 1e6:.2f}M), regret {val['regret']:.1e}, {val['seconds']:.0f}s", flush=True)
            else:
                _, mode, how, ti = key
                if mode == "fresh":
                    res["lazy"].setdefault((mode, how), {})[ti] = val
                    print(_line(f"fresh-{how}", val), flush=True)
                else:
                    res["lazy"][(mode, how)] = val
            save(res)
    report(res)
    print(f"done in {time.perf_counter() - t0:.0f}s -> {OUT}", flush=True)
    sys.exit(1 if res["errors"] else 0)


def report(res):
    n = len(res["opt"])
    print(f"\n{TAG}: {n} tasks, base {sum(res['base']['steps']) / 1e6:.1f}M steps (to converge "
          f"{sum(res['base']['conv']) / 1e6:.1f}M); zero-shot max relative regret {max(res['zero_shot']):.1e}")
    for (mode, how), rows in sorted(res["lazy"].items()):
        rows = list(rows.values()) if isinstance(rows, dict) else rows
        kinds = [g["kind"] for r in rows for g in r["grows"]]
        print(f"  lazy-{how:3s} {mode:6s}: max regret {max(r['regret'] for r in rows):.1e}, certified "
              f"{sum(r['certified'] for r in rows)}/{len(rows)}, zero-shot tasks {sum(not r['grows'] for r in rows)}/"
              f"{len(rows)}; per task: rounds {np.mean([r['rounds'] for r in rows]):.2f}, grown "
              f"{np.mean([len(r['grows']) for r in rows]):.2f}, steps {np.mean([r['steps'] for r in rows]) / 1e6:.2f}M "
              f"(to converge {np.mean([r['conv'] for r in rows]) / 1e6:.2f}M); kinds "
              f"{dict((k, kinds.count(k)) for k in sorted(set(kinds)))}")
    if res["flat"]:
        fl = list(res["flat"].values())
        print(f"  flat Q     : converged {sum(f['converged'] for f in fl)}/{len(fl)}, per task steps "
              f"{np.mean([f['steps'] for f in fl]) / 1e6:.2f}M (to converge {np.mean([f['conv'] for f in fl]) / 1e6:.2f}M)")
    if res["sfols"]:
        s = res["sfols"]
        print(f"  sfols      : {s['queries']} queries, steps {s['steps'] / 1e6:.1f}M (to converge {s['conv'] / 1e6:.1f}M),"
              f" final max regret {max(s['trace'][-1][2]):.1e}, tasks within delta "
              f"{np.mean(np.array(s['trace'][-1][2]) <= DELTA):.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
