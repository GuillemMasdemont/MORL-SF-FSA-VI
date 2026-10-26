"""Model-free SF-FSA-VI on slippery Office: how many environment steps does each automaton state u need to reach the
optimal policy, given a base of LEARNED successor features?

Setup (no learner uses the model; the model only measures):
  base     SFQL (vector TD, decaying step 1/(n+1)^KAPPA, exploring starts, eps-greedy) at the 6 extreme weights, and
           optionally at the next OLS weights; the visit count n_j(s,a) of every learned entry is kept.
  problems the automaton states u of the FSA tasks, with exit weights w(u) built from the TRUE optimal downstream
           values, so every learner faces the same problem (w(u), entry states R_u).
  learners scalar Q-learning at w(u), same exploration, step size 1/(n(s,a)+1)^KAPPA:
    scratch    Q0 = max(w) (tight optimistic), n0 = 0, all actions
    gpi        Q0 = GPI estimate max_j psi_j(s,a).w over the base, n0 = 0, all actions
    gpi-inh    same, n0(s,a) = n_{j*}(s,a): the visit count of the base entry GPI picked (inherited counts)
    meta       Q0 = GPI estimate, acting and bootstrapping restricted to B(s) (the base policies' actions), n0 = 0
    meta-inh   same with inherited counts
  measure  'steps to converge' = last check (every CHECK steps) at which the greedy policy's TRUE value at some
           entry state of u was more than DELTA below the optimum V*_{w(u)} (the full optimum, also for meta). A run
           stops after STABLE consecutive good checks or BUDGET steps.

usage (okb env, from SFOLS/): python q_fsa_stochastic.py [n_random_tasks] [extra_base] [workers]
"""
import bisect
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from certlib.domains import office, windy_two_stage     # noqa: E402
from certlib.experiments import tasks_for               # noqa: E402
from certlib.pertask import entry_states                # noqa: E402
from gpi_basis.fsa import _order, symbols                # noqa: E402

SLIP = float(os.environ.get("QSLIP", 0.05))
KAPPA, EPS, HORIZON, CHECK = 0.7, 0.1, 100, 1000
BASE_STEPS = int(float(os.environ.get("QBASE", 2.5e6)))
BUDGET = int(float(os.environ.get("QBUDGET", 2e6)))
STABLE = int(os.environ.get("QSTABLE", 300))              # good checks in a row before stopping (300k steps)
DELTA = 1e-3
SEEDS = int(os.environ.get("QSEEDS", 2))                   # learner seeds per problem (the base is learned once)
OUT = os.path.join(os.environ.get("SF_OUT") or os.path.join(HERE, "results"), f"fsa_q_office_slip{SLIP}.pkl")

DOM = os.environ.get("QDOM", "office")                    # office (slip QSLIP) | windy (two-stage, wind +-3)
mdp = office(p=SLIP) if DOM == "office" else windy_two_stage()
S, A, D, G = mdp.S, mdp.A, mdp.D, mdp.gamma
ROWS = np.arange(S)
# outcomes per (s, a): non-terminal next state with prob P, or entering object o (episode ends, feature e_o)
_CUM, _OUT = [[None] * A for _ in range(S)], [[None] * A for _ in range(S)]
for s in range(S):
    for a in range(A):
        outs = [(float(p), int(n), -1) for n, p in enumerate(mdp.P[s, a]) if p > 0]
        outs += [(float(p), -1, int(o)) for o, p in enumerate(mdp.PHI[s, a]) if p > 0]
        _CUM[s][a] = np.cumsum([o[0] for o in outs]).tolist()
        _CUM[s][a][-1] = 1.0 + 1e-12
        _OUT[s][a] = [(o[1], o[2]) for o in outs]
ALL = list(range(S))


def step(s, a, rng):
    """Returns (next state or -1, object index or -1)."""
    return _OUT[s][a][bisect.bisect_right(_CUM[s][a], rng.random())]


# ---------------------------------------------------------------- oracle (measurement only)
def value(act, w):
    return np.linalg.solve(np.eye(S) - G * mdp.P[ROWS, act], mdp.PHI[ROWS, act] @ w)


def vstar(w, allowed=None):
    act = np.zeros(S, int) if allowed is None else np.argmax(allowed, 1)
    for _ in range(500):
        V = value(act, w)
        Q = mdp.PHI @ w + G * mdp.P @ V
        if allowed is not None:
            Q = np.where(allowed, Q, -np.inf)
        best = Q.argmax(1)
        new = np.where(Q[ROWS, act] >= Q[ROWS, best] - 1e-12, act, best)
        if np.array_equal(new, act):
            return act, V
        act = new
    raise RuntimeError("PI did not converge")


# ---------------------------------------------------------------- base: model-free SFQL with visit counts
def sf_learn(w, seed):
    rng = random.Random(seed)
    Psi = np.tile(np.eye(D)[int(np.argmax(w))], (S, A, 1))     # tight optimistic: Q = max(w)
    Qw = Psi @ w
    n = np.zeros((S, A), int)
    s, t = None, HORIZON
    for _ in range(BASE_STEPS):
        if s is None or t >= HORIZON:
            s, t = rng.choice(ALL), 0
            a = rng.randrange(A)
        else:
            a = rng.randrange(A) if rng.random() < EPS else int(Qw[s].argmax())
        ns, o = step(s, a, rng)
        if o >= 0:
            target = np.eye(D)[o]
        else:
            target = G * Psi[ns, int(Qw[ns].argmax())]
        n[s, a] += 1
        Psi[s, a] += (target - Psi[s, a]) / (n[s, a] + 1) ** KAPPA
        Qw[s, a] = Psi[s, a] @ w
        t += 1
        s = None if o >= 0 else ns
    return Psi, n


def ols_weights(k):
    """The first k non-extreme query weights of exact OLS (eps = 0.05) on the same slippery Office (cached)."""
    f = os.path.join(HERE, "results", f"office_ols_weights_slip{SLIP}.npy")
    if not os.path.exists(f):
        import q_ccs_stochastic as Q
        Q.SLIP = SLIP
        M = Q._model()
        W = []

        def query(w):
            W.append(w)
            pi, _ = Q.policy_iteration(w, M)
            return Q.Policy(w, pi, Q.sf_state(pi, M), None, "exact")
        Q.ols(query, tol=Q.OLS_EPS)
        np.save(f, np.array(W))
    W = np.load(f)
    return [w for w in W if w.max() < 1 - 1e-9][:k]


# ---------------------------------------------------------------- per-automaton-state learner (scalar Q)
def q_learn(w, Q0, n0, allowed, Ru, vopt, seed):
    rng = random.Random(seed)
    Q = Q0.copy()
    n = n0.astype(float).copy()
    masked = np.where(allowed, 0.0, -np.inf)
    allowed_l = [np.flatnonzero(allowed[s]).tolist() for s in range(S)]
    wl = w.tolist()

    def bad():
        return (vopt - value((Q + masked).argmax(1), w)[Ru]).max() > DELTA

    ok0 = not bad()
    last_bad = -1 if ok0 else 0
    good, steps, s, t = 0, 0, None, HORIZON
    while steps < BUDGET:
        if s is None or t >= HORIZON:
            s, t = rng.choice(ALL), 0
            a = rng.choice(allowed_l[s])
        else:
            a = rng.choice(allowed_l[s]) if rng.random() < EPS else int((Q[s] + masked[s]).argmax())
        ns, o = step(s, a, rng)
        target = wl[o] if o >= 0 else G * (Q[ns] + masked[ns]).max()
        n[s, a] += 1
        Q[s, a] += (target - Q[s, a]) / (n[s, a] + 1) ** KAPPA
        steps += 1
        t += 1
        s = None if o >= 0 else ns
        if steps % CHECK == 0:
            if bad():
                last_bad, good = steps, 0
            else:
                good += 1
                if good >= STABLE:
                    break
    final = float((vopt - value((Q + masked).argmax(1), w)[Ru]).max())
    return dict(to_converge=max(last_bad, 0), converged=final <= DELTA and good > 0, steps=steps, final_gap=final,
                ok_at_start=ok0, never_bad=last_bad < 0)


# ---------------------------------------------------------------- problems and jobs
def problems(n_random):
    tasks = tasks_for("office", mdp, np.random.default_rng(1))[:3 + n_random]
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    cache, out, seen = {}, [], {}

    def V(w):
        k = tuple(np.round(w, 12))
        if k not in cache:
            cache[k] = vstar(w)[1]
        return cache[k]
    for ti, task in enumerate(tasks):
        R, Vu = entry_states(mdp, task), {}
        for u in _order(task):
            w = np.zeros(D)
            for sym, v in task[u].items():
                for e in SYM[sym]:
                    w[objs.index(e)] = 1.0 if v == "T" else Vu[v][ex[e]]
            Vu[u] = V(w)
            key = (tuple(np.round(w, 10)), tuple(R[u]))
            if key not in seen:
                seen[key] = len(out)
                out.append(dict(w=w, Ru=R[u], vopt=Vu[u][R[u]], tasks=[]))
            out[seen[key]]["tasks"].append(ti)
    return out


BASES, PROBS = {}, []


def _init(name, w):
    stack, counts, acts = BASES[name]
    Qj = np.einsum("ksad,d->ksa", stack, w)
    j = Qj.argmax(0)
    Q0 = np.take_along_axis(Qj, j[None], 0)[0]
    n0 = np.take_along_axis(counts, j[None], 0)[0]
    allowed = np.zeros((S, A), bool)
    for a in acts:
        allowed[ROWS, a] = True
    return Q0, n0, allowed


def _job(args):
    i, arm, base, seed = args
    p = PROBS[i]
    w = p["w"]
    if arm == "scratch":
        Q0, n0, allowed = np.full((S, A), w.max()), np.zeros((S, A)), np.ones((S, A), bool)
    else:
        Q0, n0, allowed = _init(base, w)
        if not arm.endswith("-inh"):
            n0 = np.zeros((S, A))
        if arm.startswith("gpi"):
            allowed = np.ones((S, A), bool)
        else:
            Q0 = np.where(allowed, Q0, 0.0)
    t0 = time.perf_counter()
    r = q_learn(w, Q0, n0, allowed, p["Ru"], p["vopt"], seed)
    r.update(problem=i, arm=arm, base=base, seed=seed, seconds=time.perf_counter() - t0)
    return r


def _base_job(args):
    i, w, seed = args
    return i, sf_learn(w, seed)


def main(n_random=17, extra=14, workers=8):
    global PROBS
    t0 = time.perf_counter()
    PROBS = problems(n_random)
    Wb = [np.eye(D)[i] for i in range(D)] + ols_weights(extra)
    print(f"slip {SLIP}: {len(PROBS)} automaton-state problems from {3 + n_random} tasks; base weights {len(Wb)} "
          f"(6 extremes + {len(Wb) - D} OLS); learning the base model-free, {BASE_STEPS:,} steps each", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        learned = dict(pool.map(_base_job, [(i, w, 1000 + i) for i, w in enumerate(Wb)]))
    stack = np.stack([learned[i][0] for i in range(len(Wb))])
    counts = np.stack([learned[i][1] for i in range(len(Wb))])
    acts = [np.argmax(stack[i] @ Wb[i], 1) for i in range(len(Wb))]
    for i, w in enumerate(Wb):
        loss = (vstar(w)[1] - value(acts[i], w))[mdp.relevant].max()
        if loss > 1e-2:
            print(f"  warning: base policy {i} is {loss:.3f} below optimal at a relevant state", flush=True)
    BASES["6"] = (stack[:D], counts[:D], acts[:D])
    BASES[str(len(Wb))] = (stack, counts, acts)
    for name, (_, _, ac) in BASES.items():
        allowed = np.zeros((S, A), bool)
        for a in ac:
            allowed[ROWS, a] = True
        print(f"  base {name}: |B(s)| mean {allowed.sum(1).mean():.2f} of {A}; median inherited count "
              f"{np.median(BASES[name][1]):.0f}", flush=True)
    jobs = [(i, "scratch", "-", sd) for sd in range(SEEDS) for i in range(len(PROBS))]
    jobs += [(i, arm, b, sd) for sd in range(SEEDS) for b in BASES for i in range(len(PROBS))
             for arm in ("gpi", "gpi-inh", "meta", "meta-inh")]
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap_unordered(_job, jobs, chunksize=1):
            rows.append(r)
            if len(rows) % 25 == 0:
                print(f"  {len(rows)}/{len(jobs)} learners done ({time.perf_counter() - t0:.0f}s)", flush=True)
    # restricted optimum: can the metapolicy reach the full optimum at all?
    reach = {b: [float((p["vopt"] - vstar(p["w"], _init(b, p["w"])[2])[1][p["Ru"]]).max()) for p in PROBS] for b in BASES}
    with open(OUT, "wb") as fh:
        pickle.dump(dict(rows=rows, problems=PROBS, base_weights=Wb, reach=reach, budget=BUDGET), fh)
    report(rows, PROBS, reach)


def report(rows, probs, reach):
    print(f"\nsteps until the greedy policy is within {DELTA} of the optimum at every entry state "
          f"(budget {BUDGET:,}; 'not reached' counts as the budget)")
    print(f"{'base':>5s} {'learner':9s} {'reached':>9s} {'optimal at start':>16s} {'never off':>10s} {'median':>9s} "
          f"{'mean':>9s} {'per task (mean)':>15s}")
    ntask = 1 + max(t for p in probs for t in p["tasks"])
    for b in ["-"] + [x for x in sorted({r["base"] for r in rows}) if x != "-"]:
        for arm in ("scratch", "gpi", "gpi-inh", "meta", "meta-inh"):
            rs = [r for r in rows if r["arm"] == arm and r["base"] == b]
            if not rs:
                continue
            c = np.array([r["to_converge"] if r["converged"] else BUDGET for r in rs], float)
            cost = {(r["problem"], r["seed"]): x for r, x in zip(rs, c)}
            seeds = sorted({r["seed"] for r in rs})
            per_task = [sum(cost[(i, sd)] for i, p in enumerate(probs) if t in p["tasks"])
                        for t in range(ntask) for sd in seeds]
            print(f"{b:>5s} {arm:9s} {sum(r['converged'] for r in rs):4d}/{len(rs):<4d} "
                  f"{sum(r['ok_at_start'] for r in rs):10d}/{len(rs):<5d} {sum(r['never_bad'] for r in rs):5d}/{len(rs):<4d} "
                  f"{np.median(c):9,.0f} {c.mean():9,.0f} {np.mean(per_task):15,.0f}")
        if b != "-":
            print(f"      restricted optimum within {DELTA} of the full optimum: "
                  f"{sum(x <= DELTA for x in reach[b])}/{len(reach[b])} problems")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:4]))
