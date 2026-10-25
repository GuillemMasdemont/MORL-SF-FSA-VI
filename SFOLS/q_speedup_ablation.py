"""Why is Certify-or-Grow faster than warm-started SFOLS in Figure 1 (deterministic DST, SFQL with alpha = 1)?

Same information for both learners: at OLS query q, the warm start is GPI over the EXACT SF tables of the policies
optimal at queries 1..q-1 (what both arms hold after learning them to a fixed point). The only difference is the
action set: full (SFOLS + warm start) or B(s) = the extreme policies' own actions (Certify-or-Grow).
The learner is notebook 13's sfql unchanged (exploring starts, epsilon-greedy, stop once every tracked pair is
revisited after the last change); this copy only records, at every check (oracle, measurement only):
  t_table    last change of any tracked SF entry           (Figure 1's "steps to converge")
  t_greedy   last change of an entry (s, a) with a = pi*(s)
  t_policy   last check at which the greedy policy was not optimal at some state
  t_value    last check at which the greedy value estimate was off by > 1e-6 at some state
plus the stopping time and how many tracked entries start wrong.

usage (okb env, from SFOLS/): python q_speedup_ablation.py [seeds]
"""
import json
import os
import pickle
import sys
import random
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "q_speedup_ablation_dst.pkl")


def _load_notebook():
    nb = json.load(open(os.path.join(HERE, "13_ccs_speedup_dst.ipynb")))
    ns = {}
    for i in (2, 4, 6, 8, 10, 12, 14, 28):
        exec("".join(nb["cells"][i]["source"]), ns)
    return ns


NB = _load_notebook()
S, A, NONTERM, GAMMA, PHI = NB["S"], NB["N_ACTIONS"], NB["NONTERM"], NB["GAMMA"], NB["PHI"]
ROWS = np.arange(S)


def query_sequence():
    ws = []

    def q(w):
        ws.append(w.copy())
        return NB["solve_exact"](w)

    NB["ols"](q)
    return ws


def instrumented_sfql(w, psi_init, allowed, rng, pi_star, V_star):
    """notebook 13's sfql (same exploration, update and stopping rule) + oracle timestamps."""
    NEXT_L, DONE_L, NONTERM_IDX = NB["_NEXT_L"], NB["_DONE_L"], NB["_NONTERM_IDX"]
    EPS, HORIZON, CHECK, TOL, MAX = NB["Q_EPS"], NB["Q_HORIZON"], NB["Q_CHECK"], NB["Q_TOL"], NB["Q_MAX_STEPS"]
    Psi = psi_init.copy()
    Qw = Psi @ w
    masked = np.where(allowed, 0.0, -np.inf)
    allowed_l = [np.flatnonzero(allowed[s]).tolist() for s in range(S)]
    tracked = allowed & NONTERM[:, None]
    greedy_pair = np.zeros((S, A), bool)
    greedy_pair[ROWS, pi_star] = True
    last_update = np.full((S, A), -1)
    t = dict(table=0, greedy=0, policy=0, value=0)

    def greedy(s):
        return int((Qw[s] + masked[s]).argmax())

    def check(steps):
        g = (Qw + masked).argmax(1)
        V = NB["value"](g, w)
        if (V_star - V)[NONTERM].max() > 1e-6:
            t["policy"] = steps
        if np.abs(Qw[ROWS, g] - V_star)[NONTERM].max() > 1e-6:
            t["value"] = steps

    check(0)
    steps, t_ep, s, stop = 0, HORIZON, None, None
    while steps < MAX:
        if s is None or t_ep >= HORIZON:
            s, t_ep = rng.choice(NONTERM_IDX), 0
        a = rng.choice(allowed_l[s]) if rng.random() < EPS else greedy(s)
        ns, done = NEXT_L[s][a], DONE_L[s][a]
        target = PHI[s, a] if done else PHI[s, a] + GAMMA * Psi[ns, greedy(ns)]
        delta = target - Psi[s, a]
        steps += 1
        if np.abs(delta).max() > TOL:
            t["table"] = steps
            if greedy_pair[s, a]:
                t["greedy"] = steps
        Psi[s, a] += delta
        Qw[s, a] = Psi[s, a] @ w
        last_update[s, a] = steps
        t_ep += 1
        s = None if done else ns
        if steps % CHECK == 0:
            check(steps)
            if last_update[tracked].min() > t["table"]:
                stop = steps
                break
    return dict(t, stop=stop if stop is not None else MAX, tracked=int(tracked.sum()))


def run_seed(seed):
    ws = query_sequence()
    exact = [NB["solve_exact"](w) for w in ws]                       # optimal policy + exact SF table per query
    ext = [NB["solve_exact"](np.eye(2)[i]) for i in range(2)]
    allowed_B = np.zeros((S, A), bool)
    for p in ext:
        allowed_B[ROWS, p.actions] = True
    FULL = np.ones((S, A), bool)
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(2, len(ws)):
        w = ws[q]
        pi_star, V_star, _ = NB["policy_iteration"](w)
        target = NB["sf_backup"](NB["sf_state"](pi_star))              # SF of (a, then pi*)
        init = NB["gpi_init"](w, [p.psi_sa for p in ext + exact[:q]])
        for arm, allowed in (("full", FULL), ("restricted", allowed_B)):
            tr = allowed & NONTERM[:, None]
            wrong = np.abs(init - target).max(-1) > 1e-6
            g0 = np.where(allowed, init @ w, -np.inf).argmax(1)
            r = instrumented_sfql(w, np.where(allowed[..., None], init, 0.0), allowed,
                                  random.Random(int(rng.integers(1 << 31))), pi_star, V_star)
            r.update(arm=arm, q=q + 1, seed=seed, wrong_init=int((wrong & tr).sum()),
                     wrong_greedy=int((wrong & tr)[ROWS, pi_star].sum()),
                     policy_ok_init=bool((V_star - NB["value"](g0, w))[NONTERM].max() <= 1e-6))
            rows.append(r)
    return rows


def main(seeds=10):
    with Pool(min(seeds, 8)) as pool:
        rows = [r for rs in pool.map(run_seed, range(seeds)) for r in rs]
    with open(OUT, "wb") as fh:
        pickle.dump(rows, fh)
    report(rows)


def report(rows):
    keys = ["table", "greedy", "value", "policy", "stop", "wrong_init", "wrong_greedy", "tracked", "policy_ok_init"]
    print("DST, queries 3+, same warm start (GPI over exact tables of earlier queries); sums over queries, "
          "median over seeds")
    seeds = sorted({r["seed"] for r in rows})
    for arm in ("full", "restricted"):
        tot = {k: np.median([sum(r[k] for r in rows if r["arm"] == arm and r["seed"] == s) for s in seeds])
               for k in keys}
        print(f"  {arm:10s} " + "  ".join(f"{k} {tot[k]:,.0f}" for k in keys))
    print("per query (seed 0): t_table / t_greedy / t_value / t_policy / stop | wrong init entries (greedy ones)")
    for q in sorted({r["q"] for r in rows}):
        line = f"  q{q:2d}"
        for arm in ("full", "restricted"):
            r = next(x for x in rows if x["q"] == q and x["arm"] == arm and x["seed"] == 0)
            line += (f" | {arm[:4]} {r['table']:6d} {r['greedy']:6d} {r['value']:6d} {r['policy']:6d} "
                     f"{r['stop']:6d} | {r['wrong_init']:3d} ({r['wrong_greedy']:2d})")
        print(line)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
