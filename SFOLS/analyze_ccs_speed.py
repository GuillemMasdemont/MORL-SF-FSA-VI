"""Where do the samples of Figure 2 (stochastic CCS construction) go? Splits every arm's cost by kind of learn:
  extremes   the first d learns (SFOLS: OLS's first d queries are the extremes e_i; CoG: its basis, logged at query 0)
  re-query   CoG only: OLS's first d queries are the extremes again, which CoG answers with a metapolicy learn
  certified  CoG: metapolicy learn that passed the certificate; SFOLS: every other (warm-started) full learn
  fallback   CoG: failed metapolicy learn (wasted) + the full learn that follows it (cold start, no warm tables)
for 'steps to converge' (oracle: last check at which the greedy policy was > 1e-3 below the optimum of the learner's
own problem) and for 'environment steps' (the stopping rule), as the mean over seeds of the total per kind.

usage (from SFOLS/): python analyze_ccs_speed.py [office.pkl ds.pkl]
"""
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:3] or [os.path.join(HERE, "results", "q_ccs_10seeds", "office_q_runs_slip0.05_bw.pkl"),
                          os.path.join(HERE, "results", "q_ccs_10seeds", "ds_q_runs_wind1_g0.99_eps0.001_cert0.01_stab0_bw.pkl")]
ARMS = ["standard", "standard-warm", "certify", "certify-mf"]
KINDS = ["extremes", "re-query", "certified", "fallback: meta", "fallback: full"]
CAP = {"office": 3e6, "ds": 5e6}


def learns(r, d):
    """[(kind, query, steps, conv)] with completion samples (certify-mf) included, as in the figure."""
    out, by_q = [], defaultdict(list)
    for (kind, q, _, _), (_, conv), (_, steps) in zip(r["learns"], r["log"], r["log_total"]):
        by_q[q].append((kind, steps, conv))
    for q, ls in sorted(by_q.items()):
        if q == 0 or (r["arm"].startswith("standard") and q <= d):
            out += [("extremes", q, s, c) for _, s, c in ls]
        elif r["arm"].startswith("certify") and q <= d:
            out += [("re-query", q, ls[0][1], ls[0][2])] + [("fallback: full", q, s, c) for _, s, c in ls[1:]]
        elif len(ls) == 2:
            out += [("fallback: meta", q, ls[0][1], ls[0][2]), ("fallback: full", q, ls[1][1], ls[1][2])]
        else:
            out += [("certified", q, ls[0][1], ls[0][2])]
    return out


def main():
    for f, dom in zip(FILES, ("office", "ds")):
        with open(f, "rb") as fh:
            runs = [r for r in pickle.load(fh) if "error" not in r]
        d = len(runs[0]["ccs_psi0"][0])
        print(f"\n=== {dom} (d = {d}, step cap {CAP[dom] / 1e6:g}M, {len(runs) // len(ARMS)} seeds); "
              f"per kind: mean over seeds of [#learns, total M, mean per learn M, share conv=0, share at cap]")
        for metric, j in (("steps to converge", 3), ("environment steps", 2)):
            print(f"-- {metric}")
            for arm in ARMS:
                rs = [r for r in runs if r["arm"] == arm]
                L = [learns(r, d) for r in rs]
                tot = np.mean([sum(x[j] for x in l) for l in L]) / 1e6
                parts = []
                for k in KINDS:
                    xs = [[x for x in l if x[0] == k] for l in L]
                    n = np.mean([len(x) for x in xs])
                    if n == 0:
                        continue
                    flat = [x for x in sum(xs, [])]
                    s = np.mean([sum(x[j] for x in x_) for x_ in xs]) / 1e6
                    z = np.mean([x[3] == 0 for x in flat])
                    cap = np.mean([x[2] >= CAP[dom] for x in flat])
                    parts.append(f"{k} [{n:.1f}, {s:.1f}, {s / n:.2f}, {z:.0%}, {cap:.0%}]")
                print(f"  {arm:14s} total {tot:6.1f}M | " + " | ".join(parts))
        # learn-by-learn: how good is the warm start? conv relative to the learn's length
        print("-- per learn after the extremes: median conv / median steps (M), conv = 0 share")
        for arm in ARMS:
            xs = [x for r in runs if r["arm"] == arm for x in learns(r, d) if x[0] in ("certified", "re-query")]
            print(f"  {arm:14s} n={len(xs):4d} conv {np.median([x[3] for x in xs]) / 1e6:.2f} steps "
                  f"{np.median([x[2] for x in xs]) / 1e6:.2f} conv=0 {np.mean([x[3] == 0 for x in xs]):.0%}")


if __name__ == "__main__":
    main()
