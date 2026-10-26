"""Where do the samples of Figure 2 (stochastic CCS construction) go? Splits every arm's cost by kind of learn:
  extremes   the first d learns (SFOLS: OLS's first d queries are the extremes e_i; CoG: its basis, logged at query 0)
  re-query   CoG only: OLS's first d queries are the extremes again, which CoG (QREUSE=0) answers with a metapolicy learn
  learns     SFOLS: every other (warm-started) full learn; CoG: a metapolicy learn that passed the certificate
  failed     CoG: a metapolicy learn that failed the certificate
  fallback   CoG: the full learn after a failed certificate (QFALLBACK = cold | gpi | continue)
for 'steps to converge' (oracle: last check at which the greedy policy was > 1e-3 below the optimum of the learner's
own problem) and for 'environment steps' (the stopping rule), as the mean over seeds of the total per kind.
E5 runs (QREUSE=0, cold fallback) in results/q_ccs_10seeds/, E5b runs (QREUSE=1) in results/q_ccs_cog2/.

usage (from SFOLS/): python analyze_ccs_speed.py
"""
import os
import pickle
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAG = {"office": "office_q_runs_slip0.05_bw", "ds": "ds_q_runs_wind1_g0.99_eps0.001_cert0.01_stab0_bw"}
SOURCE = {"E5": "q_ccs_10seeds/{tag}.pkl", "cold": "q_ccs_cog2/{tag}_reuse.pkl",
          "gpi": "q_ccs_cog2/{tag}_reuse_fbgpi.pkl", "continue": "q_ccs_cog2/{tag}_reuse_fbcontinue.pkl",
          "n0": "q_ccs_cog2/{tag}_reuse_fbcontinue_n010.pkl"}     # E5c: warm starts count as 10 visits (Office)
SERIES = [("SFOLS", "E5", "standard"), ("SFOLS + warm start", "E5", "standard-warm"),
          ("CoG (E5)", "E5", "certify"), ("CoG + reuse", "cold", "certify"),
          ("CoG + reuse + GPI fallback", "gpi", "certify"), ("CoG + reuse + continue", "continue", "certify"),
          ("CoG-mf (E5)", "E5", "certify-mf"), ("CoG-mf + reuse", "cold", "certify-mf"),
          ("CoG-mf + reuse + GPI fallback", "gpi", "certify-mf"),
          ("CoG-mf + reuse + continue", "continue", "certify-mf"),
          ("SFOLS + warm start, N0 = 10", "n0", "standard-warm"), ("CoG + reuse + continue, N0 = 10", "n0", "certify"),
          ("CoG-mf + reuse + continue, N0 = 10", "n0", "certify-mf")]
KINDS = ["extremes", "learns", "fallback", "failed", "re-query"]
CAP = {"office": 3e6, "ds": 5e6}
D = {"office": 6, "ds": 2}
_cache = {}


def runs_of(dom, source, arm):
    f = os.path.join(RES, SOURCE[source].format(tag=TAG[dom]))
    if f not in _cache:
        _cache[f] = pickle.load(open(f, "rb")) if os.path.exists(f) else []
    return sorted((r for r in _cache[f] if r["arm"] == arm and "error" not in r), key=lambda r: r["seed"])


def learns(r, d):
    """[(kind, query, steps, conv)] with completion samples (certify-mf) included, as in the figure."""
    out, by_q = [], defaultdict(list)
    for (kind, q, _, _), (_, conv), (_, steps) in zip(r["learns"], r["log"], r["log_total"]):
        by_q[q].append((kind, steps, conv))
    for q, ls in sorted(by_q.items()):
        if q == 0 or (r["arm"].startswith("standard") and q <= d):
            out += [("extremes", q, s, c) for _, s, c in ls]
        elif r["arm"].startswith("certify") and q <= d:
            out += [("re-query", q, ls[0][1], ls[0][2])] + [("fallback", q, s, c) for _, s, c in ls[1:]]
        elif len(ls) == 2:
            out += [("failed", q, ls[0][1], ls[0][2]), ("fallback", q, ls[1][1], ls[1][2])]
        else:
            out += [("learns", q, ls[0][1], ls[0][2])]
    return out


def breakdown(dom, source, arm, metric):
    """Mean over seeds of {kind: (number of learns, total steps)} and the per-seed totals, for metric 'conv' or
    'steps'."""
    j = {"steps": 2, "conv": 3}[metric]
    L = [learns(r, D[dom]) for r in runs_of(dom, source, arm)]
    if not L:
        return None, None
    parts = {k: (np.mean([sum(x[0] == k for x in l) for l in L]),
                 np.mean([sum(x[j] for x in l if x[0] == k) for l in L])) for k in KINDS}
    return parts, np.array([sum(x[j] for x in l) for l in L])


def main():
    for dom in ("office", "ds"):
        print(f"\n=== {dom} (d = {D[dom]}, step cap {CAP[dom] / 1e6:g}M per learn); mean over seeds, per kind: "
              f"[learns, total M, M per learn]")
        for metric, name in (("conv", "steps to converge"), ("steps", "environment steps")):
            print(f"-- {name}")
            for label, source, arm in SERIES:
                parts, tot = breakdown(dom, source, arm, metric)
                if parts is None:
                    continue
                s = " | ".join(f"{k} [{n:.1f}, {x / 1e6:.1f}, {x / n / 1e6:.2f}]" for k, (n, x) in parts.items() if n)
                print(f"  {label:30s} {tot.mean() / 1e6:6.1f} +- {tot.std() / 1e6:4.1f}M ({len(tot)} seeds) | {s}")
        print("-- per run: queries, CCS size, fallbacks, reused, learns at the step cap, false certificates")
        for label, source, arm in SERIES:
            rs = runs_of(dom, source, arm)
            if rs:
                m = lambda k: np.mean([r["n"].get(k, 0) for r in rs])
                print(f"  {label:30s} queries {np.mean([r['n_queries'] for r in rs]):5.1f}, CCS "
                      f"{np.mean([len(r['ccs_psi0']) for r in rs]):5.1f}, fallbacks {m('fallbacks'):4.1f}, reused "
                      f"{m('reused'):3.1f}, capped {m('learns that hit the step cap'):5.1f}, false certificates "
                      f"{m('false certificates'):4.1f}")
    cover_report()


def cover_report(cert_tol=1e-2, delta=1e-3):
    """Instrumented Office runs (results/q_ccs_cover/, sweep office_ccs_cover.yaml): size of B(s) as the basis grows,
    and whether the metapolicies that fail (or falsely pass) the all-state certificate lose anything at the start."""
    for name, f in (("E5", "office_q_runs_slip0.05_bw.pkl"), ("E5b", "office_q_runs_slip0.05_bw_reuse_fbcontinue.pkl")):
        path = os.path.join(RES, "q_ccs_cover", f)
        if not os.path.exists(path):
            continue
        print(f"\n=== instrumented Office runs ({name}), {path}")
        for arm in ("certify", "certify-mf"):
            rs = [r for r in pickle.load(open(path, "rb")) if r["arm"] == arm and "error" not in r]
            if not rs:
                continue
            first = [next(c for c in r["basis_cover"] if c[0] > D["office"]) for r in rs]   # first query after the extremes
            last = [r["basis_cover"][-1] for r in rs]
            print(f"  {arm}: mean |B(s)| (of 4) {np.mean([c[2] for c in first]):.2f} at the first query after the extremes (basis {np.mean([c[1] for c in first]):.1f}) -> "
                  f"{np.mean([c[2] for c in last]):.2f} at the last query (basis {np.mean([c[1] for c in last]):.1f})")
            C = np.array([c for r in rs for c in r["cert"] if c[0] > D["office"] or name == "E5b"])
            bad, loss = C[:, 1] > cert_tol, C[:, 2]
            print(f"    true gap > {cert_tol:g}: {bad.mean():.0%} of {len(C)} metapolicies; of these, loss at the start "
                  f"<= {delta:g}: {(loss[bad] <= delta).mean():.0%} (median loss {np.median(loss[bad]):.1e}); "
                  f"gap <= {cert_tol:g} but loss > {delta:g}: {(loss[~bad] > delta).mean():.0%}")


if __name__ == "__main__":
    main()
