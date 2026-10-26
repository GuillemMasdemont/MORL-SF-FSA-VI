"""Cost of growing a policy in model-free Lazy SF-FSA-VI (paper Figure 5): the metapolicy growth (CoG: learn on B(s),
then refine it on B(s) if certified or continue on all actions) against Q-learning over all actions.
  per task    mean steps to converge spent on a new task (fresh: from the 6 extremes) and over the sequence (stream)
  paired      first-round growths, which both methods make at the same weights w(u) (same extremes, same plan)
  phases      CoG's two learns: metapolicy (steps st1, to converge cv1) and refinement / continuation (st2, cv2)
Reads results/lazy_mf/inherit0/ (E10, QSTABLE1 = 100), results/lazy_mf/k<K>/ (E10b, sweep lazy_cog_k.yaml: the
metapolicy learn is confirmed for K checks before the decision; Q-learning is unchanged) and results/lazy_mf/hyb/
(E10c, lazy_cog_hyb.yaml: K = 10, and pi_hyb from the completed table is tried before learning on all actions),
results/lazy_mf/hybn/ (completed entries count as their 100 samples) and results/lazy_mf/hyblocal/ (a certified
pi_hyb is then refined on B(s) plus its own actions, not on all actions).

usage (from SFOLS/): python analyze_lazy_growth.py
"""
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "lazy_mf")
DOMS = ["windy", "office0.1", "office0.3", "ds"]
SETS = [("K = 100 (E10)", "inherit0"), ("K = 30", "k30"), ("K = 10", "k10"), ("K = 1", "k1"),
        ("K = 10, pi_hyb", "hyb"), ("pi_hyb + counts", "hybn"), ("pi_hyb local", "hyblocal")]


def runs(sub, dom):
    return [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(RES, sub, f"lazy_mf_{dom}_s*.pkl")))]


def logs(r, mode, how):
    x = r["lazy"].get((mode, how), {})
    return list(x.values()) if isinstance(x, dict) else x


def main():
    base = {dom: runs("inherit0", dom) for dom in DOMS}
    for dom in DOMS:
        print(f"\n=== {dom}")
        for label, sub in SETS:
            R = runs(sub, dom)
            if not R:
                continue
            b0 = {r["config"]["seed"]: {l["task"]: l["conv"] for l in logs(r, "fresh", "ql")} for r in base[dom]}
            same = all(b0[r["config"]["seed"]] == {l["task"]: l["conv"] for l in logs(r, "fresh", "ql")} for r in R)
            row = []
            for mode in ("fresh", "stream"):
                c = np.mean([np.mean([l["conv"] for l in logs(r, mode, "cog")]) for r in R]) / 1e6
                q = np.mean([np.mean([l["conv"] for l in logs(r, mode, "ql")]) for r in R]) / 1e6
                row.append(f"{mode}: CoG {c:.3f}M vs QL {q:.3f}M per task ({c / q:.2f}x)")
            pairs, cert = defaultdict(list), []
            for r in R:
                cog = {l["task"]: l for l in logs(r, "fresh", "cog")}
                ql = {l["task"]: l for l in logs(r, "fresh", "ql")}
                for t in cog:
                    c0 = {g["u"]: g for g in cog[t]["grows"] if g["round"] == 0}
                    q0 = {g["u"]: g for g in ql[t]["grows"] if g["round"] == 0}
                    for u in set(c0) & set(q0):
                        pairs[c0[u]["kind"]].append((c0[u]["conv"], q0[u]["conv"]))
                    cert += [g["kind"] == "metapolicy" for g in cog[t]["grows"]]
            reg = [l["regret"] for r in R for l in logs(r, "fresh", "cog")]
            npol = np.mean([len(l["grows"]) for r in R for l in logs(r, "fresh", "cog")])
            print(f"  {label:14s} {' | '.join(row)}; QL reproduces E10: {same}; metapolicy certified {np.mean(cert):.0%}; "
                  f"CoG fresh: {npol:.2f} policies per task, regret mean {np.mean(reg):.2e} max {np.max(reg):.2e}")
            for kind, v in sorted(pairs.items()):
                v = np.array(v) / 1e6
                print(f"      paired, CoG {kind:16s} n={len(v):4d}: CoG {v[:, 0].mean():.2f}M vs QL {v[:, 1].mean():.2f}M")
            ph = defaultdict(list)
            for r in R:
                for l in logs(r, "fresh", "cog"):
                    for g in l["grows"]:
                        if "phases" in g:
                            ph[g["kind"]].append(g["phases"])
            for kind, P in sorted(ph.items()):
                P = np.array(P) / 1e6
                print(f"      phases, CoG {kind:16s} n={len(P):4d} st1 {P[:, 0].mean():.2f} cv1 {P[:, 1].mean():.2f} | "
                      f"st2 {P[:, 2].mean():.2f} cv2 {P[:, 3].mean():.2f} (M)")



def heldout(sub="hybc"):
    """E10d: CoG against Q-learning per seed (both from the same run, so the same extremes and tasks), environment steps
    per task as in Figure 5, for the seeds used to choose the variants (0-9) and the held-out ones (10-19)."""
    print(f"\n=== {sub}: environment steps per task, CoG / Q-learning, per-seed ratio mean +- std")
    for dom in DOMS:
        R = runs(sub, dom)
        if not R:
            continue
        for mode in ("fresh", "stream"):
            parts = []
            for name, lo, hi in (("seeds 0-9", 0, 10), ("held out 10-19", 10, 20)):
                rs = [r for r in R if lo <= r["config"]["seed"] < hi]
                if not rs:
                    continue
                c = np.array([np.mean([l["steps"] for l in logs(r, mode, "cog")]) for r in rs])
                q = np.array([np.mean([l["steps"] for l in logs(r, mode, "ql")]) for r in rs])
                ratio = c / q
                parts.append(f"{name}: {c.mean() / 1e6:.3f} vs {q.mean() / 1e6:.3f}M (ratio of means {c.mean() / q.mean():.2f}, "
                             f"per seed {ratio.mean():.2f} +- {ratio.std():.2f}, CoG cheaper in {(ratio < 1).sum()}/{len(rs)})")
            print(f"  {dom:9s} {mode:6s} " + " | ".join(parts))
        grows = {r["config"]["seed"]: sum(len(l["grows"]) for l in logs(r, "stream", "cog")) for r in R}
        kinds = defaultdict(int)
        for r in R:
            for l in logs(r, "fresh", "cog"):
                for g in l["grows"]:
                    kinds[g["kind"]] += 1
        reg = [l["regret"] for r in R for l in logs(r, "fresh", "cog")]
        print(f"      kept: growths per seed {sorted(grows.values())}; fresh kinds {dict(kinds)}; fresh regret "
              f"mean {np.mean(reg):.1e} max {np.max(reg):.1e}")


if __name__ == "__main__":
    main()
    heldout()
