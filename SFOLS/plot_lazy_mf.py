"""Figure of the model-free comparison (q_lazy_fsa.py), in the spirit of Kuric et al.'s Figure 3, one row per domain:
(left) a new task: relative regret of the executed plan at the start state against the environment steps spent on that
task, averaged over tasks and seeds, for flat Q-learning on the product MDP (from scratch) and Lazy SF-FSA-VI from the
extremes, growing with Q-learning over all actions or with the metapolicy; SF-FSA-VI over SFOLS spends nothing per task
(its final regret is the horizontal line, its cost is in the middle panel); (middle) cumulative environment steps over
the sequence of tasks, the extremes included, keeping the policies added by Lazy SF-FSA-VI; (right) how the new policies
of the metapolicy growth were obtained (certified metapolicy, certified pi_hyb, learned on all actions), the share of
each, and their environment steps relative to Q-learning's at the same weight (first-round growths, which both Lazy
variants make at the same w(u); ratio of means with a 95% bootstrap interval, all seeds of the CoG runs).

Reads results/lazy_mf/inherit{0,1}/lazy_mf_<domain>_s<seed>.pkl (cluster/sweeps/lazy_mf.yaml, lazy_mf_ds.yaml, then
sweep.py merge --only per QINHERIT value): warm starts without / with inherited visit counts. Every method is drawn with
its better setting (SOURCE); the summary prints both. Writes paper/figures/lazy_mf.pdf and notes/figures/lazy_mf.pdf.
usage (from SFOLS/): python plot_lazy_mf.py [results dir] [domain tags, comma-separated] [output name]
(output name other than lazy_mf: written to notes/figures only)
"""
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from plot_ccs_cost_combined import PUB_RC, TEXTWIDTH, USE_TEX

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "lazy_mf")
DOMS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["windy", "office0.3"])     # the paper: windy on top
NAMES = {"office0.1": r"Office, $p=0.1$", "office0.3": r"Office, $p=0.3$", "windy": "Windy", "ds": "Double slit"}
STEM = sys.argv[3] if len(sys.argv) > 3 else "lazy_mf"
OUTS = ([os.path.join(HERE, "..", "paper", "figures", f"{STEM}.pdf")] if STEM == "lazy_mf" else []) + \
       [os.path.join(HERE, "..", "notes", "figures", f"{STEM}.pdf")]
METHODS = [("flat", "Q-learning on the product MDP", dict(color="#7a7a7a", ls=":")),
           ("sfols", "SF-FSA-VI over SFOLS", dict(color="#2a78d6", ls="-")),
           ("ql", r"Lazy SF-FSA-VI, Q-learning", dict(color="#eb6834", ls="--")),
           ("cog", r"Lazy SF-FSA-VI, metapolicy (ours)", dict(color="#1baf7a", ls="-."))]
FLOOR = 1e-5
X = np.logspace(4, 9, 400)
# better setting per method: Lazy grows faster without inherited counts everywhere; SFOLS is cheaper with them in windy and
# Office (its queries are close to stored policies) and without them in the double slit; flat Q-learning has no warm start.
# The metapolicy growth is from E10d (hybc, sweep lazy_cog_hybc.yaml: Algorithm 4's ladder, metapolicy -> pi_hyb -> all
# actions, deciding after 10 checks; 20 seeds, the Q-learning growth of the same runs is identical to E10's); COGSRC=k10
# gives E10b (no pi_hyb), COGSRC=inherit0 E10 (100 checks).
SOURCE = {"flat": "inherit0", "sfols": "inherit1", "ql": "inherit0", "cog": os.environ.get("COGSRC", "hybc")}
KINDS = [("metapolicy", "meta-\npolicy"), ("hybrid", r"$\pi^{\mathrm{hyb}}$"), ("after metapolicy", "all\nactions")]
INK = "#333333"
SOURCE_DOM = {"ds": {"sfols": "inherit0"}}


def source(dom, method):
    return SOURCE_DOM.get(dom, {}).get(method, SOURCE[method])


def load(dom, sub=None):
    runs = []
    for f in sorted(glob.glob(os.path.join(RES, sub or "", f"lazy_mf_{dom}_s*.pkl"))):
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))
    return runs


def mixed(dom):
    """One run per seed whose entries come from the setting chosen for each method in SOURCE."""
    src = {m: source(dom, m) for m in SOURCE}
    by = {sub: {r["config"]["seed"]: r for r in load(dom, sub)} for sub in set(src.values())}
    seeds = sorted(set.intersection(*(set(v) for v in by.values()))) if by else []
    out = []
    for sd in seeds:
        r = dict(by[src["cog"]][sd])
        r["flat"] = by[src["flat"]][sd]["flat"]
        r["sfols"] = by[src["sfols"]][sd]["sfols"]
        r["lazy"] = {k: v for k, v in r["lazy"].items() if k[1] == "cog"}
        r["lazy"].update({k: v for k, v in by[src["ql"]][sd]["lazy"].items() if k[1] == "ql"})
        out.append(r)
    return out


def step_curve(xs, ys, grid, before=1.0):
    """Value of a right-continuous step function given at points xs (sorted) on the grid; `before` until xs[0]."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    idx = np.searchsorted(xs, grid, side="right") - 1
    return np.where(idx >= 0, ys[np.clip(idx, 0, None)], before)


def curves(runs):
    """Relative regret on the grid X of steps spent on a task, for every (seed, task), per method; SF-FSA-VI over SFOLS:
    the final regret (it learns nothing per task)."""
    out = defaultdict(list)
    for r in runs:
        n = len(r["opt"])
        for ti in range(n):
            f = r["flat"].get(ti)
            if f:
                out["flat"].append(step_curve([t[0] for t in f["trace"]], [t[1] for t in f["trace"]], X))
            for how in ("cog", "ql"):
                for mode in ("fresh", "stream"):
                    rows = r["lazy"].get((mode, how))
                    lg = rows.get(ti) if isinstance(rows, dict) else (rows[ti] if rows else None)
                    if lg:
                        out[f"{how}-{mode}"].append(step_curve([t[1] for t in lg["trace"]], [t[2] for t in lg["trace"]], X))
            if r["sfols"]:
                out["sfols"].append(np.full(len(X), r["sfols"]["trace"][-1][2][ti]))
    return {k: np.maximum(np.array(v), FLOOR) for k, v in out.items()}


def stream(runs):
    """Cumulative environment steps over the task sequence (index 0: before the first task), per method and seed."""
    out = defaultdict(list)
    for r in runs:
        base, n = sum(r["base"]["steps"]), len(r["opt"])
        if r["flat"]:
            out["flat"].append(np.concatenate([[0], np.cumsum([r["flat"][ti]["steps"] for ti in range(n)])]))
        for how in ("cog", "ql"):
            rows = r["lazy"].get(("stream", how))
            if rows:
                out[how].append(base + np.concatenate([[0], np.cumsum([x["steps"] for x in rows])]))
        if r["sfols"]:
            out["sfols"].append(np.full(n + 1, base + r["sfols"]["steps"], float))
    return {k: np.array(v, float) for k, v in out.items()}


def kind_costs(dom):
    """Per way the metapolicy growth obtained a new policy: its share among all new policies from the extremes, and the
    environment steps of the first-round growths relative to Q-learning's at the same weight (ratio of means, 95%
    bootstrap interval over the pairs), from every seed of the CoG runs."""
    pairs, count = defaultdict(list), defaultdict(int)
    for r in load(dom, SOURCE["cog"]):
        cog, ql = r["lazy"].get(("fresh", "cog"), {}), r["lazy"].get(("fresh", "ql"), {})
        for t, lg in cog.items():
            for g in lg["grows"]:
                count[g["kind"]] += 1
            c0 = {g["u"]: g for g in lg["grows"] if g["round"] == 0}
            q0 = {g["u"]: g for g in ql[t]["grows"] if g["round"] == 0}
            for u in set(c0) & set(q0):
                pairs[c0[u]["kind"]].append((c0[u]["steps"], q0[u]["steps"]))
    rng, total, out = np.random.default_rng(0), sum(count.values()), {}
    for k, v in pairs.items():
        v = np.array(v, float)
        boot = [v[i, 0].mean() / v[i, 1].mean() for i in (rng.integers(0, len(v), len(v)) for _ in range(2000))]
        out[k] = (v[:, 0].mean() / v[:, 1].mean(), np.percentile(boot, [2.5, 97.5]), count[k] / total, len(v))
    return out


def kinds_panel(ax, dom):
    K = kind_costs(dom)
    names = [(k, lab) for k, lab in KINDS if k in K]
    x = np.arange(len(names))
    val = np.array([K[k][0] for k, _ in names])
    err = np.array([[K[k][0] - K[k][1][0], K[k][1][1] - K[k][0]] for k, _ in names]).T
    ax.bar(x, val, width=0.62, color=METHODS[3][2]["color"], alpha=0.85, lw=0)
    ax.errorbar(x, val, yerr=err, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5, capthick=0.6)
    ax.axhline(1.0, color=METHODS[2][2]["color"], ls="--", lw=1.1)               # Q-learning at the same weight
    for xi, (k, _) in zip(x, names):
        pct = r"\%" if USE_TEX else "%"
        ax.text(xi, 0.04, f"{100 * K[k][2]:.0f}{pct}", ha="center", va="bottom", fontsize=6.5, color="white")
        ax.text(xi, K[k][1][1] + 0.03, f"{100 * (K[k][0] - 1):+.0f}{pct}", ha="center", va="bottom", fontsize=6.5,
                color=INK)
    ax.set_xticks(x, [lab for _, lab in names])
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.45)
    ax.set_ylabel("steps / Q-learning's")
    ax.set_title("new policies", fontsize=8)


def style(ax, tag):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.02, 1.03, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def main():
    data = {dom: mixed(dom) for dom in DOMS}
    data = {d: v for d, v in data.items() if v}
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 1.95 * len(data) + 0.35), layout="constrained")
        subs = np.atleast_1d(fig.subfigures(len(data), 1, hspace=0.04))
        tags = iter("abcdefghijkl")
        axes = []
        for sub, (dom, runs) in zip(subs, data.items()):
            sub.suptitle(NAMES.get(dom, dom), fontsize=8.5)
            ax_l, ax_s, ax_k = row = sub.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.72]})
            axes.append(row)
            C = curves(runs)
            for key, lab, st in METHODS:                                   # Lazy: from the extremes (fresh)
                k = key if key in ("flat", "sfols") else f"{key}-fresh"
                if k in C:
                    ax_l.plot(X, C[k].mean(0), lw=1.3, label=lab, **st)
            ax_l.set_xscale("log")
            ax_l.set_yscale("log")
            ax_l.set_ylim(FLOOR, 1.5)
            ax_l.set_xlim(X[0], X[-1])
            ax_l.set_xlabel("environment steps on the task")
            ax_l.set_ylabel("relative regret")
            ax_l.set_title("a new task", fontsize=8)
            style(ax_l, next(tags))
            if len(runs[0]["opt"]) == 1:                                    # one task: no sequence to show
                ax_s.axis("off")
                kinds_panel(ax_k, dom)
                style(ax_k, next(tags))
                continue
            Sd = stream(runs)
            for key, lab, st in METHODS:
                if key in Sd:
                    Y = Sd[key] / 1e6
                    x = np.arange(Y.shape[1])
                    ax_s.plot(x, Y.mean(0), lw=1.3, label=lab, **st)
                    ax_s.fill_between(x, Y.min(0), Y.max(0), color=st["color"], alpha=0.15, lw=0)
            ax_s.set_yscale("log")
            ax_s.set_xlabel("tasks solved")
            ax_s.set_ylabel(r"environment steps ($\times 10^6$)")
            ax_s.set_title("sequence of tasks", fontsize=8)
            style(ax_s, next(tags))
            kinds_panel(ax_k, dom)
            style(ax_k, next(tags))
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, handlelength=2.4, columnspacing=1.2)
        for out in OUTS:
            fig.savefig(out)
            print(f"saved {os.path.normpath(out)}  (TeX rendering: {USE_TEX})")
    for dom in DOMS:
        for sub in ("inherit0", "inherit1"):
            runs = load(dom, sub)
            if runs:
                print(f"\n===== {dom}, {sub}")
                summary(dom, runs)


def summary(dom, runs):
    """The numbers of the text: per new task, how many certificates and new policies, at which cost; zero-shot share."""
    n, seeds = len(runs[0]["opt"]), len(runs)
    base = np.array([sum(r["base"]["steps"]) for r in runs]) / 1e6
    zs = np.array([r["zero_shot"] for r in runs])
    print(f"\n{dom}: {seeds} seeds x {n} tasks; base {base.mean():.1f}M steps ({base.min():.1f}-{base.max():.1f}); "
          f"zero-shot over the learned extremes: mean relative regret {zs.mean():.1e}, max {zs.max():.1e}, within "
          f"1e-3 {np.mean(zs <= 1e-3):.0%}")
    for mode in ("fresh", "stream"):
        for how in ("cog", "ql"):
            rows = [x for r in runs for x in (list(r["lazy"][(mode, how)].values()) if mode == "fresh"
                                              else r["lazy"][(mode, how)])]
            if not rows:
                continue
            g = [len(x["grows"]) for x in rows]
            kinds = [k["kind"] for x in rows for k in x["grows"]]
            steps = np.array([x["steps"] for x in rows]) / 1e6
            grow_steps = np.array([k["steps"] for x in rows for k in x["grows"]]) / 1e6
            reg = np.array([x["regret"] for x in rows])
            half = [x for r in runs for x in (r["lazy"][(mode, how)][n // 2:] if mode == "stream" else [])]
            print(f"  lazy-{how:3s} {mode:6s}: regret mean {reg.mean():.1e} max {reg.max():.1e} (within 1e-3 "
                  f"{np.mean(reg <= 1e-3):.0%}); per task: checks {np.mean([x['checks'] for x in rows]):.1f} "
                  f"(certified {np.mean([x['passed'] for x in rows]):.1f}), new policies {np.mean(g):.2f}, rounds "
                  f"{np.mean([x['rounds'] for x in rows]):.2f}, steps mean {steps.mean():.2f}M median "
                  f"{np.median(steps):.2f}M; zero-shot tasks {np.mean(np.array(g) == 0):.0%}"
                  + (f" (second half {np.mean([len(x['grows']) == 0 for x in half]):.0%})" if half else "")
                  + f"; per new policy {np.median(grow_steps) if len(grow_steps) else 0:.2f}M (median); kinds "
                  f"{dict((k, kinds.count(k)) for k in sorted(set(kinds)))}; certified {np.mean([x['certified'] for x in rows]):.0%}")
    fl = [r["flat"][ti] for r in runs for ti in range(n) if ti in r["flat"]]
    if fl:
        st = np.array([f["steps"] for f in fl]) / 1e6
        print(f"  flat Q        : per task steps mean {st.mean():.2f}M median {np.median(st):.2f}M, converged "
              f"{np.mean([f['converged'] for f in fl]):.0%}, final regret mean {np.mean([f['regret'] for f in fl]):.1e}")
    sf = [r["sfols"] for r in runs if r["sfols"]]
    if sf:
        fin = np.array([s["trace"][-1][2] for s in sf])
        print(f"  sfols         : {np.mean([s['queries'] for s in sf]):.0f} queries, {np.mean([s['steps'] for s in sf]) / 1e6:.1f}M "
              f"steps after the base; final regret mean {fin.mean():.1e} max {fin.max():.1e}, within 1e-3 {np.mean(fin <= 1e-3):.0%}")
        for s_ in sf:
            m = np.array([np.mean(t[2]) for t in s_["trace"]])
            i = int(m.argmin())
            print(f"      mean regret per query: start {m[0]:.1e}, best {m[i]:.1e} at query {i} "
                  f"({s_['trace'][i][1] / 1e6:.1f}M steps), final {m[-1]:.1e} at query {len(m) - 1}")
        for thr in (1e-2, 1e-3):
            first = []
            for s in sf:
                hit = [t[1] for t in s["trace"] if max(t[2]) <= thr]
                first.append(hit[0] / 1e6 if hit else np.inf)
            print(f"      steps after the base until every task is within {thr:g}: {first}")


if __name__ == "__main__":
    main()
