"""Figure 1 of the paper on the double slit (wind +-1): sample cost of building the eps-CCS with learned SFs, model-free.
Reads results/ds_q_runs_<tag>.pkl written by q_ccs_stochastic.py (QDOMAIN=ds) and writes
notes/figures/ds_q_ccs_cost.pdf: the domain map and the cumulative steps to converge / environment steps per OLS
query, mean +- std over seeds. Certify-or-Grow's basis learns are counted at queries 1..d (the extremes they are for).

usage (okb env, from SFOLS/): python plot_ds_q_cost.py [office] [tag] [cog_tag]
office: plot results/office_q_runs_<tag>.pkl (stochastic Office, default tag slip0.05) with the Office map.
cog_tag (optional): take the two Certify-or-Grow arms from this run instead (e.g. the one with QBASISWARM=1).
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot_ccs_cost_combined import (ARM_STYLE, PUB_RC, SF_FSA_VI_ROOT, TEXTWIDTH, USE_TEX, WALL_C, FLOOR_C,
                                    draw_grid, label_ends, plot_office_map, start_marker)

HERE = os.path.dirname(os.path.abspath(__file__))
ARGV = sys.argv[1:]
DOM = ARGV.pop(0) if ARGV and ARGV[0] in ("ds", "office") else "ds"
TAG = ARGV[0] if ARGV else ("slip0.05" if DOM == "office" else "wind1_g0.99_eps0.001_cert0.01_stab0")
COG_TAG = ARGV[1] if len(ARGV) > 1 else None
OUT = os.path.join(HERE, "..", "notes", "figures", f"{DOM}_q_ccs_cost.pdf" if DOM == "ds" else "office_slip_q_ccs_cost.pdf")
ARMS = {"standard": "standard", "standard-warm": "standard-warm", "certify": "certify-pool", "certify-mf": "certify-mf"}
LABEL = {"standard": "SFOLS", "standard-warm": "SFOLS + warm start", "certify": "Certify-or-Grow + warm start",
         "certify-mf": "Certify-or-Grow + warm start, model-free"}
YLABEL = {"log": "Steps to converge", "log_total": "Environment steps"}
D = 2 if DOM == "ds" else 6


def per_query(r, key):
    """Cost per OLS query; the i-th learn logged at query 0 (Certify-or-Grow's basis extreme e_i) goes to query i."""
    c, i = np.zeros(r["n_queries"] + 1), 0
    for q, x in r[key]:
        if q == 0:
            i += 1
            c[i] += x
        else:
            c[q] += x
    return c[1:]


def plot_map(ax):
    sys.path.insert(0, SF_FSA_VI_ROOT)
    from envs.grid_envs import DoubleSlit
    M = DoubleSlit.MAP
    colors = np.full(M.shape, FLOOR_C, dtype=object)
    colors[M == "X"] = WALL_C
    colors[M == "O1"], colors[M == "O2"] = "#8e6cc9", "#3b8fb0"
    draw_grid(ax, colors)
    for (r, c), v in np.ndenumerate(M):
        if v in ("O1", "O2"):
            ax.annotate(rf"$O_{v[1]}$", (c, r), xytext=(c - 4.5, r + (2.0 if v == "O1" else -1.4)), fontsize=7)
    r, c = map(int, np.argwhere(M == "_")[0])
    start_marker(ax, r, c, dx=0.6)


def main():
    load = lambda t: [r for r in pickle.load(open(os.path.join(HERE, "results", f"{DOM}_q_runs_{t}.pkl"), "rb"))
                      if "error" not in r]
    runs = load(TAG)
    if COG_TAG:
        runs = [r for r in runs if not r["arm"].startswith("certify")] + load(COG_TAG)
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 2.3), layout="constrained")
        ax_map, *axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.06})
        (plot_map if DOM == "ds" else plot_office_map)(ax_map)
        ax_map.set_anchor("N")
        for ax, key, tag in zip(axes, ("log", "log_total"), "bc"):
            ends, x_max = {}, 0
            for arm in ARMS:
                rs = [r for r in runs if r["arm"] == arm]
                if not rs:
                    continue
                curves = [np.concatenate([[0], np.cumsum(per_query(r, key))]) for r in rs]
                n = max(len(c) for c in curves)
                Y = np.stack([np.pad(c, (0, n - len(c)), mode="edge") for c in curves]) / 1000
                mean, std, x = Y.mean(0), Y.std(0), np.arange(n)
                st = ARM_STYLE[ARMS[arm]]
                ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=0.18, lw=0)
                ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=1.4, marker=st["marker"], ms=2.6, label=LABEL[arm],
                        markevery=1 if n < 12 else max(1, n // 10))
                ends[arm] = (x[-1], mean[-1])
                x_max = max(x_max, x[-1])
            ax.set_xlim(0, x_max * 1.18)
            ax.set_xticks(np.arange(0, x_max + 1, 1 if x_max < 12 else 25))
            ax.set_ylim(bottom=0)
            if DOM == "office":
                from matplotlib.ticker import FuncFormatter
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:g}"))
                ax.set_ylabel(YLABEL[key] + r" ($\times 10^6$)")
            else:
                ax.set_ylabel(YLABEL[key] + r" ($\times 10^3$)")
            ax.set_xlabel("OLS query")
            label_ends(ax, ends, False, x_off=0.025 * x_max)
            ax.text(-0.02, 1.03, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(axis="y", color="#e6e6e6", lw=0.5)
            ax.set_axisbelow(True)
        ax_map.text(-0.02, 1.03, "(a)", transform=ax_map.transAxes, ha="right", va="bottom", fontsize=8)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, handlelength=2.4, columnspacing=1.2)
        fig.savefig(OUT)
    print(f"saved {os.path.normpath(OUT)}  (TeX rendering: {USE_TEX})")
    print(f"{'arm':42s} {'queries':>7s} {'CCS':>4s} {'to converge':>12s} {'all steps':>11s}  counters (median over seeds)")
    for arm in ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        med = lambda f: np.median([f(r) for r in rs])
        keys = sorted({k for r in rs for k in r["n"] if "steps" not in k})
        print(f"{LABEL[arm]:42s} {med(lambda r: r['n_queries']):7.0f} {med(lambda r: len(r['ccs_psi0'])):4.0f} "
              f"{med(lambda r: r['n']['env steps to converge']) / 1e6:11.2f}M {med(lambda r: r['n']['env steps']) / 1e6:10.2f}M  "
              + ", ".join(f"{k} {med(lambda r: r['n'].get(k, 0)):.0f}" for k in keys))


if __name__ == "__main__":
    main()
