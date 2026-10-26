"""Figure of the paper for the stochastic domains, the counterpart of q_ccs_cost.pdf (Figure 1): sample cost of building
the eps-CCS with learned SFs in the slippery Office (slip 0.05, eps-CCS with eps = 0.05, top) and the double slit (wind
+-1, gamma 0.99, eps = 1e-3, bottom). Each row: the map, the cumulative steps to converge and all environment steps per
OLS query, mean +- std over the seeds. Certify-or-Grow's basis learns are counted at queries 1..d.

The SFOLS arms come from E5 (results/q_ccs_10seeds/, sweeps office_ccs_mf.yaml and ds_ccs_mf.yaml); the CoG arms from
E5b (results/q_ccs_cog2/, sweeps office_ccs_cog2.yaml and ds_ccs_cog2.yaml: the basis answers the re-queried extremes
and the full learn after a failed certificate continues the metapolicy learn), or from the run set named by COG
(analyze_ccs_speed.SOURCE: E5 | cold | gpi | continue). Writes paper/figures/q_ccs_cost_stochastic.pdf and
notes/figures/q_ccs_cost_stochastic.pdf.
usage (from SFOLS/): [COG=continue] python plot_ccs_cost_stochastic.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from analyze_ccs_speed import runs_of
from plot_ccs_cost_combined import ARM_LABEL, ARM_STYLE, PUB_RC, TEXTWIDTH, USE_TEX, label_ends, plot_office_map
from plot_ds_q_cost import per_query, plot_map as plot_ds_map

HERE = os.path.dirname(os.path.abspath(__file__))
COG = os.environ.get("COG", "continue")
OUTS = [os.path.join(HERE, "..", "paper", "figures", "q_ccs_cost_stochastic.pdf"),
        os.path.join(HERE, "..", "notes", "figures", "q_ccs_cost_stochastic.pdf")]
ARMS = {"standard": "standard", "standard-warm": "standard-warm", "certify": "certify-pool", "certify-mf": "certify-mf"}
YLABEL = {"log": "Steps to converge", "log_total": "Environment steps"}


def load(dom):
    return [r for arm in ARMS for r in runs_of(dom, "E5" if arm.startswith("standard") else COG, arm)]


def plot_row(axes, runs, tags):
    for ax, key, tag in zip(axes, ("log", "log_total"), tags):
        ends, x_max = {}, 0
        for arm, style_key in ARMS.items():
            rs = [r for r in runs if r["arm"] == arm]
            curves = [np.concatenate([[0], np.cumsum(per_query(r, key))]) for r in rs]
            n = max(len(c) for c in curves)
            Y = np.stack([np.pad(c, (0, n - len(c)), mode="edge") for c in curves]) / 1000
            mean, std, x = Y.mean(0), Y.std(0), np.arange(n)
            st = ARM_STYLE[style_key]
            ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=0.18, lw=0)
            ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=1.4, marker=st["marker"], ms=2.6,
                    markevery=1 if n < 12 else max(1, n // 8), label=ARM_LABEL[style_key])
            ends[arm] = (x[-1], mean[-1])
            x_max = max(x_max, x[-1])
        ax.set_xlim(0, x_max * 1.18)
        ax.set_xticks(np.arange(0, x_max + 1, 1 if x_max < 12 else 25))
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:g}"))
        ax.set_ylabel(YLABEL[key] + r" ($\times 10^6$)")
        ax.set_xlabel("OLS query")
        label_ends(ax, ends, False, x_off=0.025 * x_max)
        ax.text(-0.02, 1.03, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)


def main():
    office, ds = load("office"), load("ds")
    rows = ((plot_office_map, r"Office, slip $0.05$ ($\varepsilon$-CCS)", office, "abc"),
            (plot_ds_map, r"Double slit, wind $\pm 1$", ds, "def"))
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 4.5), layout="constrained")
        for sub, (plot_map, title, runs, tags) in zip(fig.subfigures(2, 1, hspace=0.04), rows):
            sub.suptitle(title, fontsize=8.5)
            ax_map, *axes = sub.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.06})
            plot_map(ax_map)
            ax_map.set_anchor("N")
            ax_map.text(-0.02, 1.03, rf"({tags[0]})", transform=ax_map.transAxes, ha="right", va="bottom", fontsize=8)
            plot_row(axes, runs, tags[1:])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, handlelength=2.4,
                   columnspacing=1.2)
        for out in OUTS:
            fig.savefig(out)
            print(f"saved {os.path.normpath(out)}  (TeX rendering: {USE_TEX})")
    for name, runs in (("office", office), ("double slit", ds)):
        print(f"{name}: seeds per arm {dict((a, sum(r['arm'] == a for r in runs)) for a in ARMS)}")
        for arm in ARMS:
            rs = [r for r in runs if r["arm"] == arm]
            conv = np.array([r["n"]["env steps to converge"] for r in rs]) / 1e6
            tot = np.array([r["n"]["env steps"] for r in rs]) / 1e6
            ex = {k: np.mean([r["n"].get(k, 0) for r in rs]) for k in ("fallbacks", "certified", "false certificates")
                  if any(k in r["n"] for r in rs)}
            print(f"  {arm:14s} to converge {conv.mean():7.2f}M +- {conv.std():.2f} (median {np.median(conv):.2f}), all "
                  f"steps {tot.mean():7.2f}M +- {tot.std():.2f}, CCS {np.mean([len(r['ccs_psi0']) for r in rs]):.1f}, "
                  f"queries {np.mean([r['n_queries'] for r in rs]):.1f} {ex}")


if __name__ == "__main__":
    main()
