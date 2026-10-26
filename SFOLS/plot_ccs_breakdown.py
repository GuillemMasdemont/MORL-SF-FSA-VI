"""Where Certify-or-Grow's samples go in the stochastic CCS construction (Figure 2 of the paper), before (E5) and after
the two fixes of E5b: reusing the basis for OLS's re-queried extremes (QREUSE=1) and warm-starting the full learn after
a failed certificate (QFALLBACK=gpi | continue). One stacked bar per arm: mean over the seeds of the steps to converge
and of all environment steps, split by kind of learn (analyze_ccs_speed.py); the whisker is the std of the total.
Writes notes/figures/ccs_cog_breakdown.pdf.

usage (from SFOLS/): python plot_ccs_breakdown.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from analyze_ccs_speed import KINDS, breakdown
from plot_ccs_cost_combined import PUB_RC, TEXTWIDTH, USE_TEX

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notes", "figures", "ccs_cog_breakdown.pdf")
ROWS = [("SFOLS + warm start", "E5", "standard-warm"),
        ("CoG, as in E5", "E5", "certify"),
        ("CoG, reuse extremes", "cold", "certify"),
        ("CoG, reuse + GPI fallback", "gpi", "certify"),
        ("CoG, reuse + continued fallback", "continue", "certify"),
        ("CoG model-free, as in E5", "E5", "certify-mf"),
        ("CoG model-free, reuse + continued", "continue", "certify-mf")]
KIND_STYLE = {"extremes": ("extremes", "#2a78d6"),
              "learns": ("other queries (SFOLS: full learn; CoG: certified metapolicy)", "#eb6834"),
              "fallback": ("full learn after a failed certificate", "#1baf7a"),
              "failed": ("failed metapolicy learn", "#eda100"),
              "re-query": ("re-queried extremes", "#e87ba4")}
DOMAINS = [("office", r"Office, slip $0.05$"), ("ds", r"Double slit, wind $\pm 1$")]
METRICS = [("conv", "Steps to converge"), ("steps", "Environment steps")]
INK, MUTED, SURFACE = "#1f1f1f", "#6b6b6b", "#ffffff"


def panel(ax, dom, metric, show_labels):
    y = np.arange(len(ROWS))[::-1]
    for yi, (label, source, arm) in zip(y, ROWS):
        parts, tot = breakdown(dom, source, arm, metric)
        if parts is None:
            continue
        left = 0.0
        for k in KINDS:
            x = parts[k][1] / 1e6
            if x > 0:
                ax.barh(yi, x, left=left, height=0.62, color=KIND_STYLE[k][1], edgecolor=SURFACE, lw=0.6)
                left += x
        ax.errorbar(tot.mean() / 1e6, yi, xerr=tot.std() / 1e6, color=INK, lw=0.6, capsize=1.5, capthick=0.6)
        ax.text(tot.mean() / 1e6 + tot.std() / 1e6 + 0.02 * ax.get_xlim()[1], yi, f"{tot.mean() / 1e6:.1f}M",
                va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.tick_params(axis="y", length=0, labelleft=show_labels)
    ax.axhline(y[0] - 0.5, color=MUTED, lw=0.4, ls=":")          # SFOLS above, CoG below
    ax.axhline(y[4] - 0.5, color=MUTED, lw=0.4, ls=":")          # model certificate above, model-free below
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)


def main():
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 4.0), layout="constrained")
        for i, (sub, (dom, title)) in enumerate(zip(fig.subfigures(2, 1, hspace=0.03), DOMAINS)):
            sub.suptitle(title, fontsize=8)
            for j, (ax, (metric, name)) in enumerate(zip(sub.subplots(1, 2, gridspec_kw={"wspace": 0.04}), METRICS)):
                xmax = max((breakdown(dom, s, a, metric)[1] for _, s, a in ROWS if breakdown(dom, s, a, metric)[1]
                            is not None), key=lambda t: t.mean() + t.std())
                ax.set_xlim(0, (xmax.mean() + xmax.std()) / 1e6 * 1.22)
                panel(ax, dom, metric, j == 0)
                ax.set_title(name, fontsize=7.5)
                if i == 1:
                    ax.set_xlabel(r"Steps ($\times 10^6$)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_STYLE[k][1]) for k in KINDS]
        fig.legend(handles, [KIND_STYLE[k][0] for k in KINDS], loc="outside upper center", ncol=2, frameon=False,
                   fontsize=6.5, handlelength=1.2, columnspacing=1.0)
        fig.savefig(OUT)
        print(f"saved {os.path.normpath(OUT)}  (TeX rendering: {USE_TEX})")


if __name__ == "__main__":
    main()
