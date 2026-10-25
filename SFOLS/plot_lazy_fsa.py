"""Figure of Lazy SF-FSA-VI (certlib/lazyfsa.py) for the paper: (a) max relative regret of zero-shot SF-FSA-VI over the
6 extremes and over the full eps-CCS (Lazy SF-FSA-VI is exact on every task); (b) policies stored by Lazy SF-FSA-VI as
tasks are solved, with the policies carried over, against the size of the eps-CCS; (c) how the added policies were
certified: as the metapolicy, as pi_hyb, or after policy iteration.

Reads results/certlib/lazyfsa_<domain>_<n>.pkl; writes paper/figures/lazy_fsa.pdf and notes/figures/lazy_fsa.pdf.
usage (okb env, from SFOLS/): python plot_lazy_fsa.py
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from plot_ccs_cost_combined import PUB_RC, TEXTWIDTH, USE_TEX

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "certlib")
OUTS = [os.path.join(HERE, "..", "paper", "figures", "lazy_fsa.pdf"),
        os.path.join(HERE, "..", "notes", "figures", "lazy_fsa.pdf")]
DOMAINS = [("office1f2", 30, r"Office, $p=0.1$"), ("office3f2", 30, r"Office, $p=0.3$"), ("windy", 60, "Windy")]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
KIND = [("metapolicy", "metapolicy", "#1baf7a"), ("hybrid", r"$\pi^{\mathrm{hyb}}$", "#2a78d6"),
        ("policy iteration", "policy iteration", "#c8c8c8")]
LINES = {"office1f2": dict(ls="-", marker="o"), "office3f2": dict(ls="--", marker="s"), "windy": dict(ls=":", marker="^")}
PCT = r"\%" if USE_TEX else "%"


def load(dom, n):
    with open(os.path.join(RES, f"lazyfsa_{dom}_{n}.pkl"), "rb") as fh:
        return pickle.load(fh)


def style(ax, tag, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.02, 1.03, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def main():
    data = {dom: load(dom, n) for dom, n, _ in DOMAINS}
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 3.55), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.35, 1.0], hspace=0.06)
        ax_r, ax_b = top.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})
        ax_k = bottom.subplots(1, 1)

        # (a) max relative regret of zero-shot SF-FSA-VI, one row per domain
        y = np.arange(len(DOMAINS))[::-1]
        for key, lab, mk, face in (("reg_extremes", "zero-shot, 6 extremes", "o", "white"),
                                   ("reg_ccs", r"zero-shot, $\epsilon$-CCS", "s", INK)):
            vals = np.array([max(max(r[key], 0.0) / r["opt"] for r in data[dom]["rows"]) for dom, _, _ in DOMAINS])
            pos = vals > 1e-12
            ax_r.plot(vals[pos], y[pos], ls="none", marker=mk, ms=4.5, mfc=face, mec=INK, mew=0.9, label=lab)
            for yi in y[~pos]:
                ax_r.annotate("0", (1.5e-8, yi), fontsize=6.5, color=MUTED, ha="left", va="center")
        ax_r.set_xscale("log")
        ax_r.set_xlim(1e-8, 1e-1)
        ax_r.set_ylim(-0.6, len(DOMAINS) + 0.45)
        ax_r.set_yticks(y, [d[2] for d in DOMAINS])
        ax_r.set_xlabel("max relative regret")
        ax_r.legend(loc="upper left", fontsize=6.5, frameon=False, handletextpad=0.3, borderaxespad=0.2)
        style(ax_r, "a", grid_axis="x")

        # (b) policies stored along the task sequence (policies carried over), against the size of the eps-CCS
        ends = {"office1f2": -2.2, "office3f2": 2.2, "windy": 0.0}                     # label offsets (policies)
        for dom, n, lab in DOMAINS:
            rows = data[dom]["rows"]
            sizes = [6] + [r["cog_stream"]["size"] for r in rows]
            st = LINES[dom]
            ax_b.plot(np.arange(len(sizes)), sizes, color=INK, lw=1.1, ls=st["ls"], marker=st["marker"], ms=2.5,
                      markevery=max(1, len(sizes) // 5), label=lab)
            ax_b.annotate(f"{sizes[-1]}", (len(sizes) - 1, sizes[-1]),
                          xytext=(len(sizes) - 1 + 1.2, sizes[-1] + ends[dom]), fontsize=6.5, va="center", ha="left",
                          color=INK)
        ccs = [data[dom]["ccs_size"] for dom, _, _ in DOMAINS]
        ax_b.text(0.02, 0.98, rf"size of the $\epsilon$-CCS: {ccs[0]}, {ccs[1]} and {ccs[2]}", transform=ax_b.transAxes,
                  fontsize=6.5, ha="left", va="top", color=MUTED)
        ax_b.set_ylim(0, 40)
        ax_b.set_xlim(0, 64)
        ax_b.set_xlabel("tasks solved")
        ax_b.set_ylabel("policies stored")
        ax_b.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=2.4, borderaxespad=0.2)
        style(ax_b, "b")

        # (c) how the added policies were certified, per task from the extremes or with the policies carried over
        rows_k, ylab = [], []
        for dom, n, lab in DOMAINS:
            for mode, short in (("cog_fresh", "from the extremes"), ("cog_stream", "carried over")):
                kinds = [k for r in data[dom]["rows"] for k in r[mode]["grown"]]
                rows_k.append(kinds)
                ylab.append(f"{lab}, {short}")
        yk = np.arange(len(rows_k))[::-1]
        for yi, kinds in zip(yk, rows_k):
            left = 0.0
            for key, klab, col in KIND:
                frac = 100 * kinds.count(key) / len(kinds)
                ax_k.barh(yi, frac, 0.72, left=left, color=col, edgecolor="white", lw=0.8,
                          label=klab if yi == yk[0] else None)
                left += frac
            ax_k.text(101, yi, f"{len(kinds)}", fontsize=6.5, va="center", ha="left", color=MUTED)
        ax_k.text(101, yk[0] + 0.85, "added", fontsize=6.5, va="center", ha="left", color=MUTED)
        ax_k.set_yticks(yk, ylab, fontsize=6.5)
        ax_k.set_xlim(0, 100)
        ax_k.set_xlabel(f"added policies ({PCT})")
        ax_k.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.5, frameon=False,
                    handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)
        style(ax_k, "c", grid_axis="x")

        for out in OUTS:
            fig.savefig(out)
            print(f"saved {os.path.normpath(out)}  (TeX rendering: {USE_TEX})")


if __name__ == "__main__":
    main()
