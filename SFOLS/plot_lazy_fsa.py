"""Figure of Lazy SF-FSA-VI (certlib/lazyfsa.py) for the paper: (a) max relative regret of zero-shot SF-FSA-VI over the
6 extremes and over the full eps-CCS (Lazy SF-FSA-VI is exact on every task); (b) policies stored by Lazy SF-FSA-VI as
tasks are solved, with the base carried over, against the size of the eps-CCS; (c) how the added policies were
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
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
KIND = [("metapolicy", "metapolicy", "#1baf7a"), ("hybrid", r"$\pi^{\mathrm{hyb}}$", "#2a78d6"),
        ("policy iteration", "policy iteration", "#c8c8c8")]
LINES = {"office1f2": dict(ls="-", marker="o"), "office3f2": dict(ls="--", marker="s"), "windy": dict(ls="-.", marker="^")}


def load(dom, n):
    with open(os.path.join(RES, f"lazyfsa_{dom}_{n}.pkl"), "rb") as fh:
        return pickle.load(fh)


def style(ax, tag):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(-0.02, 1.04, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


def main():
    data = {dom: load(dom, n) for dom, n, _ in DOMAINS}
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 2.15), layout="constrained")
        ax_r, ax_b, ax_k = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.0, 1.15, 1.25]})

        # (a) max relative regret of zero-shot SF-FSA-VI
        x = np.arange(len(DOMAINS))
        width = 0.36
        for off, key, lab, col in ((-width / 2, "reg_extremes", "6 extremes", "#b5b5b5"),
                                   (width / 2, "reg_ccs", r"$\epsilon$-CCS", "#555555")):
            vals = [max(max(r[key], 0.0) / r["opt"] for r in data[dom]["rows"]) for dom, _, _ in DOMAINS]
            for xi, v in zip(x + off, vals):
                if v > 1e-12:
                    ax_r.bar(xi, v, width * 0.92, color=col, label=lab if xi == x[0] + off else None)
                else:
                    ax_r.text(xi, 2e-8, "0", ha="center", va="bottom", fontsize=6.5, color=MUTED)
        ax_r.set_yscale("log")
        ax_r.set_ylim(1e-8, 1e-1)
        ax_r.set_xticks(x, [d[2].replace(", ", ",\n") for d in DOMAINS], fontsize=6.5)
        ax_r.set_ylabel("max relative regret")
        ax_r.text(0.5, 0.985, "Lazy SF-FSA-VI: 0 on all tasks", transform=ax_r.transAxes, ha="center", va="top",
                  fontsize=6.5, color=INK)
        ax_r.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=6.5, frameon=False, handlelength=1.0,
                    title="zero-shot over", title_fontsize=6.5, alignment="left")
        style(ax_r, "a")

        # (b) stored policies along the task sequence (base carried over), and the eps-CCS size
        for dom, n, lab in DOMAINS:
            rows = data[dom]["rows"]
            sizes = [6] + [r["cog_stream"]["size"] for r in rows]
            st = LINES[dom]
            ax_b.plot(np.arange(len(sizes)), sizes, color=INK, lw=1.1, ls=st["ls"], marker=st["marker"], ms=2.4,
                      markevery=max(1, len(sizes) // 6), label=lab)
            ax_b.annotate(f"{sizes[-1]}", (len(sizes) - 1, sizes[-1]), xytext=(2, 0), textcoords="offset points",
                          fontsize=6.5, va="center", color=INK)
            ccs = data[dom]["ccs_size"]
            ax_b.axhline(ccs, color="#9a9a9a", lw=0.7, ls=st["ls"])
            ax_b.text(61, ccs, f"{ccs}", fontsize=6.5, va="center", ha="left", color=MUTED)
        ax_b.set_yscale("log")
        ax_b.set_ylim(4, 900)
        ax_b.set_xlim(0, 60)
        ax_b.set_xlabel("tasks solved")
        ax_b.set_ylabel("policies stored")
        ax_b.text(30, 700, r"$\epsilon$-CCS (grey)", fontsize=6.5, ha="center", va="center", color=MUTED)
        ax_b.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=6.5, frameon=False, handlelength=2.2)
        style(ax_b, "b")

        # (c) how the added policies were certified, from the extremes on every task and with the base carried over
        pos, ticks, labels = 0.0, [], []
        for gi, (dom, n, lab) in enumerate(DOMAINS):
            for mode, short in (("cog_fresh", "fresh"), ("cog_stream", "stream")):
                kinds = [k for r in data[dom]["rows"] for k in r[mode]["grown"]]
                tot = len(kinds)
                bottom = 0.0
                for key, klab, col in KIND:
                    frac = 100 * kinds.count(key) / tot if tot else 0.0
                    ax_k.bar(pos, frac, 0.8, bottom=bottom, color=col, edgecolor="white", lw=0.6,
                             label=klab if (gi == 0 and short == "fresh") else None)
                    bottom += frac
                ax_k.text(pos, 101, f"{tot}", ha="center", va="bottom", fontsize=6, color=MUTED)
                ticks.append(pos)
                labels.append(short)
                pos += 1.0
            ax_k.text(pos - 1.5, -26, lab, ha="center", va="top", fontsize=6.5, color=INK)
            pos += 0.6
        ax_k.set_xticks(ticks, labels, fontsize=6)
        ax_k.set_ylim(0, 112)
        ax_k.set_ylabel(r"added policies (\%)" if USE_TEX else "added policies (%)")
        ax_k.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=6.5, frameon=False,
                    handlelength=1.0, columnspacing=0.8)
        style(ax_k, "c")

        for out in OUTS:
            fig.savefig(out)
            print(f"saved {os.path.normpath(out)}  (TeX rendering: {USE_TEX})")


if __name__ == "__main__":
    main()
