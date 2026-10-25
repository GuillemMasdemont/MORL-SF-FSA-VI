"""Double slit paper figure: (a) the domain with the optimal policy at w = (0.5, 0.5), (b) suboptimal held-out weights vs number of basis policies.

Reads notes/figures/data/double_slit.npz (saved by 16_double_slit_hyb.ipynb) and writes
notes/figures/double_slit.pdf and paper/figures/double_slit.pdf. No recomputation.

Run from SFOLS/ with the okb env:  /opt/anaconda3/envs/okb/bin/python plot_double_slit.py
"""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "notes", "figures", "data", "double_slit.npz")
TEXTWIDTH = 4.98      # inches: \textwidth of the paper (11pt article, 360pt), so it is placed at 100%
OUTS = [os.path.join(HERE, "..", "notes", "figures", "double_slit.pdf"),
        os.path.join(HERE, "..", "paper", "figures", "double_slit.pdf")]

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"   # MacTeX, if PATH lacks it
USE_TEX = shutil.which("latex") is not None
PUB_RC = {
    "text.usetex": USE_TEX, "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}
# domain colours (same as the previous double_slit_domain.pdf)
WALL, UP, RIGHT, DOWN, EXIT = 0, 1, 2, 3, 4
COLORS = ["#4d4d4d", "#2a78d6", "#e9e9e9", "#eb6834", "#1baf7a"]
CMAP = ListedColormap(COLORS)
# method styles (same as the previous double_slit_basis_growth.pdf)
POLICIES = ("GPI", "metapolicy", "hyb")
STYLE = {"GPI":        dict(color="#eb6834", ls="--", marker="s"),
         "metapolicy": dict(color="#1baf7a", ls="-",  marker="^"),
         "hyb":        dict(color="#2a78d6", ls="-.", marker="o")}
LEGEND = {"GPI": "GPI", "metapolicy": "Metapolicy",
          "hyb": r"$\pi^{\mathrm{hyb}}$"}


def action_grid(d, actions):
    g = np.full(tuple(d["map_shape"]), WALL)
    for i, (r, c) in enumerate(d["coords"]):
        g[r, c] = EXIT if d["term"][i] else 1 + int(actions[i])   # UP=0 -> 1, RIGHT=1 -> 2, DOWN=2 -> 3
    return g


def plot_domain(ax, d):
    grid = action_grid(d, d["actions_opt"])
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")

    r0, c0 = d["start"]
    ax.plot(c0, r0, marker="o", ms=4, color="black")
    ax.annotate(r"$s_0$", (c0, r0), xytext=(c0 + 1.2, r0 + 0.4), fontsize=7, va="center")
    for (r, c), name in zip(d["exits"], (r"$\mathbf e_1$", r"$\mathbf e_2$")):
        ax.annotate(name, (c, r), xytext=(c - 3.4, r + (1.8 if r == 0 else -1.2)), fontsize=7,
                    color="#11704f", va="center")
    ax.add_patch(FancyArrowPatch((6, 13.5), (6, 6.5), arrowstyle="<->", mutation_scale=8,
                                 color="#222222", lw=0.8))
    ax.annotate(rf"wind $\pm {int(d['max_wind'])}$", (6.6, 10), fontsize=6.5, va="center")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [Patch(color=COLORS[UP], label="up"), Patch(color=COLORS[RIGHT], label="right"),
               Patch(color=COLORS[DOWN], label="down"), Patch(color=COLORS[EXIT], label="exit"),
               Patch(color=COLORS[WALL], label="wall")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8, fontsize=6.5)


def plot_growth(ax, d):
    ks = d["ks"]
    for name in POLICIES:
        st = STYLE[name]
        ax.plot(ks, d[f"subopt_{name}"], color=st["color"], ls=st["ls"], lw=1.4, marker=st["marker"],
                ms=2.6, label=LEGEND[name])
    ax.axvline(2, color="#bbbbbb", lw=0.6, ls=":")          # k = 2: both corners
    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, ks[-1] + 0.5)
    ax.set_xticks(ks[::3] if len(ks) > 10 else ks)
    ax.set_xlabel(r"Basis policies $k$ (OLS order)")
    ax.set_ylabel(r"Suboptimal test weights (\%)" if USE_TEX else "Suboptimal test weights (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, handlelength=2.6)


def main():
    d = np.load(DATA)
    H, W = d["map_shape"]
    with plt.rc_context(PUB_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.15),
                                         gridspec_kw={"width_ratios": [W / H * 1.0, 1.25]})
        plot_domain(ax_a, d)
        plot_growth(ax_b, d)
        for ax, tag in ((ax_a, "a"), (ax_b, "b")):
            ax.text(-0.02, 1.03, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        fig.tight_layout(w_pad=2.0)
        for out in OUTS:
            fig.savefig(out)
            print(f"saved {os.path.normpath(out)}  (TeX rendering: {USE_TEX})")


if __name__ == "__main__":
    main()
