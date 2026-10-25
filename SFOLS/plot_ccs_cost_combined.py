"""Combined 2x3 figure of the CCS sample cost: Deep Sea Treasure (top row) and Office (bottom row),
each row being the domain map followed by the two cost panels.

Reads the per-seed cumulative curves saved by the figure cells of 13_ccs_speedup_dst.ipynb and
14_ccs_speedup_office.ipynb (notes/figures/data/*_q_ccs_cost.npz) -- no experiment is rerun; the maps
come straight from the environments -- and writes notes/figures/q_ccs_cost.pdf and
paper/figures/q_ccs_cost.pdf.

Run from SFOLS/ with the okb env:  /opt/anaconda3/envs/okb/bin/python plot_ccs_cost_combined.py
"""
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
SF_FSA_VI_ROOT = os.path.join(HERE, "..", "RM", "sf-fsa-vi")
DATA_DIR = os.path.join(HERE, "..", "notes", "figures", "data")
TEXTWIDTH = 4.98      # inches: \textwidth of the paper (11pt article, 360pt), so it is placed at 100%
OUTS = [os.path.join(HERE, "..", "notes", "figures", "q_ccs_cost.pdf"),
        os.path.join(HERE, "..", "paper", "figures", "q_ccs_cost.pdf")]

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"   # MacTeX, if PATH lacks it
USE_TEX = shutil.which("latex") is not None

PUB_RC = {
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}
# same styles as the per-domain figures in notebooks 13 and 14
ARM_STYLE = {
    "standard":      dict(color="#2a78d6", ls="-",  marker="o"),
    "standard-warm": dict(color="#eb6834", ls="--", marker="s"),
    "certify-pool":  dict(color="#1baf7a", ls="-.", marker="^"),
    "certify-mf":    dict(color="#e87ba4", ls=(0, (5, 1.5, 1, 1.5, 1, 1.5)), marker="v"),
}
ARM_LABEL = {"standard": "SFOLS", "standard-warm": "SFOLS + warm start",
             "certify-pool": "Certify-or-Grow + warm start (ours)",
             "certify-mf": "Certify-or-Grow + warm start, model-free (ours)"}
KEYS = ["log", "log_total"]
OFFICE_LOG = False         # y scale of the Office row (the DST row is linear)
OFFICE_MAX_QUERY = 20      # cut the Office curves at this OLS query (None: all of them)
YLABEL = {"log": "Steps to converge", "log_total": "Environment steps"}


def fmt(y):
    return f"{y:.0f}k" if y < 1000 else f"{y / 1000:.2f}M"


def label_ends(ax, ends, log, x_off, min_gap=0.075):
    # value labels at the line ends, nudged apart so they never overlap; the gap is a fraction of the
    # axis height, so it works on linear and log axes alike (call after the limits and scale are set)
    to_ax = (lambda y: np.log10(y)) if log else (lambda y: y)
    lo, hi = (to_ax(v) for v in ax.get_ylim())
    gap = min_gap * (hi - lo)
    placed = []
    for (x, y) in sorted(ends.values(), key=lambda xy: xy[1]):
        y_ax = max(to_ax(y), placed[-1] + gap) if placed else to_ax(y)
        placed.append(y_ax)
        ax.annotate(fmt(y), xy=(x, y), xytext=(x + x_off, 10 ** y_ax if log else y_ax),
                    va="center", ha="left", fontsize=6.5, color="#333333")


# ---- domain maps ----------------------------------------------------------------------------------
WALL_C, FLOOR_C, GRID_C = "#4d4d4d", "#f2f2f2", "#d0d0d0"     # as in the double slit figure
OBJ_C = {"C": "#a0703f", "O": "#8e6cc9", "M": "#3b8fb0"}     # coffee, office, mail


def draw_grid(ax, colors):
    # colors: (H, W) array of colour strings, one per cell
    H, W = colors.shape
    names = sorted(set(colors.ravel()))
    idx = np.vectorize(names.index)(colors)
    ax.imshow(idx, cmap=ListedColormap(names), vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
    for x in np.arange(-0.5, W, 1.0):
        ax.axvline(x, color=GRID_C, lw=0.3)
    for y in np.arange(-0.5, H, 1.0):
        ax.axhline(y, color=GRID_C, lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def start_marker(ax, r, c, label=r"$s_0$", dx=0.0, dy=0.0):
    ax.plot(c, r, marker="o", ms=4, color="black")
    if label:
        ax.annotate(label, (c, r), xytext=(c + 0.55 + dx, r + 0.1 + dy), fontsize=7, va="center")


def plot_dst_map(ax):
    import mo_gymnasium as mo_gym
    sea = mo_gym.make("deep-sea-treasure-v0").unwrapped.sea_map
    colors = np.where(sea == -10, WALL_C, FLOOR_C).astype(object)
    # treasures: light to dark green with their value (0.7 ... 23.7; values given in the caption)
    shades = plt.get_cmap("Greens")(np.linspace(0.35, 0.95, 256))
    vmax = sea.max()
    for (r, c), v in np.ndenumerate(sea):
        if v > 0:
            rgb = shades[int(255 * v / vmax)][:3]
            colors[r, c] = "#%02x%02x%02x" % tuple(int(255 * x) for x in rgb)
    draw_grid(ax, colors)
    start_marker(ax, 0, 0)


def plot_office_map(ax):
    if SF_FSA_VI_ROOT not in sys.path:
        sys.path.insert(0, SF_FSA_VI_ROOT)
    from envs.grid_envs import Office
    office = Office(add_obj_to_start=True, random_act_prob=0.0)
    M = office.MAP
    colors = np.full(M.shape, FLOOR_C, dtype=object)
    colors[M == "X"] = WALL_C
    for (r, c), v in np.ndenumerate(M):
        if v[0] in OBJ_C:
            colors[r, c] = OBJ_C[v[0]]
    draw_grid(ax, colors)
    for (r, c), v in np.ndenumerate(M):
        if v[0] in OBJ_C:
            ax.text(c, r, v[0], ha="center", va="center", fontsize=6.5, color="white")   # C / O / M
    r, c = next((r, c) for r, c in office.initial if M[r, c] == "_")
    start_marker(ax, r, c)


def plot_row(axes, data, log, tags, max_query=None):
    for ax, key, tag in zip(axes, KEYS, tags):
        ends, x_max = {}, 0
        for arm, st in ARM_STYLE.items():
            Y = data[f"{arm}__{key}"] / 1000
            if max_query is not None:
                Y = Y[:, :max_query + 1]
            x0 = 1 if log else 0                               # from query 1 on a log scale
            Y = Y[:, x0:]
            mean, std = Y.mean(axis=0), Y.std(axis=0)
            x = np.arange(len(mean)) + x0
            every = 3 if x[-1] < 30 else (5 if x[-1] <= 40 else 10)
            ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=0.18, lw=0)
            ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=1.4, marker=st["marker"], ms=2.6,
                    markevery=every, label=ARM_LABEL[arm])
            ends[arm] = (x[-1], mean[-1])
            x_max = max(x_max, x[-1])
        ax.set_xlim(0, x_max * 1.14)
        ax.set_xticks(np.arange(0, x_max + 1, every))
        if log:
            ax.set_yscale("log")
            ax.set_ylabel(YLABEL[key] + r" ($\times 10^3$, log)")
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel(YLABEL[key] + r" ($\times 10^3$)")
        ax.set_xlabel("OLS query")
        label_ends(ax, ends, log, x_off=0.025 * x_max)
        ax.text(-0.02, 1.03, rf"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)


def main():
    dst = np.load(os.path.join(DATA_DIR, "dst_q_ccs_cost.npz"))
    office = np.load(os.path.join(DATA_DIR, "office_q_ccs_cost.npz"))
    rows = ((plot_dst_map, "Deep Sea Treasure", dst, False, "abc", None),
            (plot_office_map, r"Office ($\varepsilon$-CCS)", office, OFFICE_LOG, "def", OFFICE_MAX_QUERY))
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(TEXTWIDTH, 4.5), layout="constrained")
        for sub, (plot_map, title, data, log, tags, max_query) in zip(fig.subfigures(2, 1, hspace=0.04), rows):
            sub.suptitle(title, fontsize=8.5)
            ax_map, *axes = sub.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.06})
            plot_map(ax_map)
            ax_map.set_anchor("N")          # map top-aligned with the plots, right under its letter
            ax_map.text(-0.02, 1.03, rf"({tags[0]})", transform=ax_map.transAxes, ha="right", va="bottom",
                        fontsize=8)
            plot_row(axes, data, log, tags[1:], max_query)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
                   handlelength=2.4, columnspacing=1.2)
        for out in OUTS:
            fig.savefig(out)
            print(f"saved {os.path.normpath(out)}  (TeX rendering: {USE_TEX})")


if __name__ == "__main__":
    main()
