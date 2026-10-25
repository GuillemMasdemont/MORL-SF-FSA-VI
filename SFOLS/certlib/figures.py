"""Anytime-certificate figures for the certified-library note (notes/figures/certlib_*.pdf).

usage: PYTHONPATH=SFOLS python -m certlib.figures <domain> [eps_of_runs]
Left: certified gap Gamma against the number of TRAINED policies; right: against the library size.
The true worst-case loss of the certified library (if a curve file exists) is overlaid (thin line).
"""
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "certlib")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "notes", "figures")
STYLE = {"sfols": ("SFOLS (same queries with or without warm start)", "#eb6834", "--", "s"),
         "cog": ("Certify-or-Grow", "#1baf7a", ":", "^"),
         "cl": ("Certified library (this note)", "#2a78d6", "-", "o")}


def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def run_file(domain, lad):
    n = lad.count("_")
    fs = [f for f in glob.glob(os.path.join(OUT, f"run_{domain}_{lad}_*_0.0.pkl"))
          if "_".join(os.path.basename(f).split("_")[2:3 + n]) == lad
          and _isfloat(os.path.basename(f).split("_")[3 + n])]
    fs = sorted(fs, key=lambda f: float(os.path.basename(f).split("_")[3 + n]))
    return fs[0] if fs else None


def anytime(run):
    """(trained, library, gap) after each query; gap = largest gap left (non-increasing)."""
    log = run["log"]
    pts = [(e["learned"], e["library"], e["gap"]) for e in log if e["gap"] is not None]
    # the gap popped at query i is the certificate of the library built before it
    out, prev = [], None
    for i, e in enumerate(log):
        if e["gap"] is None:
            continue
        b = log[i - 1]
        out.append((b["learned"], b["library"], e["gap"]))
    out.append((log[-1]["learned"], log[-1]["library"], run["final_gap"]))
    a = np.array(out, float)
    a[:, 2] = np.minimum.accumulate(a[:, 2])
    return a


def main(domain):
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    has_true = False
    for lad in ("sfols", "cog", "cl"):
        f = run_file(domain, lad)
        if f is None:
            continue
        with open(f, "rb") as fh:
            run = pickle.load(fh)
        a = anytime(run)
        lab, col, ls, mk = STYLE[lad]
        for ax, x in ((axes[0], a[:, 0]), (axes[1], a[:, 1])):
            ax.step(x, np.maximum(a[:, 2], 1e-13), where="post", color=col, ls=ls, lw=1.8, label=lab)
            ax.plot(x[-1], max(a[-1, 2], 1e-13), marker=mk, color=col, ms=6)
        cf = os.path.join(OUT, f"curve_{domain}_{lad}_{os.path.basename(f).split('_')[3 + lad.count('_')]}_0.0.pkl")
        if os.path.exists(cf) and lad in ("cl", "sfols"):
            with open(cf, "rb") as fh:
                c = pickle.load(fh)
            lib = np.array([d["library"] for d in c]); lr = np.array([d["learned"] for d in c])
            tl = np.maximum([d["max_loss"] for d in c], 1e-13)
            has_true = True
            axes[1].plot(lib, tl, color=col, lw=0.8, alpha=0.9, label=f"{lab.split(' (')[0]}: true loss")
            axes[0].plot(lr, tl, color=col, lw=0.9, alpha=0.8)
    for ax, xl in ((axes[0], "trained policies (control problems)"), (axes[1], "library size (stored SFs)")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl)
        ax.grid(True, which="major", color="#dddddd", lw=0.5)
    axes[0].set_ylabel("worst-case GPI loss\n(thick: certified, thin: true)" if has_true
                       else "certified worst-case GPI loss")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = os.path.join(FIG, f"certlib_{domain}.pdf")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
