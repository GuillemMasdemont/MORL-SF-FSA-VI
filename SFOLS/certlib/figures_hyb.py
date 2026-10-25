"""Figure for notes/gpi_basis.tex: % of held-out weights where GPI, the metapolicy and the hybrid policy are optimal,
against the size of the base (first k SFOLS policies). Data: results/certlib/hybcurve_<domain>.pkl.

usage: PYTHONPATH=SFOLS python -m certlib.figures_hyb            the curves (hyb_optimal.pdf)
       PYTHONPATH=SFOLS python -m certlib.figures_hyb example    loss maps of a hedging windy task (hyb_example_windy.pdf)
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "certlib")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "notes", "figures", "hyb_optimal.pdf")
PANELS = [("ds", "Double slit ($\\gamma=0.9$)"), ("office1f2", "Office, slip 0.1"),
          ("office3f2", "Office, slip 0.3"), ("windy", "Windy two-stage")]
SERIES = [("gpi", "GPI", "#eb6834", "--", "s"), ("meta", "metapolicy", "#1baf7a", ":", "^"),
          ("hyb", "hybrid $\\pi^{\\mathrm{hyb}}$", "#2a78d6", "-", "o")]


def main():
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "xtick.color": "#444444", "ytick.color": "#444444"})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.85), sharey=True)
    for ax, (dom, title) in zip(axes, PANELS):
        with open(os.path.join(OUT, f"hybcurve_{dom}.pkl"), "rb") as fh:
            rows = pickle.load(fh)
        k = [r["k"] for r in rows]
        for key, lab, col, ls, mk in SERIES:
            ax.plot(k, [r[f"opt_{key}"] for r in rows], color=col, ls=ls, marker=mk, ms=3.5, lw=1.4, label=lab)
        ax.set_title(title, fontsize=8, color="#222222")
        ax.set_xlabel("base size $k$", color="#444444")
        ax.set_ylim(30, 101)
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
    axes[0].set_ylabel("% weights optimal", color="#444444")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG)
    print("saved", FIG)


def example(task_index=7):
    """Windy two-stage task (A -> u1, B -> u2, both with valuable corridors): per-state loss at u0 of GPI, the
    metapolicy and pi_hyb over the 6 extremes, and where pi_hyb takes an action that no base policy takes."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from gpi_basis.fsa import _order, symbols
    from .experiments import get, library_from, load, tasks_for
    from .hybrid import Cost, evaluate, gpi, meta_hyb
    mdp, _ = get("windy")
    task = tasks_for("windy", mdp, np.random.default_rng(1))[task_index]
    lib = library_from(mdp, load("run", "windy", "sfols", "1e-5", 0.0), 6)
    mask = np.zeros((mdp.S, mdp.A), bool)
    for a in lib.acts:
        mask[mdp.R, a] = True
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    Vu = {}
    for u in _order(task):                                     # exit weights from the optimal downstream values
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else Vu[v][ex[e]]
        act, psi = mdp.solve(w)
        Vu[u] = psi[mdp.R, act] @ w
    Vs = Vu["u0"]
    c = Cost(mdp.A)
    gact, Vh = gpi(lib.stack, w)
    mh = meta_hyb(mdp, lib.stack, lib.acts, mask, w, c)
    vals = {"GPI": evaluate(mdp, gact, w, Vh, c), "metapolicy": mh["meta"][1], r"$\pi^{\mathrm{hyb}}$": mh["hyb"][1]}
    off = ~mask[mdp.R, mh["hyb"][0]]
    H, W = mdp.grid.shape
    s0 = mdp.relevant[0]
    print(f"task {task_index}: {task}; w(u0) = {np.round(w, 3)}; V*(s0) = {Vs[s0]:.4f}")
    for k, V in vals.items():
        print(f"  {k:24s} loss at s0 {100 * (Vs[s0] - V[s0]) / Vs[s0]:.2f}%, max {100 * ((Vs - V) / np.maximum(Vs, 1e-12)).max():.2f}%")
    names = ["UP", "RIGHT", "DOWN", "LEFT"]
    print(f"  pi_hyb takes an action outside B(s) at {off.sum()} states:",
          {names[a]: int(((mh['hyb'][0] == a) & off).sum()) for a in range(mdp.A)},
          "| base actions there:", {names[a]: int(mask[off, a].sum()) for a in range(mdp.A)})
    plt.rcParams.update({"font.size": 8})
    cmap = LinearSegmentedColormap.from_list("loss", ["#ffffff", "#f6c39f", "#eb6834", "#8a3511"])
    rel = {k: 100 * (Vs - V) / np.maximum(Vs, 1e-12) for k, V in vals.items()}
    vmax = max(r.max() for r in rel.values())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for ax, (k, r) in zip(axes, rel.items()):
        img = np.full((H, W), np.nan)
        wall = np.ones((H, W))
        for i, (rr, cc) in enumerate(mdp.coords):
            img[rr, cc] = max(r[i], 0.0)
            wall[rr, cc] = 0
        ax.imshow(np.where(wall == 1, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("w", ["#4d4d4d", "#4d4d4d"]),
                  interpolation="nearest")
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        if k.startswith("$"):
            pts = np.array([mdp.coords[i] for i in np.flatnonzero(off)])
            ax.plot(pts[:, 1], pts[:, 0], "o", ms=1.6, color="#2a78d6", label="action outside $B(s)$")
            ax.legend(loc="lower left", fontsize=6.5, frameon=False, handletextpad=0.2, borderpad=0.1,
                      bbox_to_anchor=(0.0, -0.2))
        for name in ("A", "B"):
            rr, cc = mdp.coords[ex[name]]
            ax.text(cc - 1.6, rr + (2.2 if name == "A" else -1.0), name, fontsize=7, color="#222222")
        rr, cc = mdp.coords[s0]
        ax.plot(cc, rr, "o", ms=3, color="black")
        ax.set_title(f"{k}: loss at $s_0$ {(Vs[s0] - vals[k][s0]) / Vs[s0] * 100:.2f}%", fontsize=8, color="#222222")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(r"$100\,(V^*-V^\pi)/V^*$ (%)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    out = os.path.join(os.path.dirname(FIG), "hyb_example_windy.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    import sys
    example() if len(sys.argv) > 1 and sys.argv[1] == "example" else main()
