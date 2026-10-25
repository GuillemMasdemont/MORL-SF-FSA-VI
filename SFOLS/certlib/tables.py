"""Write notes/certlib_results.tex (tables) and the figures from the saved results.

usage: PYTHONPATH=SFOLS python -m certlib.tables
"""
import glob
import os
import pickle

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "certlib")
NOTES = os.path.join(os.path.dirname(__file__), "..", "..", "notes")


def load(name):
    f = os.path.join(OUT, name + ".pkl")
    if not os.path.exists(f):
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def sci(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 1) >= 10:
        e, m = e + 1, m / 10
    return f"${m:.1f}\\cdot10^{{{e}}}$" if abs(m - 1) > 0.05 else f"$10^{{{e}}}$"


def eps_label(e):
    return "exact" if e <= 1e-9 else sci(e)


def stops_table(domain, caption, tasks=False, eps_keep=None, label=None):
    rows = load(f"stops_{domain}")
    if rows is not None and eps_keep is not None:
        rows = [r for r in rows if not np.isfinite(r["eps"]) or any(abs(r["eps"] - e) < 1e-15 for e in eps_keep)]
    if rows is None:
        return f"% no stops for {domain}\n"
    cols = "llrrrrr" + ("r" if tasks else "")
    head = ("$\\epsilon$ & method & trained & library & sweeps & certified $\\Gamma$ & true max loss"
            + (" & max task regret" if tasks else "") + "\\\\")
    lab = f"\\label{{{label}}}" if label else ""
    lines = [f"\\begin{{table}}[h]\\centering\\small", f"\\caption{{{caption}}}{lab}", f"\\begin{{tabular}}{{{cols}}}",
             "\\toprule", head, "\\midrule"]
    eps_vals = sorted({r["eps"] for r in rows if np.isfinite(r["eps"])}, reverse=True)
    for e in eps_vals:
        for r in [r for r in rows if r["eps"] == e]:
            line = (f"{eps_label(e)} & {r['method']} & {r['learned']} & {r['library']} & {int(r['sweeps'])} & "
                    f"{sci(r['gap'])} & {sci(r['true_max'])}")
            if tasks:
                line += f" & {sci(r['task_max'])} ({100 * r['task_rel']:.2f}\\%)"
            lines.append(line + "\\\\")
        lines.append("\\midrule")
    for r in [r for r in rows if not np.isfinite(r["eps"])]:
        line = f"-- & {r['method']} & {r['learned']} & {r['library']} & -- & -- & {sci(r['true_max'])}"
        if tasks:
            line += f" & {sci(r['task_max'])} ({100 * r['task_rel']:.2f}\\%)"
        lines.append(line + "\\\\")
    lines[-1] = lines[-1] if not lines[-1] == "\\midrule" else ""
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(l for l in lines if l is not None)


def sufficiency_table():
    rows = [load(f"test_ds_1e-10_{k}") for k in (2, 4, 6, 8, 11)]
    rows = [r for r in rows if r is not None]
    if not rows:
        return ""
    lines = ["\\begin{table}[h]\\centering\\small",
             "\\caption{Sufficiency test on double slit: the base is the first $k$ trained policies of the certified-library "
             "run; afterwards only compositions (\\textsc{free}, \\textsc{compose}, \\textsc{meta}), no training.}\\label{tab:suff}",
             "\\begin{tabular}{rrrrrr}", "\\toprule",
             "trained base & open corners & library & sweeps & certified $\\Gamma$ & true max loss\\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['n_learned']} & {r['open']} & {r['library']} & {r['sweeps']} & {sci(r['cert'])} & {sci(r['true_max'])}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def sampled_table():
    rows = [load(f"sampled_ds_{n}_1e-4_0") for n in (10, 100, 1000, 10000)]
    rows = [r for r in rows if r is not None]
    if not rows:
        return ""
    lines = ["\\begin{table}[h]\\centering\\small",
             "\\caption{Sample-based construction on double slit ($\\epsilon=10^{-4}$): $n$ sampled outcomes per $(s,a)$ "
             "define $\\hat P$; the library is certified on $\\hat P$ and deployed on the true $P$.}\\label{tab:sampled}",
             "\\begin{tabular}{rrrrr}", "\\toprule",
             "$n$ & trained & library & $\\Gamma$ on $\\hat P$ & true max loss on $P$\\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['n']} & {r['learned']} & {r['library']} & {sci(r['cert'])} & {sci(r['true_max'])}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    parts = []
    parts.append(stops_table("ds", "Double slit (Kuric et al., wind $\\pm3$, $\\gamma=0.9$), weights on the whole simplex "
                             "(the segment), relevant state = start. Each row is the state of one anytime run when "
                             "$\\Gamma\\le\\epsilon$ is first certified; true loss on 2001 weights.",
                             eps_keep=[1e-3, 1e-4, 1e-6, 1e-10], label="tab:ds"))
    parts.append(sufficiency_table())
    parts.append(sampled_table())
    for d, cap, lab in (("windy", "Windy two-stage ($\\gamma=0.95$): the faces of every two-stage task ($\\{A,B\\}$ and every "
                         "pair of corridor exits); relevant = start and exits; regret over 60 tasks.", "tab:windy"),
                        ("office1f2", "Office with slip $p=0.1$ ($\\gamma=0.99$): FSAs whose states branch on $\\le2$ symbols "
                         "(three 4-exit faces); relevant = start and exits; regret over 103 tasks.", "tab:office1"),
                        ("office3f2", "Office with slip $p=0.3$, same task class.", "tab:office3")):
        parts.append(stops_table(d, cap, tasks=True, label=lab))
    with open(os.path.join(NOTES, "certlib_results.tex"), "w") as fh:
        fh.write("\n".join(parts))
    print("written")


if __name__ == "__main__":
    main()
