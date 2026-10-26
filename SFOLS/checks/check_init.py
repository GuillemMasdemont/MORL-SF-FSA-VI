# usage (from SFOLS/, on a compute node): python checks/check_init.py
"""Steps to converge (relative delta 1e-3 at the relevant states) of SFQL at extreme weights: tight optimistic start
(Psi = e_argmax(w)) vs zero start; and flat Q-learning on two Office tasks with Q0 = 0 vs 1."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import warnings; warnings.filterwarnings("ignore")
os.environ["QPARTS"] = ""
from certlib.domains import office, windy_two_stage
import mf

for name, mdp, ws in (("windy", windy_two_stage(), (0, 2, 4)), ("office p=0.1", office(0.1), (0, 3))):
    smp, orc = mf.Sampler(mdp), None
    orc = mf.Oracle(mdp, smp); S, A, D = mdp.S, mdp.A, mdp.D
    mf.warmup(smp, mdp.gamma)
    for wi in ws:
        w = np.eye(D)[wi]; v = orc.vstar(w)[1][mdp.relevant]; tol = 1e-3 * v.max()
        ok = lambda act, vest=None: (v - orc.value(act, w)[mdp.relevant]).max() <= tol
        for init in ("optimistic", "zero"):
            res = []
            for sd in range(3):
                Psi0 = np.tile(w, (S, A, 1)) if init == "optimistic" else np.zeros((S, A, D))
                Psi, n, act, st, cv = mf.sfql(smp, w, Psi0, np.zeros((S, A)), np.ones((S, A), bool), ok, 30_000_000, 100, 1000, sd, mdp.gamma)
                res.append((cv, (v - orc.value(act, w)[mdp.relevant]).max()))
            print(f"{name} e{wi} ({mdp.objects[wi]}), V* at relevant max {v.max():.3f}, {init:10s}: to converge {[r[0] for r in res]}, final loss {max(r[1] for r in res):.1e}", flush=True)
