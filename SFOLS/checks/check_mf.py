# usage (from SFOLS/, on a compute node): python checks/check_mf.py
"""(1) numba evaluators vs exact solves; (2) numba vs Python SFQL at a fixed budget (no early stop): distribution of the
Q error and of the value loss of the greedy policy over seeds; (3) windy timing with oracle checks."""
import os, random, sys, time, bisect
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import warnings; warnings.filterwarnings("ignore")
from certlib.domains import office, windy_two_stage
from certlib.experiments import tasks_for
from certlib.lazyfsa import Task, product_value, product_optimum
import mf

def python_sfql_fixed(mdp, w, budget, seed):
    S, A, D, G = mdp.S, mdp.A, mdp.D, mdp.gamma
    CUM, OUT = [[None] * A for _ in range(S)], [[None] * A for _ in range(S)]
    for s in range(S):
        for a in range(A):
            outs = [(float(p), int(n), -1) for n, p in enumerate(mdp.P[s, a]) if p > 0]
            outs += [(float(p), -1, int(o)) for o, p in enumerate(mdp.PHI[s, a]) if p > 0]
            CUM[s][a] = np.cumsum([o[0] for o in outs]).tolist(); CUM[s][a][-1] = 1.0 + 1e-12
            OUT[s][a] = [(o[1], o[2]) for o in outs]
    rng = random.Random(seed)
    Psi = np.tile(np.eye(D)[int(np.argmax(w))], (S, A, 1)); Qw = Psi @ w; n = np.zeros((S, A)); E = np.eye(D)
    s, t = None, 100
    for _ in range(budget):
        if s is None or t >= 100:
            s, t = rng.randrange(S), 0; a = rng.randrange(A)
        else:
            a = rng.randrange(A) if rng.random() < 0.1 else int(Qw[s].argmax())
        ns, o = OUT[s][a][bisect.bisect_right(CUM[s][a], rng.random())]
        target = E[o] if o >= 0 else G * Psi[ns, int(Qw[ns].argmax())]
        n[s, a] += 1; Psi[s, a] += (target - Psi[s, a]) / (n[s, a] + 1) ** 0.7; Qw[s, a] = Psi[s, a] @ w
        t += 1; s = None if o >= 0 else ns
    return Psi, n

mdp = office(0.1); smp = mf.Sampler(mdp); orc = mf.Oracle(mdp, smp)
S, A, D = mdp.S, mdp.A, mdp.D
# (1) evaluators
w = np.array([0.3, 0.1, 0.2, 0.15, 0.05, 0.2]); act = np.random.default_rng(0).integers(0, A, S)
Vex = np.linalg.solve(np.eye(S) - mdp.gamma * mdp.P[np.arange(S), act], mdp.PHI[np.arange(S), act] @ w)
print("eval_policy max error", np.abs(orc.value(act, w, np.zeros(S)) - Vex).max())
tasks = tasks_for("office", mdp, np.random.default_rng(1))[:6]
for task in tasks[:3]:
    T = Task(mdp, task); acts, Vopt, _ = product_optimum(mdp, T)
    tau = np.array([[-1 if v == "T" else T.U.index(v) for v in T.tau[u]] for u in T.U], np.int64)
    A2 = np.stack([acts[u] for u in T.U]).astype(np.int64)
    Vp = mf.eval_product(A2, tau, T.exits.astype(np.int64), smp.prob, smp.nxt, smp.obj, mdp.gamma, np.zeros((len(T.U), S)), 1e-12)
    print("eval_product vs product_value max error", max(np.abs(Vp[i] - Vopt[u]).max() for i, u in enumerate(T.U)))
# (2) fixed-budget statistics
for wi in (0, 3):
    w = np.eye(D)[wi]; Vs = orc.vstar(w)[1]; Qs = mdp.PHI @ w + mdp.gamma * (mdp._P2 @ Vs).reshape(S, A)
    for budget in (300_000, 1_500_000):
        stats = {"python": [], "numba": []}
        for sd in range(8):
            Psi, n = python_sfql_fixed(mdp, w, budget, 7 + sd)
            Q = Psi @ w; stats["python"].append((np.abs(Q - Qs)[n > 0].mean(), (Vs - orc.value(Q.argmax(1), w))[mdp.relevant].max()))
            Psi0 = np.tile(np.eye(D)[wi], (S, A, 1))
            Psi, n, act, st, cv = mf.sfql(smp, w, Psi0, np.zeros((S, A)), np.ones((S, A), bool), lambda a, v: False, budget, 10**9, 1000, 7 + sd, mdp.gamma)
            Q = Psi @ w; stats["numba"].append((np.abs(Q - Qs)[n > 0].mean(), (Vs - orc.value(Q.argmax(1), w))[mdp.relevant].max()))
        for k, v in stats.items():
            v = np.array(v)
            print(f"  e{wi} budget {budget:>9,} {k:6s}: mean|Q-Q*| {v[:,0].mean():.2e} (sd {v[:,0].std():.1e}), value loss at REL median {np.median(v[:,1]):.1e} max {v[:,1].max():.1e}", flush=True)
# (3) windy timing
mdp = windy_two_stage(); smp = mf.Sampler(mdp); orc = mf.Oracle(mdp, smp); S, A, D = mdp.S, mdp.A, mdp.D
for wi in (0, 2):
    w = np.eye(D)[wi]; vref = orc.vstar(w)[1][mdp.relevant]
    ok = lambda act, vest=None: (vref - orc.value(act, w)[mdp.relevant]).max() <= 1e-3
    t0 = time.perf_counter(); out = []
    for sd in range(3):
        Psi, n, act, st, cv = mf.sfql(smp, w, np.tile(np.eye(D)[wi], (S, A, 1)), np.zeros((S, A)), np.ones((S, A), bool), ok, 20_000_000, 100, 1000, sd, mdp.gamma)
        out.append((st, cv, (vref - orc.value(act, w)[mdp.relevant]).max()))
    dt = time.perf_counter() - t0
    print(f"windy e{wi}: steps {[o[0] for o in out]} to-converge {[o[1] for o in out]} final loss {max(o[2] for o in out):.1e} ({dt:.1f}s, {sum(o[0] for o in out)/dt/1e6:.2f}M steps/s)", flush=True)
