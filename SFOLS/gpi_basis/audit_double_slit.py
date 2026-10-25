"""Audit of the double-slit claims. Is the certified basis better than the CCS, given that it needs an
epsilon? Would the first k policies of the CCS do as well?

Losses here are EXACT over ALL weights of the segment, not on a grid. Inside a GPI cell the loss
V*(x) - V^{pi_C}(x) is convex (V* convex, V^{pi_C} linear), so its sup is attained at the cell ends. V* at the
start is the upper envelope of every policy we ever solved, which includes the exact CCS (SFOLS run to
tol 1e-12); it is checked against independent exact solves.

usage: PYTHONPATH=SFOLS python -m gpi_basis.audit_double_slit <step>
  ccs             SFOLS (full solves) to tol 1e-12: the exact CCS, in SFOLS' query order
  sfols_tol       SFOLS' own stopping rule at tol 1e-3 .. 1e-6 (pool size when it stops)
  targeted        GPI-targeted + Certify-or-Grow, eps 1e-6 (re-run to keep the policies)
  targeted_exact  the same with eps 1e-10: a basis certified EXACTLY optimal for GPI
  curves          exact sup loss of every prefix of every order, and of random orders of the exact CCS
  oracle          oracle-targeted order (adds the policy optimal where GPI's TRUE loss is largest)
  elim <eps>      backward elimination from the exact CCS (how small can an eps-optimal GPI basis be?)
"""
import os
import pickle
import sys
import time

import numpy as np

from .algos1d import run_sfols, run_targeted
from .core import PolicyCache, tol_argmax
from .domains import double_slit
from .segment import Segment, _merge, envelope_label_breaks

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")
EPS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-10]


def _save(name, G, **extra):
    res = dict(actions=np.stack([r.actions for r in G.pool.records]).astype(np.int16),
               weights=[w.tolist() for w in G.pool.weights], kinds=G.pool.kinds, counters=G.c, log=G.log,
               cert_log=getattr(G, "cert_log", None), **extra)
    with open(os.path.join(OUT, f"audit_ds_{name}.pkl"), "wb") as fh:
        pickle.dump(res, fh)


def _load(name):
    with open(os.path.join(OUT, name), "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------- fast GPI cells
def _pareto_pad(I, M):
    """Per (state, action): the pool lines that no single other line dominates on [0, 1] (dropping the
    others leaves GPI's estimate unchanged at every x in [0, 1]), padded to (S, A, H)."""
    v0, v1 = I, I + M                                       # values at x = 0 and x = 1, (k, S, A)
    order = np.lexsort((-v1, -v0), axis=0)                  # v0 descending, then v1 descending
    v1s = np.take_along_axis(v1, order, 0)
    keep = np.ones_like(v1s, dtype=bool)
    keep[1:] = v1s[1:] > np.maximum.accumulate(v1s, axis=0)[:-1]
    H = int(keep.sum(0).max())
    first = np.argsort(~keep, axis=0, kind="stable")[:H]    # kept rows first, (H, S, A)
    idx = np.take_along_axis(order, first, 0)
    ok = np.take_along_axis(keep, first, 0)
    Ip = np.where(ok, np.take_along_axis(I, idx, 0), -np.inf)
    Mp = np.where(ok, np.take_along_axis(M, idx, 0), 0.0)
    return Ip.transpose(1, 2, 0), Mp.transpose(1, 2, 0), ok.transpose(1, 2, 0)


def fast_gpi_cells(mdp, stack, seg, cache):
    """Same cells as segment.gpi_cells, but every (state, action) keeps only the pool lines that are not
    dominated on [0, 1], which leaves GPI's estimate unchanged and makes each evaluation much cheaper."""
    states = np.flatnonzero((mdp.P.sum((1, 2)) > 0) | (np.abs(mdp.PHI).sum((1, 2)) > 0))
    Ip, Mp, ok = _pareto_pad(stack @ seg.b, stack @ seg.d)
    xs = set()
    for s in states:
        xs.update(envelope_label_breaks(Ip[s][ok[s]], Mp[s][ok[s]], np.nonzero(ok[s])[0]))
    edges = _merge(sorted(xs))

    def act(x):
        return tol_argmax((Ip + Mp * x).max(2))

    cells = []
    stackx = [(lo, hi, 0) for lo, hi in zip(edges[:-1], edges[1:])][::-1]
    while stackx:
        lo, hi, depth = stackx.pop()
        mid = 0.5 * (lo + hi)
        a_mid = act(mid)
        d = 1e-7 * (hi - lo)
        if depth < 25 and (not np.array_equal(act(lo + d), a_mid) or not np.array_equal(act(hi - d), a_mid)):
            stackx.append((mid, hi, depth + 1))
            stackx.append((lo, mid, depth + 1))
            continue
        rec = cache.get(a_mid)
        if cells and cells[-1][2].key == rec.key and abs(cells[-1][1] - lo) < 1e-15:
            cells[-1] = (cells[-1][0], hi, rec)
        else:
            cells.append((lo, hi, rec))
    return cells


# ---------------------------------------------------------------------- exact losses
class Exact:
    def __init__(self, mdp, seg, cache):
        self.mdp, self.seg, self.cache, self.r = mdp, seg, cache, mdp.relevant[0]
        self.lines = np.zeros((0, mdp.D))

    def add_lines(self, actions_list):
        L = [self.cache.get(a).psi_s[self.r] for a in actions_list]
        self.lines = np.unique(np.round(np.vstack([self.lines] + L), 15), axis=0)

    def vstar(self, xs):
        return (self.seg.w(np.atleast_1d(xs)) @ self.lines.T).max(1)

    def stack(self, actions_list):
        return np.stack([self.cache.get(a).psi_sa for a in actions_list])

    def sup_loss(self, stack):
        """Exact sup over the segment of V*(start) - V^GPI(start), and the cell count."""
        cells = fast_gpi_cells(self.mdp, stack, self.seg, self.cache)
        xs = np.array([[lo, hi] for lo, hi, _ in cells])
        vals = np.array([[rec.psi_s[self.r] @ self.seg.w(lo), rec.psi_s[self.r] @ self.seg.w(hi)]
                         for lo, hi, rec in cells])
        loss = self.vstar(xs.ravel()).reshape(-1, 2) - vals
        return float(max(loss.max(), 0.0)), len(cells), cells, xs


def first_stay(ks, losses, eps):
    """First pool size from which the loss stays <= eps (GPI is not monotone in the pool)."""
    losses = np.asarray(losses)
    for i in range(len(losses)):
        if (losses[i:] <= eps).all():
            return ks[i]
    return None


# ---------------------------------------------------------------------- steps
def step_ccs(mdp, seg):
    t0 = time.perf_counter()
    G = run_sfols(mdp, seg, "full", tol=1e-12)
    print(f"exact SFOLS: {G.c}, CCS {len(G.ccs)}, {time.perf_counter() - t0:.0f}s", flush=True)
    _save("ccs", G, ccs_size=len(G.ccs), t_run=time.perf_counter() - t0)


def step_sfols_tol(mdp, seg):
    out = {}
    for tol in EPS[:4]:
        t0 = time.perf_counter()
        G = run_sfols(mdp, seg, "full", tol=tol)
        out[tol] = dict(pool=len(G.pool), ccs=len(G.ccs), time=time.perf_counter() - t0,
                        actions=np.stack([r.actions for r in G.pool.records]).astype(np.int16))
        print(f"SFOLS tol {tol:.0e}: stops with {len(G.pool)} queries (CCS {len(G.ccs)}), "
              f"{out[tol]['time']:.0f}s", flush=True)
    with open(os.path.join(OUT, "audit_ds_sfols_tol.pkl"), "wb") as fh:
        pickle.dump(out, fh)


def step_targeted(mdp, seg, eps, name):
    t0 = time.perf_counter()
    G = run_targeted(mdp, seg, "cog", eps=eps, max_queries=400, verbose=True)
    print(f"targeted eps {eps:.0e}: {G.c}, pool {len(G.pool)}, cert {G.final_cert['cert']:.2e}, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    _save(name, G, t_run=time.perf_counter() - t0)


RUNS = {"sfols": "audit_ds_ccs.pkl", "targeted": "audit_ds_targeted.pkl",
        "targeted_exact": "audit_ds_targeted_exact.pkl", "tchord": "ds_tchord_cog.pkl",
        "gpils": "ds_gpils_full.pkl"}
_W = {}


def _init():
    mdp = double_slit()
    seg = Segment([1, 0], [0, 1])
    cache = PolicyCache(mdp, maxsize=6000)
    E = Exact(mdp, seg, cache)
    runs = {k: _load(f) for k, f in RUNS.items() if os.path.exists(os.path.join(OUT, f))}
    for r in runs.values():
        E.add_lines(list(r["actions"]))
    _W.update(mdp=mdp, seg=seg, cache=cache, E=E, runs=runs, stacks={})


def _stack(name):
    if name not in _W["stacks"]:
        _W["stacks"][name] = _W["E"].stack(list(_W["runs"][name]["actions"]))
    return _W["stacks"][name]


def _task(t):
    name, idx = t
    st = _stack(name)
    sub = st[:idx] if isinstance(idx, int) else st[list(idx)]
    return t, _W["E"].sup_loss(sub)[0]


def ccs_members():
    """Indices (in SFOLS' query order) of the policies that improve the envelope at their own weight."""
    E, r = _W["E"], _W["runs"]["sfols"]
    lines = np.stack([_W["cache"].get(a).psi_s[E.r] for a in r["actions"]])
    members = []
    for i, w in enumerate(r["weights"]):
        if not members or lines[i] @ w > max(lines[j] @ w for j in members) + 1e-12:
            members.append(i)
    return members


def _pool(n):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(n, initializer=_init)


def step_curves(workers=10):
    _init()
    E, mdp, seg = _W["E"], _W["mdp"], _W["seg"]
    z = np.load(os.path.join(OUT, "vstar_double_slit.npz"))
    err_grid = np.abs(E.vstar(z["x"]) - z["V"][:, E.r]).max()
    xr = np.random.default_rng(0).random(40)
    err_rand = max(abs(E.vstar(x)[0] - mdp.optimal(seg.w(x))[1][E.r]) for x in xr)
    print(f"V* envelope ({len(E.lines)} lines): max error vs grid {err_grid:.1e}, vs 40 random solves "
          f"{err_rand:.1e}", flush=True)
    members = ccs_members()
    print(f"exact CCS: {len(members)} policies in {len(_W['runs']['sfols']['actions'])} SFOLS queries", flush=True)
    rng = np.random.default_rng(1)
    ks_r = [4, 8, 16, 36, 79, 109, 165, 220]
    perms = [[members[0], members[1]] + list(rng.permutation(members[2:])) for _ in range(20)]
    tasks = []
    for name, r in _W["runs"].items():
        n = len(r["actions"])
        tasks += [(name, k) for k in sorted(set(range(2, min(n, 120) + 1)) | set(range(120, n + 1, 3)) | {n})]
    tasks += [("sfols", tuple(p[:k])) for p in perms for k in ks_r]
    tasks.sort(key=lambda t: -(t[1] if isinstance(t[1], int) else len(t[1])))
    t0 = time.perf_counter()
    with _pool(workers) as P:
        res = dict(P.imap_unordered(_task, tasks, chunksize=1))
        print(f"{len(tasks)} exact losses in {time.perf_counter() - t0:.0f}s", flush=True)
        # densify: every pool size inside the bracket where 'stays <= eps' starts
        extra = set()
        for name in _W["runs"]:
            ks = sorted(k for (nm, k) in res if nm == name and isinstance(k, int))
            ls = [res[(name, k)] for k in ks]
            for eps in EPS:
                k1 = first_stay(ks, ls, eps)
                if k1 is not None and ks.index(k1) > 0:
                    extra |= {(name, k) for k in range(ks[ks.index(k1) - 1] + 1, k1)}
        res.update(dict(P.imap_unordered(_task, sorted(extra), chunksize=1)))
    out = dict(verr=(err_grid, err_rand), members=members, perms=perms, ks_r=ks_r, loss=res, curves={})
    for name, r in _W["runs"].items():
        ks = sorted(k for (nm, k) in res if nm == name and isinstance(k, int))
        ls = [res[(name, k)] for k in ks]
        full = [e["full"] for e in r["log"]]
        out["curves"][name] = dict(k=ks, loss=ls, full=full)
        print(f"{name:15s} n={len(r['actions']):3d}: " + ", ".join(
            f"<={e:.0e} from {first_stay(ks, ls, e)}" for e in EPS) + f"; final {ls[-1]:.1e}", flush=True)
    for k in ks_r:
        v = [res[("sfols", tuple(p[:k]))] for p in perms]
        print(f"random CCS order k={k}: median {np.median(v):.1e}, worst {np.max(v):.1e}", flush=True)
    with open(os.path.join(OUT, "audit_ds_curves.pkl"), "wb") as fh:
        pickle.dump(out, fh)


def step_oracle(max_pool=320):
    """Oracle-targeted order: always add the policy that is optimal where GPI's TRUE loss is largest."""
    _init()
    E, seg = _W["E"], _W["seg"]
    st = _stack("sfols")
    lines = np.stack([_W["cache"].get(a).psi_s[E.r] for a in _W["runs"]["sfols"]["actions"]])
    chosen, log = [0, 1], []
    t0 = time.perf_counter()
    while len(chosen) <= max_pool:
        loss, nc, cells, xs = E.sup_loss(st[chosen])
        log.append((len(chosen), loss))
        print(f"  oracle pool {len(chosen):4d} loss {loss:.2e} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if loss <= 1e-10:
            break
        vals = np.array([[rec.psi_s[E.r] @ seg.w(lo), rec.psi_s[E.r] @ seg.w(hi)] for lo, hi, rec in cells])
        lo_hi = E.vstar(xs.ravel()).reshape(-1, 2) - vals
        x = xs.ravel()[int(lo_hi.argmax())]
        j = int(np.argmax(lines @ seg.w(x)))
        if j in chosen:
            print("  optimal policy already in the pool: stop", flush=True)
            break
        chosen.append(j)
    with open(os.path.join(OUT, "audit_ds_oracle.pkl"), "wb") as fh:
        pickle.dump(dict(chosen=chosen, log=log), fh)


def step_elim(eps, source="sfols"):
    """Backward elimination: drop a policy (latest first) whenever GPI stays eps-optimal at every weight.
    source 'sfols' starts from the exact CCS, 'targeted_exact' from our exactly certified basis,
    'oracle' from the oracle-targeted basis."""
    _init()
    E = _W["E"]
    if source == "oracle":
        start = _load("audit_ds_oracle.pkl")["chosen"]
        st_src = _stack("sfols")
    else:
        st_src = _stack(source)
        start = ccs_members() if source == "sfols" else list(range(len(_W["runs"][source]["actions"])))
    keep = list(start)
    t0 = time.perf_counter()
    for n, i in enumerate(sorted(start[2:], reverse=True)):
        trial = [j for j in keep if j != i]
        if E.sup_loss(st_src[trial])[0] <= eps:
            keep = trial
        if n % 20 == 0:
            print(f"  elim {source} {eps:.0e}: tried {n + 1}, kept {len(keep)} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    print(f"backward elimination from {source}, sup loss <= {eps:.0e}: {len(keep)} of {len(start)} policies",
          flush=True)
    with open(os.path.join(OUT, f"audit_ds_elim_{source}_{eps:.0e}.pkl"), "wb") as fh:
        pickle.dump(dict(keep=keep, start=start), fh)


def _cert_task(k):
    """Our certificate (eps = 0, exact refinement) on the first k policies of SFOLS' exact run."""
    from .segment import certify
    E, seg, mdp = _W["E"], _W["seg"], _W["mdp"]
    r = _W["runs"]["sfols"]
    st = _stack("sfols")[:k]
    cells = fast_gpi_cells(mdp, st, seg, _W["cache"])
    kx = [float(w[0]) for w in r["weights"][:k]]
    kv = [_W["cache"].get(a).psi_s[mdp.relevant] @ np.asarray(w) for a, w in zip(r["actions"][:k], r["weights"][:k])]
    return k, certify(mdp, cells, seg, kx, kv, eps=0.0)["cert"]


def step_cert_sfols(workers=4):
    ks = list(range(100, 546, 10)) + [546]
    with _pool(workers) as P:
        res = dict(P.imap_unordered(_cert_task, ks, chunksize=1))
    for k in sorted(res):
        print(f"  SFOLS order, first {k}: certified {res[k]:.2e}", flush=True)
    with open(os.path.join(OUT, "audit_ds_cert_sfols.pkl"), "wb") as fh:
        pickle.dump(res, fh)


def step_report():
    C = _load("audit_ds_curves.pkl")
    tol = _load("audit_ds_sfols_tol.pkl")
    sf_cert = _load("ds_sfols_full.pkl")["cert_curve"]
    T, TX = _load("audit_ds_targeted.pkl"), _load("audit_ds_targeted_exact.pkl")
    O = _load("audit_ds_oracle.pkl") if os.path.exists(os.path.join(OUT, "audit_ds_oracle.pkl")) else None

    def cert_first(log, eps):
        return next((d["pool"] for d in log if d["cert"] <= eps), None)

    cv = C["curves"]
    print("eps     | SFOLS own stop | SFOLS order: true / certified | targeted: true / certified | oracle true")
    for eps in EPS:
        own = tol[eps]["pool"] if eps in tol else "546 (exact CCS 274)"
        sf_true = first_stay(cv["sfols"]["k"], cv["sfols"]["loss"], eps)
        sf_c = cert_first(sf_cert, eps)
        t_true = first_stay(cv["targeted_exact"]["k"], cv["targeted_exact"]["loss"], eps)
        t_c = cert_first(TX["cert_log"], eps)
        o = None
        if O is not None:
            o = first_stay([k for k, _ in O["log"]], [l for _, l in O["log"]], eps)
        print(f"{eps:.0e} | {own} | {sf_true} / {sf_c} | {t_true} / {t_c} | {o}")
    for name, k in (("sfols", 36), ("sfols", 109), ("targeted", 109), ("targeted_exact", 145), ("sfols", 203)):
        ks, ls = cv[name]["k"], cv[name]["loss"]
        if k in ks:
            print(f"exact sup loss of the first {k} policies of {name}: {ls[ks.index(k)]:.2e}")
    for src in ("targeted_exact", "oracle", "sfols"):
        for eps in (1e-10, 1e-6):
            f = f"audit_ds_elim_{src}_{eps:.0e}.pkl"
            if os.path.exists(os.path.join(OUT, f)):
                e = _load(f)
                print(f"backward elimination from {src}, eps {eps:.0e}: {len(e['keep'])} of {len(e['start'])}")


if __name__ == "__main__":
    mdp = double_slit()
    seg = Segment([1, 0], [0, 1])
    step = sys.argv[1]
    if step == "ccs":
        step_ccs(mdp, seg)
    elif step == "sfols_tol":
        step_sfols_tol(mdp, seg)
    elif step == "targeted":
        step_targeted(mdp, seg, 1e-6, "targeted")
    elif step == "targeted_exact":
        step_targeted(mdp, seg, 1e-10, "targeted_exact")
    elif step == "curves":
        step_curves()
    elif step == "oracle":
        step_oracle()
    elif step == "elim":
        step_elim(float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "sfols")
    elif step == "cert_sfols":
        step_cert_sfols()
    elif step == "report":
        step_report()
