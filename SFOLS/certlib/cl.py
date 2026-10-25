"""Certified library construction (OLS loop on the GPI certificate, answers by an escalation ladder).

At the corner with the largest gap U_r - E_r, the query weight w is answered by the cheapest step that
closes the gap there (new gap <= accept), in this order:

  'free'   upper bounds from what is already stored: for every library policy j, V^j(r) + B^j(r) with its
           Bellman gap read off its stored action SFs (no transition model needed), and the Bellman
           residual of the GPI estimate V_hat (one backup). Adds a record, no policy.
  'gpi'    GPI's policy over the library at w, evaluated (SF backups warm-started at the GPI estimate);
           added to the library if certified.
  'meta'   metapolicy: policy iteration restricted to B(s), the actions of the learned policies.
  'learn'  policy iteration over all actions (anytime: stops as soon as the certificate closes).
Every evaluated answer is added to the library together with its record (w, V(r) + B(r)).
"""
import time

import numpy as np

from .certificate import Certificate
from .library import Library
from .mdp import tol_argmax

ACC_ZERO = 1e-9          # 'exact' acceptance (numerical zero for the reach bound)

LADDERS = {
    "sfols": ("learn",),                         # every query learned (warm-started at GPI's policy)
    "sfols_cold": ("learn_cold",),               # every query learned from scratch (standard SFOLS)
    "cog": ("meta", "learn"),                    # Certify-or-Grow (the paper)
    "cl": ("free", "gpi", "meta", "learn"),      # certified library
    "test": ("free", "gpi", "meta"),             # sufficiency test of the learned base: no learning
}


class Builder:
    def __init__(self, mdp, face, eps, ladder="cl", eta=0.0, eval_tol=None, verbose=False):
        """face: a list of coordinates (one face of the simplex) or a list of such lists (a union of faces,
        e.g. every face an FSA class can produce). One library is shared by all faces."""
        self.mdp, self.eps, self.verbose = mdp, eps, verbose
        self.ladder = LADDERS[ladder] if isinstance(ladder, str) else tuple(ladder)
        self.accept = max(eta * eps, ACC_ZERO)
        # SF error tol moves the reach bound by <= 2 tol / (1 - gamma): keep that below 1e-4 eps
        self.eval_tol = min(1e-10, 1e-4 * eps * (1 - mdp.gamma)) if eval_tol is None else eval_tol
        self.lib = Library(mdp)
        faces = [face] if np.ndim(face[0]) == 0 else face
        self.certs = [Certificate(mdp, f, use_hull=eps >= 1e-6) for f in faces]
        self.cert = self.certs[0]
        self.face = np.unique(np.concatenate([np.asarray(f, int) for f in faces]))
        self.keys = {}
        self.count = {k: 0 for k in ("free", "gpi", "meta", "learn", "fail")}
        self.open = []                                             # corners no rung could close
        self._verified = False
        self.checkpoint = None                                     # callable(builder), every 25 queries
        self.log = []
        self.t0 = time.perf_counter()
        self.sweeps0 = mdp.sweeps

    # ------------------------------------------------------------------ helpers
    def _rel(self, act, psi):
        r = self.mdp.relevant
        return psi[r, act[r]]                                       # (nr, D)

    def _bound(self, act, psi, w):
        return self.mdp.reach_bound(self.mdp.gap(psi, act, w))

    def _add(self, act, psi, kind, w, B):
        key = np.asarray(act, int).tobytes()
        if key not in self.keys:
            self.keys[key] = len(self.lib)
            self.lib.add(act, psi, kind, w)
            for c in self.certs:
                c.add_policy(psi[self.mdp.relevant])
        u = self._rel(act, psi) @ w + B
        self._record(w, u)

    def _record(self, w, u):
        for c in self.certs:
            if np.abs(np.delete(w, c.F)).sum() < 1e-12:            # w lies in this face
                c.add_record(w, u)

    def _evaluate(self, act, warm):
        key = np.asarray(act, int).tobytes()
        if key in self.keys:                                        # already stored: no evaluation
            return self.lib.stack[self.keys[key]].copy()
        return self.mdp.evaluate(act, warm, tol=self.eval_tol)

    # ------------------------------------------------------------------ the ladder
    def answer(self, w, r_star, cert):
        mdp, lib = self.mdp, self.lib
        E = lambda: cert.lower(w, r_star)
        best = None                                                 # best evaluated candidate so far
        for rung in self.ladder:
            if rung == "free" and len(lib):
                Q = np.tensordot(lib.stack, w, axes=([3], [0]))     # (n, S, A)
                own = np.take_along_axis(Q, np.stack(lib.acts)[:, :, None], 2)[:, :, 0]
                gaps = np.maximum(Q.max(2) - own, 0.0)              # (n, S): stored gaps, no model
                Vr = own[:, mdp.relevant]                           # (n, nr)
                u = np.min(Vr + np.stack([mdp.reach_bound(g) for g in gaps]), axis=0)
                Vh = Q.max(0).max(1)                                # GPI estimate (S,)
                res = np.maximum(mdp.backup_scalar(Vh, w).max(1) - Vh, 0.0)
                u = np.minimum(u, Vh[mdp.relevant] + mdp.reach_bound(res))
                if u[r_star] - E() <= self.accept:
                    self._record(w, u)
                    return "free"
            if rung == "gpi" and len(lib):
                act, warm = lib.gpi(w)
                psi = self._evaluate(act, warm)
                B = self._bound(act, psi, w)
                if B[r_star] <= self.accept:
                    self._add(act, psi, "gpi", w, B)
                    return "gpi"
                best = (act, psi, B)
            if rung == "meta" and "learned" in lib.kinds:
                mask = lib.base_mask()
                Qh = np.where(mask, lib.qhat(w), -np.inf)
                act = tol_argmax(Qh)
                psi = self._evaluate(act, lib.gpi(w)[1] if best is None else best[1])
                act, psi, B, _ = mdp.improve(w, act, psi, allowed=mask, stop_bound=self.accept, tol=self.eval_tol)
                if B[r_star] <= self.accept:
                    self._add(act, psi, "meta", w, B)
                    return "meta"
                if best is None or (self._rel(act, psi) @ w).sum() > (self._rel(best[0], best[1]) @ w).sum():
                    best = (act, psi, B)
            if rung == "learn_cold":
                act = np.zeros(mdp.S, int)
                psi = mdp.evaluate(act, tol=self.eval_tol)
                act, psi, B, _ = mdp.improve(w, act, psi, stop_bound=self.accept, tol=self.eval_tol)
                self._add(act, psi, "learned", w, B)
                return "learn"
            if rung == "learn":
                if best is None:
                    act = lib.gpi(w)[0] if len(lib) else np.zeros(mdp.S, int)
                    psi = self._evaluate(act, lib.gpi(w)[1] if len(lib) else None)
                else:
                    act, psi, _ = best
                act, psi, B, _ = mdp.improve(w, act, psi, stop_bound=self.accept, tol=self.eval_tol)
                self._add(act, psi, "learned", w, B)
                return "learn"
        # no rung closed the gap and learning is not allowed: keep the best composition anyway (it is a
        # valid policy with known SFs, so it raises E and lowers U), and report the corner as open
        if best is not None:
            self._add(best[0], best[1], "gpi", w, best[2])
        return "fail"

    # ------------------------------------------------------------------ main loop
    def run(self, max_queries=5000, exact_final=True, init=True):
        mdp = self.mdp
        for i in (self.face if init else []):                                         # vertices of the face: learned
            w = np.eye(mdp.D)[i]
            act = np.zeros(mdp.S, int)
            act, psi, B, _ = mdp.improve(w, act, mdp.evaluate(act, tol=self.eval_tol), stop_bound=-1.0,
                                         tol=self.eval_tol)
            self._add(act, psi, "learned", w, B)
            self.count["learn"] += 1
            self._log(None, None, "learn")
        q = 0
        self.final_gap = np.inf
        while True:
            tops = [(c.worst(), c) for c in self.certs]
            tops = [(t, c) for t, c in tops if t is not None]
            top, cert = max(tops, key=lambda tc: tc[0][0]) if tops else (None, None)
            if q >= max_queries:                                   # budget exhausted: report current gap
                self.final_gap = 0.0 if top is None else top[0]
                break
            if top is None or top[0] <= self.eps:
                if exact_final and not self._verified:            # one from-scratch recomputation
                    self._verified = True
                    if any(c.k >= 3 and c.verify_full(self.eps) for c in self.certs):
                        continue
                if exact_final and any(c.k >= 3 and c.verify_lp(self.eps) for c in self.certs):
                    continue                                       # cheap: only corners near eps
                self.final_gap = 0.0 if top is None else top[0]
                break
            g, r, w = top
            kind = self.answer(w, r, cert)
            self.count[kind] += 1
            if kind == "fail":
                cert.drop(r, w)
                self.open.append((cert, r, w))
            q += 1
            self._log(g, w, kind)
            if self.checkpoint is not None and q % 25 == 0:
                self.checkpoint(self)
        if self.open:                                              # certificate includes the open corners
            self.final_gap = max([self.final_gap] + [c.gap(w, r) for c, r, w in self.open])
        self.time = time.perf_counter() - self.t0
        return self

    def _log(self, g, w, kind):
        self.log.append(dict(gap=g, w=None if w is None else w.copy(), kind=kind, library=len(self.lib),
                             learned=self.lib.kinds.count("learned"), sweeps=self.mdp.sweeps - self.sweeps0,
                             time=time.perf_counter() - self.t0, **{f"n_{k}": v for k, v in self.count.items()}))
        if self.verbose and len(self.log) % 25 == 0:
            print(f"  q {len(self.log):4d} gap {g if g is not None else float('nan'):.2e} lib {len(self.lib)} "
                  f"learned {self.lib.kinds.count('learned')} sweeps {self.mdp.sweeps - self.sweeps0} "
                  f"{self.count}", flush=True)
