"""Stochastic versions of Kuric et al.'s domains as solve-free MOMDPs (models built in gpi_basis.domains):

  double_slit(max_wind)   Kuric et al.'s DoubleSlit (wind +-max_wind rows per step), 2 exits
  office(p)               Kuric et al.'s Office with random_act_prob = p (the chosen action is replaced
                          by one of the other three w.p. p), 6 exits; relevant = start + exits
  windy_two_stage()       our two-stage windy domain, 6 exits; relevant = start + exits

Every exit ends the episode with its one-hot feature, so PHI[s, a, i] is the probability of entering
exit i, and (P[s, a, :], PHI[s, a, :]) is the full next-outcome distribution (used for sampling).
"""
import numpy as np

from gpi_basis import domains as _d

from .mdp import MOMDP


def _wrap(m, **extra):
    out = MOMDP(m.P, m.PHI, m.gamma, m.relevant, name=m.name)
    for k in ("coords", "objects", "exit_index", "symbols", "grid"):
        if hasattr(m, k):
            setattr(out, k, getattr(m, k))
    for k, v in extra.items():
        setattr(out, k, v)
    return out


def double_slit(max_wind=3, gamma=0.9):
    return _wrap(_d.double_slit(max_wind=max_wind, gamma=gamma))


def office(p=0.1, gamma=0.99):
    return _wrap(_d.slippery_office(p=p, gamma=gamma))


def windy_two_stage(max_wind=3, gamma=0.95):
    return _wrap(_d.windy_two_stage(max_wind=max_wind, gamma=gamma))


def sampled(mdp, n, rng):
    """Empirical model from n sampled outcomes per (s, a) (a generative-model replay buffer).
    Outcomes: the next non-terminal state, or entering exit i (terminal, feature e_i)."""
    S, A, D = mdp.S, mdp.A, mdp.D
    probs = np.concatenate([mdp.P, mdp.PHI], axis=2)                 # (S, A, S + D)
    probs = probs / np.maximum(probs.sum(2, keepdims=True), 1e-300)
    live = mdp.P.sum(2) + mdp.PHI.sum(2) > 0                        # (s, a) with any outcome
    P, PHI = np.zeros_like(mdp.P), np.zeros_like(mdp.PHI)
    for s in range(S):
        for a in range(A):
            if not live[s, a]:
                continue
            c = rng.multinomial(n, probs[s, a]) / n
            P[s, a], PHI[s, a] = c[:S], c[S:]
    out = MOMDP(P, PHI, mdp.gamma, mdp.relevant, name=f"{mdp.name}_n{n}")
    for k in ("coords", "objects", "exit_index", "symbols"):
        if hasattr(mdp, k):
            setattr(out, k, getattr(mdp, k))
    return out
