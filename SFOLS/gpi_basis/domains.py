"""Exact models of the stochastic domains used in the GPI-basis experiments."""
import os
import sys

import numpy as np

from .core import TabularMOMDP

SF_FSA_VI = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RM", "sf-fsa-vi"))
if SF_FSA_VI not in sys.path:
    sys.path.insert(0, SF_FSA_VI)


def double_slit(max_wind=3, gamma=0.9):
    """Kuric et al.'s DoubleSlit (wind +-max_wind rows per step, always drifting right).
    Two exits (O1 top, O2 bottom), feature e_i on entering exit i; relevant state = the start."""
    from envs.grid_envs import DoubleSlit
    env = DoubleSlit(max_wind=max_wind)
    coords = [env.state_to_coords[i] for i in range(env.s_dim)]
    S, A, D = len(coords), env.a_dim, env.feat_dim
    term = np.array([c in env.object_ids for c in coords])
    P, PHI = np.zeros((S, A, S)), np.zeros((S, A, D))
    for i, c in enumerate(coords):
        if term[i]:
            continue
        for a in range(A):
            for ns, p in env.coords_act_transition_distr(c, a):
                j = env.coords_to_state[ns]
                PHI[i, a] += p * env.features(c, a, ns)
                if not term[j]:
                    P[i, a, j] += p                  # entering an exit ends the episode
    start = env.coords_to_state[env.initial[0]]
    mdp = TabularMOMDP(P, PHI, gamma, [start], name=f"double_slit_wind{max_wind}")
    mdp.coords, mdp.objects = coords, env.PHI_OBJ_TYPES
    return mdp


def slippery_office(p=0.1, gamma=0.99):
    """Office (Kuric et al.) where the chosen action is replaced by one of the other 3 w.p. p.
    Entering any object ends the episode with its one-hot feature. Relevant states: the start cell
    and the 6 exits (SF-FSA-VI reads values at exits; episodes of later subtasks start there)."""
    from envs.grid_envs import Office
    M = Office.MAP
    H, W = M.shape
    objs = list(Office.PHI_OBJ_TYPES)                      # ['C1', 'C2', 'O1', 'O2', 'M1', 'M2']
    moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    states = [(r, c) for r in range(H) for c in range(W) if M[r, c] != "X"]
    sidx = {s: i for i, s in enumerate(states)}
    S, A, D = len(states), 4, len(objs)
    nxt = np.zeros((S, A), dtype=int)
    for i, (r, c) in enumerate(states):
        for a, (dr, dc) in enumerate(moves):
            n = (r + dr, c + dc)
            nxt[i, a] = sidx[n] if (0 <= n[0] < H and 0 <= n[1] < W and M[n] != "X") else i
    isobj = np.array([M[s] in objs for s in states])
    objfeat = np.zeros((S, D))
    for i, s in enumerate(states):
        if isobj[i]:
            objfeat[i, objs.index(M[s])] = 1.0
    P, T = np.zeros((S, A, S)), np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for b in range(A):
                pr = 1 - p if a == b else p / (A - 1)
                n = nxt[s, b]
                (T if isobj[n] else P)[s, a, n] += pr
    PHI = T @ objfeat
    exits = [sidx[s] for o in objs for s in states if M[s] == o]
    start = sidx[(9, 3)]
    mdp = TabularMOMDP(P, PHI, gamma, [start] + exits, name=f"slippery_office_p{p}")
    mdp.coords, mdp.objects, mdp.exit_index = states, objs, {o: e for o, e in zip(objs, exits)}
    return mdp


def windy_two_stage(max_wind=3, gamma=0.95):
    """Two-stage windy domain (new): stage 1 is a double slit (wind +-max_wind rows per step and a forced
    drift to the right) from S to exit A (top) or B (bottom); from A / B calm corridors lead back left to
    exits E (8 steps) and C (27 steps) on top, F and D on the bottom. Tasks like '(A or B) then (C or D)' put
    the weight on the hedging {A, B} edge, with a different ratio for each task.
    Actions: UP, RIGHT, DOWN, LEFT. In the windy zone LEFT cancels the drift; in corridors moves are
    deterministic. Entering any exit ends the episode with its one-hot feature (as in grid_envs)."""
    H, W = 25, 30
    grid = np.full((H, W), "X", dtype=object)
    grid[3:22, 0:29] = "w"                                    # windy zone
    grid[1, 1:29] = "c"; grid[23, 1:29] = "c"                 # corridors
    exits = {"A": (2, 28), "B": (22, 28), "C": (1, 1), "E": (0, 20), "D": (23, 1), "F": (24, 20)}
    for name, pos in exits.items():
        grid[pos] = name
    start = (12, 0)
    objs = ["A", "B", "C", "D", "E", "F"]
    states = [(r, c) for r in range(H) for c in range(W) if grid[r, c] != "X"]
    sidx = {s: i for i, s in enumerate(states)}
    S, A, D = len(states), 4, len(objs)
    free = lambda rc: 0 <= rc[0] < H and 0 <= rc[1] < W and grid[rc] != "X"
    isexit = lambda rc: grid[rc] in objs

    def walk(pos, dr, dc, n):
        """Move n unit steps in direction (dr, dc); blocked by walls; stops on entering an exit."""
        for _ in range(n):
            nxt = (pos[0] + dr, pos[1] + dc)
            if not free(nxt):
                return pos
            pos = nxt
            if isexit(pos):
                return pos
        return pos

    P, PHI = np.zeros((S, A, S)), np.zeros((S, A, D))
    for s in states:
        i = sidx[s]
        kind = grid[s]
        for a in range(A):
            if kind == "w":
                outs = []
                for wind in range(-max_wind, max_wind + 1):
                    v = wind - (a == 0) + (a == 2)
                    pos = walk(s, int(np.sign(v)), 0, abs(v))
                    if not isexit(pos):
                        h = 1 + (a == 1) - (a == 3)
                        pos = walk(pos, 0, 1, h)
                    outs.append((pos, 1.0 / (2 * max_wind + 1)))
            else:                                             # corridor cell, start cell or exit cell
                dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
                outs = [(walk(s, dr, dc, 1), 1.0)]
            for pos, pr in outs:
                if isexit(pos):
                    PHI[i, a, objs.index(grid[pos])] += pr
                else:
                    P[i, a, sidx[pos]] += pr
    exit_idx = [sidx[exits[o]] for o in objs]
    mdp = TabularMOMDP(P, PHI, gamma, [sidx[start]] + exit_idx, name=f"windy_two_stage_w{max_wind}")
    mdp.coords, mdp.objects = states, objs
    mdp.exit_index = {o: sidx[exits[o]] for o in objs}
    mdp.symbols = {o: (o,) for o in objs}                     # every exit is its own proposition
    mdp.grid = grid
    return mdp
