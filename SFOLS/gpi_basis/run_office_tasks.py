"""Demand-driven certified SF-FSA-VI on slippery Office over a stream of FSA tasks.

usage: PYTHONPATH=SFOLS python -m gpi_basis.run_office_tasks <p>
Compares, task by task: (a) the paper's SF-FSA-VI planning with GPI's ESTIMATE over the 6 corner policies,
(b) demand-driven certified SF-FSA-VI (pool shared across the stream), against the exact optimum.
"""
import os
import pickle
import sys
import time

import numpy as np

from .algosnd import GrowthND
from .domains import slippery_office
from .fsa import PAPER_TASKS, gpi_value, optimal_value, random_tasks
from .fsa_certified import solve_task_certified

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpi_basis")


def main(p):
    mdp = slippery_office(p)
    rng = np.random.default_rng(7)
    tasks = (list(PAPER_TASKS.values()) + random_tasks(40, rng, max_out=1) + random_tasks(40, rng, max_out=2)
             + random_tasks(20, rng, max_out=3))
    opt = [optimal_value(mdp, t)[0] for t in tasks]
    G = GrowthND(mdp)
    for i in range(mdp.D):
        G.grow(np.eye(mdp.D)[i], "full")
    corners_only = [gpi_value(mdp, t, G.pool, upto=mdp.D) for t in tasks]
    rows = []
    t0 = time.perf_counter()
    for i, t in enumerate(tasks):
        before = dict(G.c)
        v, st = solve_task_certified(mdp, t, G)
        rows.append(dict(task=i, opt=opt[i], certified=v, corners_estimate=corners_only[i],
                         n_states=len(t), max_out=max(len(e) for e in t.values()),
                         grows=st["grows"], rounds=st["rounds"],
                         new_full=G.c["full"] - before["full"], new_meta=G.c["meta_ok"] - before["meta_ok"],
                         pool=len(G.pool)))
        print(f"task {i:3d} ({len(t)} states, out<= {rows[-1]['max_out']}): regret certified "
              f"{opt[i] - v:.1e}, corners+estimate {opt[i] - corners_only[i]:.1e}; grows {st['grows']} "
              f"(full {rows[-1]['new_full']}, meta {rows[-1]['new_meta']}), pool {len(G.pool)}", flush=True)
    res = dict(p=p, rows=rows, counters=G.c, time=time.perf_counter() - t0, tasks=tasks)
    with open(os.path.join(OUT, f"office_tasks_p{p}.pkl"), "wb") as fh:
        pickle.dump(res, fh)
    print(f"p={p}: {G.c}, final pool {len(G.pool)}, {res['time']:.0f}s")


if __name__ == "__main__":
    main(float(sys.argv[1]))
