"""SF-FSA-VI (Kuric et al., ICAPS 2024) with a GPI pool on Office-like domains, and its TRUE value.

A task is {u: {symbol: next_state or 'T'}}. Each automaton state u gets an exit-reward vector w(u):
w_e(u) = 1 if exit e's symbol leads u to the terminal state, the value of the rest of the task from exit e
if it leads to a non-terminal state v, and 0 otherwise. Planning computes these from the basis' value
estimates at the exits; the low-level policy in u is GPI over the basis at weight w(u).
"""
import numpy as np

SYM = {"C": ("C1", "C2"), "M": ("M1", "M2"), "O": ("O1", "O2")}

PAPER_TASKS = {
    "sequential": {"u0": {"C": "u1"}, "u1": {"M": "u2"}, "u2": {"O": "T"}},
    "disjunction": {"u0": {"C": "u1", "M": "u1"}, "u1": {"O": "T"}},
    "composite": {"u0": {"C": "u1", "M": "u2"}, "u1": {"M": "u3"}, "u2": {"C": "u3"}, "u3": {"O": "T"}},
}


def random_tasks(n, rng, max_states=4, max_out=2, symbols=("C", "M", "O")):
    """Random acyclic FSAs: states u0..u_{k-1}, each with 1..max_out distinct outgoing symbols, edges to a
    later state or to the terminal. An edge never enters a state that accepts the same symbol again."""
    tasks = []
    while len(tasks) < n:
        k = int(rng.integers(2, max_states + 1))
        names = [f"u{i}" for i in range(k)]
        task, ok = {}, True
        for i in range(k - 1, -1, -1):
            n_out = int(rng.integers(1, max_out + 1))
            syms = list(rng.choice(symbols, size=n_out, replace=False))
            edges = {}
            for s in syms:
                targets = ["T"] + [names[j] for j in range(i + 1, k) if s not in task.get(names[j], {})]
                edges[s] = targets[int(rng.integers(len(targets)))] if i < k - 1 else "T"
            task[names[i]] = edges
        # keep only states reachable from u0
        reach, frontier = {"u0"}, ["u0"]
        while frontier:
            u = frontier.pop()
            for v in task[u].values():
                if v != "T" and v not in reach:
                    reach.add(v); frontier.append(v)
        tasks.append({u: task[u] for u in names if u in reach})
    return tasks


def _order(task):
    """Reverse topological order (states closest to the terminal first)."""
    out, seen = [], set()

    def visit(u):
        if u in seen or u == "T":
            return
        seen.add(u)
        for v in task[u].values():
            visit(v)
        out.append(u)
    visit("u0")
    for u in task:
        visit(u)
    return out


def symbols(mdp):
    return getattr(mdp, "symbols", SYM)


def plan(mdp, task, value_fn):
    """High-level weights w(u) for every automaton state (one backward pass: tasks are acyclic)."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    W = {}
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else value_fn(W[v])[ex[e]]
        W[u] = w
    return W


def true_start_value(mdp, task, W, policy_fn):
    """True value at the start of executing policy_fn(w(u)) in every automaton state u."""
    objs, ex, SYM = mdp.objects, mdp.exit_index, symbols(mdp)
    psi = {u: mdp.sf(policy_fn(W[u])) for u in task}
    Wt = {}
    for u in _order(task):
        w = np.zeros(mdp.D)
        for sym, v in task[u].items():
            for e in SYM[sym]:
                w[objs.index(e)] = 1.0 if v == "T" else psi[v][ex[e]] @ Wt[v]
        Wt[u] = w
    return psi["u0"][mdp.start] @ Wt["u0"]


def optimal_value(mdp, task):
    def vfun(w):
        return mdp.optimal(w)[1] if np.any(w) else np.zeros(mdp.S)
    W = plan(mdp, task, vfun)
    return true_start_value(mdp, task, W, lambda w: mdp.optimal(w)[0]), W


def gpi_value(mdp, task, pool, upto=None, planner="estimate", cache=None):
    """True value of SF-FSA-VI with GPI over the first `upto` pool policies.
    planner='estimate': exit values from GPI's estimate max_{j,a} Psi_j(e, a).w(v)  (as in the paper);
    planner='evaluated': exit values from the true value of the GPI policy (one evaluation per state)."""
    def act(w):
        return pool.actions(w, upto)
    if planner == "estimate":
        vfun = lambda w: pool.qhat(w, upto).max(1)
    else:
        vfun = lambda w: (cache.get(act(w)).psi_s @ w if cache is not None else mdp.value(act(w), w))
    W = plan(mdp, task, vfun)
    return true_start_value(mdp, task, W, act)


def windy_tasks(n, rng):
    """Two-stage tasks on windy_two_stage: stage 1 'A or B' (or one of them), each leading to a corridor
    stage that accepts 1-2 of {C, D, E, F}. Different tasks put the stage-1 weight at different points of
    the hedging {A, B} edge."""
    tasks = []
    corr = ["C", "D", "E", "F"]
    for _ in range(n):
        k1 = rng.choice([1, 2], p=[0.2, 0.8])
        first = list(rng.choice(["A", "B"], size=k1, replace=False))
        task = {"u0": {}}
        for i, f in enumerate(first):
            u = f"u{i + 1}"
            acc = list(rng.choice(corr, size=int(rng.integers(1, 3)), replace=False))
            task[u] = {c: "T" for c in acc}
            task["u0"][f] = u
        tasks.append(task)
    return tasks
