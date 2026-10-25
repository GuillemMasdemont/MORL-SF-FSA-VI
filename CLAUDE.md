# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research workspace (not a single installable package, not a git repo at this top level) for Multi-Objective
Reinforcement Learning experiments built on **Successor Features (SF)** and **Generalized Policy Improvement
(GPI)**. It holds three loosely-coupled, independently-installed subprojects plus exploratory notebooks and a
LaTeX write-up, with no shared build system, no test suite, and no CI/lint config anywhere — treat this as
research/experiment code, not production code.

- `OKB/` — code for the paper *"Constructing an Optimal Behavior Basis for the Option Keyboard"*. A pip-installable
  package (`rl/`) plus CLI experiment scripts and analysis notebooks.
- `RM/sf-fsa-vi/` — a vendored, separately git-tracked clone of the paper *"Planning with a Learned Policy Basis to
  Optimally Solve Complex Tasks"* (ICAPS 2024). Has its own `.git` and two nested git checkouts under `lof/` and
  `sfols/` (declared as submodules in `.gitmodules`, pointing at `guillermoim/lof` and `guillermoim/sfols`).
- `SFOLS/` — standalone Jupyter notebooks (`01_dst_exploration.ipynb`, `02_sfols_implementation.ipynb`,
  `03_adaptive_basis.ipynb`) that build up SFOLS/OLS/GPI/Certify-or-Grow concepts from scratch and exactly (tabular
  policy/value iteration, no learning), on Deep Sea Treasure. `03_adaptive_basis.ipynb` is the reference
  implementation for the "Certify-or-Grow" adaptive-basis idea.
- `notes/adaptive_basis.tex` (+ `notes/figures/`) — the theory write-up backing the adaptive-basis / Certify-or-Grow
  work; several notebook docstrings across `OKB/` and `SFOLS/` explicitly reference it (`see notes/adaptive_basis.tex`).
- `venv/` — a plain Python 3.13 venv (not conda) used only as the kernel for `SFOLS/01_dst_exploration.ipynb`.

There is no root-level `requirements.txt`/`pyproject.toml` tying these together — each subproject has its own
environment, and combining them (see "Cross-project notebooks" below) is done by manual `sys.path` manipulation,
not a shared install.

## Environments

Three conda environments plus one venv, registered as Jupyter kernels (`jupyter kernelspec list`). Match the
notebook/script to its environment — don't assume the currently-active shell env is the right one:

| Environment | Used for |
|---|---|
| `okb` (conda, `/opt/anaconda3/envs/okb`) | `OKB/notebooks/*.ipynb`, `OKB/experiments/*.py`. Also used by `SFOLS/02_*.ipynb` and `SFOLS/03_*.ipynb`, and by notebooks that bridge OKB with `RM/sf-fsa-vi` (see below) — it has both OKB's deps (jax/flax/torch) and enough to run `RM/sf-fsa-vi`'s pure-Python `fsa`/`envs` modules. |
| `sf-fsa-vi` (conda, `/opt/anaconda3/envs/sf-fsa-vi`) | `RM/sf-fsa-vi/run_sfols.py`, `run_lof.py`, and `RM/sf-fsa-vi/notebooks/*.ipynb`. |
| `morl-sfols` (kernel name) → `venv/` at repo root | `SFOLS/01_dst_exploration.ipynb` only (uses `mo_gymnasium`, no OKB/sf-fsa-vi deps). |

## Common commands

**Install OKB** (from `OKB/`): `pip install -e .` — pulls in pinned jax/jaxlib/flax/optax/chex versions (flax 0.7.5
needs jax==0.4.20 exactly; jax's `define_bool_state` was removed later), plus cvxpy, mo-gymnasium, mujoco, etc. Don't
casually bump these pins.

**Run an OKB experiment** (from `OKB/`, `okb` env), CLI via `fire`:
```bash
python experiments/okb_minecart.py --g=20 --use_ok=True --weight_selection=okb --num_nets=5 --run_name="OKB"
python experiments/okb_highway.py --use_ok=True --weight_selection=okb --run_name="OKB"
python experiments/okb_pickplace.py --use_ok=True --weight_selection=okb --num_nets=10 --g=10 --d=8 --run_name="OKB"
python experiments/okb_item_collector.py --use_ok=True --weight_selection=okb --run_name="OKB"
```

**Run sf-fsa-vi** (from `RM/sf-fsa-vi/`, `sf-fsa-vi` env), CLI via Hydra (`env` and `algorithm` are required
overrides, `conf/default.yaml` marks them `???`):
```bash
python run_sfols.py algorithm=sfols-qlearning env=office +fsa_name=task1 +seed=1001
python run_lof.py   algorithm=lof-qlearning   env=office +fsa_name=task1 +seed=1001
```
See `run_office.sh` / `run_delivery.sh` for the full seed sweeps used in the paper's experiments. Note:
`conf/env/office.yaml` doesn't exist upstream — it was reconstructed locally (inferred from
`envs/grid_envs.py`'s `Office.PHI_OBJ_TYPES` and `fsa/tasks_specification.py`'s `fsa_office1/2/3`) to make these
commands runnable; treat other missing `conf/env/*.yaml` files the same way if you hit `MissingMandatoryValue`.

**Execute a notebook headlessly** (this is the closest thing to a test in this repo — there is no pytest anywhere).
Always invoke the target env's own `jupyter`, not whatever's on `$PATH`:
```bash
/opt/anaconda3/envs/okb/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 path/to/notebook.ipynb
```
Swap the env prefix for `sf-fsa-vi`, or use `venv/bin/jupyter` for `SFOLS/01_dst_exploration.ipynb`. Full CCS/OLS
discovery over the Office domain is slow (tens of seconds to minutes per run — OLS's corner-weight polytope
enumeration via `cvxpy`/`cdd` dominates, not the exact solves) and certify-or-grow/multi-seed sweeps over many test
weights can run into the tens of minutes; give `--ExecutePreprocessor.timeout` a generous value and expect to run
these in the background.

## Cross-cutting architecture

**Successor Features + GPI + OLS vocabulary recurs everywhere** and is worth knowing before reading any one file:
- **Successor Features (SF)**: a policy's value under any linear reward weight `w` decomposes as `Ψ(s,a)·w`, where
  `Ψ` (a vector per state-action) is learned/solved once per policy and reused for any `w`.
  `rl.successor_features.ols.OLS` (OKB) and `sfols/rl/successor_features/*.py` (sf-fsa-vi's vendored copy) both
  implement variants of this.
- **OLS (Optimistic Linear Support)**: the algorithm that discovers a Convex Coverage Set (CCS) — the minimal set of
  policies whose SF vectors cover the Pareto front — by iteratively querying "corner weights" of the current CCS
  and solving new policies for them. `corner_weights_new`/`get_k_corner_weights` in `OKB/rl/successor_features/ols.py`
  do the polytope-vertex enumeration (`cvxpy` + `pycddlib`) driving this.
- **GPI (Generalized Policy Improvement)**: given several policies' SF tables, `argmax` over the FULL action set of
  `max_j Ψⱼ(s,a)·w` is guaranteed at least as good as any individual base policy — a genuine one-step
  policy-improvement lookahead, not just picking one of the base policies' own behaviors.
- **Metapolicy / Certify-or-Grow** (the `SFOLS/03_adaptive_basis.ipynb` and `notes/adaptive_basis.tex` idea, and
  `OKB/notebooks/office_rmvi_comparison.ipynb`'s tabular reimplementation): a *more restricted* composition than
  GPI — at each state, only choose among the k base policies' own prescribed actions (not the full action set), but
  refine that choice via real policy iteration to a fixed point. "Certify-or-Grow" checks whether the resulting
  value already satisfies the true Bellman optimality equation (a zero-gap certificate); if not, it falls back to
  an exact solve for the query weight and grows the basis. This is provably optimal at any weight it actually
  certifies against, but is a *different*, non-nested comparison to GPI — GPI can beat it (full action set) or lose
  to it (exact per-weight optimality guarantee), depending on the weight and basis. Don't assume one dominates the
  other without checking which restriction applies.
- **FSA (non-Markovian tasks)**: `RM/sf-fsa-vi/fsa/fsa.py` (`FiniteStateAutomaton`) and `fsa/planning.py`
  (`SFFSAValueIteration.traverse`) compose per-predicate base policies over an automaton graph — backward value
  iteration over automaton states, using GPI/metapolicy at each edge to get exit-state values. This is the
  non-tabular, general-env version of what `office_rmvi_comparison.ipynb` reimplements exactly/tabularly for the
  `Office` grid domain.

**Cross-project notebooks bridge OKB and sf-fsa-vi via `sys.path`, not installation.** Notebooks like
`OKB/notebooks/office_rmvi_comparison.ipynb` need both `OKB/rl` (for `OLS`) and `RM/sf-fsa-vi`'s `fsa`/`envs`
modules (for the `Office-v0` gym env and `FiniteStateAutomaton`) in the same process. They do this by inserting
both directories into `sys.path` at the top of the notebook (`OKB_ROOT`, `SF_FSA_VI_ROOT` computed via
`os.path.abspath(os.path.join(os.getcwd(), ...))`) rather than pip-installing either — so these notebooks are
sensitive to being run from their own directory (`OKB/notebooks/`), and the relative `..`/`../../RM/sf-fsa-vi`
paths will break if moved.

**`Office-v0` (old-gym) vs. OKB's gymnasium contract**: `RM/sf-fsa-vi/envs/grid_envs.py`'s `Office` env is registered
under the old `gym` API. Notebooks that need it inside OKB's gymnasium-based tooling wrap it
(`OfficeGymnasiumEnv`-style classes) — normalizing observations to `Box(2,)` and converting the `phi`
vector-reward from `info` into the gymnasium 5-tuple `step()` return. Expect this wrapper pattern whenever you see
`old_gym` imported alongside `gymnasium` in the same notebook.

**`OKB/rl/successor_features/okb.py`** is the deep-RL counterpart to the tabular/exact notebooks: a JAX/Flax-based
agent (batch-renorm encoders, target networks, orbax checkpointing) implementing the actual "Option Keyboard"
learning algorithm, reusing the same `OLS` class as the exact notebooks for CCS discovery, but solving each queried
weight via function approximation instead of exact policy iteration.
