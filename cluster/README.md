# Running SFOLS/ experiments on the FRIDA cluster

Everything here is CPU-only, tabular, single-threaded numpy (see HANDOFF.md §9). Commands are run from the
repository root on the login node; the Python is the conda env `sffsa` (`~/miniconda3/envs/sffsa/bin/python`,
below `PY`).

## Setup (once, idempotent)

```bash
bash cluster/setup.sh                                  # conda env `sffsa` from environment.yml + RM/sf-fsa-vi
PY=~/miniconda3/envs/sffsa/bin/python
$PY cluster/sweep.py submit cluster/sweeps/smoke.yaml  # every entry point on a tiny budget, ~1 min
$PY cluster/sweep.py status                            # all sweeps; expect "5 done"
```

`environment.yml` follows the laptop's `okb` pins (python 3.10, numpy 1.26.4, scipy 1.12.0, pycddlib 2.1.6, gym
0.23.1, mo-gymnasium 1.3.1) without jax/torch, plus numba, pyyaml and pymupdf (PDF page previews).

**TeX.** The nodes have none, so `cluster/tex.sh` runs on a compute node inside the official TeX Live image (TeX Live
2026, pdflatex + biber, through pyxis/enroot), with the repository mounted and the `sffsa` python first on PATH:

```bash
enroot import -o ~/containers/texlive.sqsh docker://texlive/texlive:latest     # once, on a compute node (srun)
bash cluster/tex.sh paper article                                             # paper/article.pdf
bash cluster/tex.sh notes lazy_sf_fsa_vi nobib                                # a note without biber
bash cluster/tex.sh --run 'cd SFOLS && python plot_lazy_fsa.py'               # figures with text.usetex
```

The output is identical to the laptop's pdflatex build (TeX Live 2025): same pages and line breaks.

## The cluster

| partition | node | cores | time limit | use |
|---|---|---|---|---|
| `amd` (default here) | api, AMD EPYC 9684X | 384 threads | 2 days | CPU jobs (the cluster recommends it) |
| `frida` | GPU nodes | 112–256 | 7 days | runs longer than 2 days |
| `dev` | GPU nodes | 112–256 | 12 h | quick interactive tests (`srun -p dev ...`) |

- **At most 8 running jobs per user** (QOS `normal`). Do not rely on wide arrays: pack work into jobs instead, either
  with the scripts' own worker pools (`slurm.cpus`) or with `slurm.pack` (several runs per job, see below).
- `os.cpu_count()` reports the whole node (e.g. 112) inside a 2-core allocation. Always pass the worker count
  (`{cpus}` in the command); `q_ccs_stochastic.py run` now defaults to the allocated cores.
- Measured: ~59k vector SF-Q steps/s per core on `api` with the Python learners (one SFOLS learn of 1M steps in 17 s),
  5–7M steps/s with the numba learners of `SFOLS/mf.py` (used by `q_lazy_fsa.py`), oracle checks included.

## Sweeps

A sweep is a YAML spec (`cluster/sweeps/*.yaml`):

```yaml
name: office_ccs_mf
command: python q_ccs_stochastic.py run {slip} 1 0 1   # run from SFOLS/; {lowercase} keys are filled in
grid:                       # cartesian product
  slip: [0.05]              # lowercase -> {slip} in the command
  ARMS: [standard, certify] # UPPERCASE -> environment variable
  QSEED0: [0, 1, 2]
runs:                       # optional explicit combinations, appended to the grid (a `name` key names the run dir)
  - {name: extra, slip: 0.1, ARMS: certify, QSEED0: 0}
env:                        # fixed environment variables for every run
  QBASISWARM: 1
slurm:
  partition: amd            # defaults: amd, cpus 1, mem 4G, time 12:00:00, max_parallel 8, pack 1
  cpus: 1                   # per run; available as {cpus}
  pack: 4                   # runs per job: the job asks for pack*cpus cores and pack*mem, and runs them concurrently
  mem: 4G                   # per run
  time: "1-00:00:00"
  max_parallel: 8           # array throttle (%8)
  extra: ["--exclusive"]    # optional raw #SBATCH lines
```

```bash
$PY cluster/sweep.py submit SPEC [--dry-run] [--live] [--name NAME]
$PY cluster/sweep.py status [SWEEP_DIR]               # per run: done / failed / lost / running / queued, + log tails
$PY cluster/sweep.py resubmit SWEEP_DIR               # failed and lost runs (--all-unfinished: also never started)
$PY cluster/sweep.py collect SWEEP_DIR                # SWEEP_DIR/index.csv: params, state, time, output files
$PY cluster/sweep.py merge SWEEP_DIR --into DIR       # gather outputs; same-name list pickles are concatenated
$PY cluster/sweep.py local SWEEP_DIR INDEX            # run one run in the foreground (debugging, no SLURM)
```

Each sweep is `SFOLS/runs/<name>-<timestamp>/` (git-ignored):

- `spec.yaml`, `runs.json` (expanded runs), `git.txt` (commit and status), `git.diff` (uncommitted changes, if any);
- `code/` — a **snapshot** of `SFOLS/**/*.py`, so editing code while jobs wait in the queue does not change them
  (`code/RM` and `code/SFOLS/results` are symlinks, so inputs resolve as usual). `--live` runs the working tree;
- `r/<id>/log.txt` (stdout + stderr), `r/<id>/status.json`, `r/<id>/out/` — every file the script writes.

**Outputs.** The scripts write to `$SF_OUT` when it is set (the sweep sets it to `r/<id>/out`) and to
`SFOLS/results/` otherwise, so running them by hand behaves as before. Inputs (stored SFOLS runs `run_*.pkl`,
`taskopt_*.pkl`, `office_ols_weights_*.npy`) are still read from `SFOLS/results/`; `certlib` loads look in `$SF_OUT`
first. This covers `q_ccs_stochastic.py`, `q_fsa_stochastic.py`, `q_lazy_fsa.py`, `q_speedup_ablation.py` and
everything under `certlib/`; the older `gpi_basis/run_*.py` scripts still write to `SFOLS/results/gpi_basis/`.

**Seeds.** `q_ccs_stochastic.py run <slip> <seeds>` runs seeds `QSEED0 .. QSEED0+seeds-1` (default `QSEED0=0`), so
one (arm, seed) per run is `ARMS=<arm> QSEED0=<seed> ... run <slip> 1`. `merge` concatenates the per-run lists into
the single `office_q_runs_<tag>.pkl` the plot scripts read.

**Promoting results.** Sweep outputs stay in `SFOLS/runs/`. When a sweep is final, `merge --into SFOLS/results`
(or `SFOLS/results/certlib` for `certlib`) puts the files where the plot scripts and HANDOFF expect them; `merge`
refuses to overwrite without `--force` (those files are tracked in git, so an overwrite shows in `git diff`).

## Specs in `cluster/sweeps/`

| spec | what | size |
|---|---|---|
| `smoke.yaml` | every entry point, tiny budgets | 5 runs, ~1 min |
| `lazyfsa_model.yaml` | E8: Lazy SF-FSA-VI with a model (Figure 3 inputs) | 3 runs, ≤ 1 h |
| `office_ccs_mf.yaml` | E5: model-free CCS on slippery Office, 4 arms × 10 seeds | 40 runs (2 jobs), ~1 h each |
| `ds_ccs_mf.yaml` | E5: model-free CCS on the double slit (wind ±1), 4 arms × 10 seeds | 40 runs (2 jobs), 10 min each |
| `lazy_mf.yaml` | model-free Lazy SF-FSA-VI vs SFOLS, Q-learning growth and flat Q-learning (`q_lazy_fsa.py`), windy and Office p ∈ {0.1, 0.3}, with and without inherited counts, 10 seeds | 60 runs (30 jobs), 10–30 min each |
| `lazy_mf_ds.yaml` | the same on the double slit | 20 runs (4 jobs) |
| `lazy_mf_pilot*.yaml` | the pilots that fixed the settings of `lazy_mf.yaml` (see HANDOFF) | small |

## Pitfalls

- `q_ccs_stochastic.py` catches per-job exceptions and still exits 0: a run marked `done` can contain `FAILED` lines.
  Check with `grep -l FAILED SFOLS/runs/<sweep>/r/*/log.txt`.
- A run killed by the time limit shows as `lost`. `resubmit` reuses the sweep's `job.sbatch`, so to give it more time
  edit `#SBATCH --time` (or `--partition`) in that file first.
- The gym deprecation banner at the top of every log is harmless.
