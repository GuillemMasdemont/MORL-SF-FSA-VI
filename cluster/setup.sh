#!/bin/bash
# One-time (idempotent) setup on the cluster: conda env `sffsa` + the RM/sf-fsa-vi code SFOLS/ imports its envs from.
#   bash cluster/setup.sh            then   python cluster/sweep.py submit cluster/sweeps/smoke.yaml
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"

if "$CONDA" env list | grep -qE '^sffsa\s'; then
    "$CONDA" env update -n sffsa -f "$REPO/environment.yml" --prune -q
else
    "$CONDA" env create -f "$REPO/environment.yml" -q
fi

# RM/sf-fsa-vi is a gitlink (Kuric et al.'s code); SFOLS/ only needs its envs/ (lof/ and sfols/ are not used).
if [ ! -f "$REPO/RM/sf-fsa-vi/envs/grid_envs.py" ]; then
    git -C "$REPO" submodule update --init RM/sf-fsa-vi
fi

"$HOME/miniconda3/envs/sffsa/bin/python" - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
import numpy, scipy, matplotlib, cdd, gym, numba, yaml
print("sffsa ok: numpy", numpy.__version__, "scipy", scipy.__version__, "gym", gym.__version__, "numba", numba.__version__)
EOF
