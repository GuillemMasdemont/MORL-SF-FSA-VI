#!/bin/bash
# TeX on the cluster: there is none on the nodes, so this runs on a compute node inside the official TeX Live image
# (pyxis/enroot; TeX Live 2026, pdflatex + biber), with the repository mounted and the sffsa python first on PATH.
#   bash cluster/tex.sh paper article                       # cd paper; pdflatex, biber, pdflatex, pdflatex
#   bash cluster/tex.sh notes lazy_sf_fsa_vi nobib           # no biber pass
#   bash cluster/tex.sh --run 'cd SFOLS && python plot_lazy_fsa.py'   # any command (figures with text.usetex)
# One-time: enroot import -o ~/containers/texlive.sqsh docker://texlive/texlive:latest   (on a compute node)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TEX_IMAGE:-$HOME/containers/texlive.sqsh}"
PYBIN="$HOME/miniconda3/envs/sffsa/bin"

if [ "${1:-}" = "--run" ]; then
    CMD="$2"
else
    DIR="$1"; DOC="$2"; BIB="${3:-bib}"
    TEX="pdflatex -interaction=nonstopmode -halt-on-error '$DOC'"
    FAIL="{ grep -A4 '^!' '$DOC.log'; exit 1; }"                  # show the LaTeX error of a failed pass
    CMD="cd '$DIR' && { $TEX >/dev/null || $FAIL; }"
    [ "$BIB" = "bib" ] && CMD="$CMD && biber --quiet '$DOC'"
    CMD="$CMD && { $TEX >/dev/null || $FAIL; } && { $TEX | grep -E 'Output written|Warning' || true; }"
fi
srun -p amd -c 2 --mem=6G -t 00:30:00 --container-image="$IMAGE" --container-mounts=/shared/home:/shared/home \
    --container-workdir="$REPO" bash -c "export PATH='$PYBIN':\$PATH MPLBACKEND=Agg; $CMD" 2>&1 \
    | grep -v -E '^-+$|FRIDA|^ ?[0-9:]+ up |MAINT|start a month|^$|^srun: job|pyxis'
