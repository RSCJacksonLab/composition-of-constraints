#!/usr/bin/env bash
# Copy the exact `#PBS` lines from your current submission script into this block.
#PBS -N kras_figure4

set -euo pipefail

REPO_DIR="/home/matthew-spence/graph-ruggedness-de"
INPUT_LSBUNDLE="${INPUT_LSBUNDLE:-${1:-}}"
RESULTS_DIR="${RESULTS_DIR:-${2:-${REPO_DIR}/RESULTS}}"

if [[ -z "${INPUT_LSBUNDLE}" ]]; then
  echo "Usage: $0 /path/to/input.lsbundle [RESULTS_DIR]" >&2
  echo "   or: qsub -v INPUT_LSBUNDLE=/path/to/input.lsbundle,RESULTS_DIR=${REPO_DIR}/RESULTS $0" >&2
  exit 1
fi

cd "${REPO_DIR}"

export MPLBACKEND=Agg

python3 "${REPO_DIR}/scripts/analyze_kras_figure4_from_lsbundle.py" \
  "${INPUT_LSBUNDLE}" \
  --results-dir "${RESULTS_DIR}" \
  --overwrite
