#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/paper_env.sh"

PYTHON_BIN="$(paper_resolve_python "${ROOT}")"
export PYTHON="${PYTHON_BIN}"
paper_require_fitness_landscape "${PYTHON_BIN}" "${ROOT}"

analyses=(
  "analyses/figure-1_diffusion-scale-validation/notebooks/analysis.py"
  "analyses/figure-2_stability-dms-ruggedness-and-spectral-cutsets/notebooks/analysis.py"
  "analyses/figure-2_edge-energy-autocorrelation-localization/notebooks/analysis.py"
  "analyses/figure-2_node-label-permutation-nulls/notebooks/analysis.py"
  "analyses/figure-3_svd-substitution-regime-boundaries/notebooks/analysis.py"
  "analyses/figures-4-5_smooth-constraint-composition/notebooks/analysis.py"
  "analyses/figure-6_experimental-limiter-boundaries/notebooks/analysis.py"
)

for rel_script in "${analyses[@]}"; do
  analysis_script="${ROOT}/${rel_script}"
  analysis_dir="$(dirname "${analysis_script}")"
  echo "[run-all-analyses] ${analysis_script#${ROOT}/}"
  (cd "${analysis_dir}" && "${PYTHON_BIN}" analysis.py)
done
