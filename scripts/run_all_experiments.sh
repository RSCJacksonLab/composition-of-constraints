#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

experiments=(
  "experiments/figure-1_synthetic-nk-diffusion-scale-validation/run.sh"
  "experiments/figure-1_sparse-phylogenetic-diffusion-scale-validation/run.sh"
  "experiments/figure-2_stability-dms-ruggedness-and-edge-cutsets/run.sh"
  "experiments/figure-3_stability-dms-svd-substitution-regime-boundaries/run.sh"
  "experiments/figures-4-5_coupled-additive-constraint-composition/run.sh"
  "experiments/figure-6_sh3-folding-binding-boundary-ruggedness/run.sh"
  "experiments/figure-6_kras-ddpca-boundary-enrichment/run.sh"
  "experiments/figures-5-6_lycov-rbd-minimum-operator-and-boundaries/run.sh"
)

for rel_script in "${experiments[@]}"; do
  run_script="${ROOT}/${rel_script}"
  echo "[run-all-experiments] ${run_script#${ROOT}/}"
  bash "${run_script}"
done
