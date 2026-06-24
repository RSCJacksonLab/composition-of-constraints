#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PAPER_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
WORK_DIR="${PAPER_WORKDIR:-${SCRIPT_DIR}/work}"
SOURCE_ROOT="${SCRIPT_DIR}/source"
NOTEBOOK_DIR="${SOURCE_ROOT}/figure_notebooks_rev"

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}" "${OUTPUT_DIR}/figures" "${OUTPUT_DIR}/si_figures"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PAPER_PROJECT_ROOT="${PROJECT_ROOT}"
export PAPER_OUTPUT_DIR="${OUTPUT_DIR}"
export PAPER_WORKDIR="${WORK_DIR}"
export PAPER_PROCESSED_DIR="${PROJECT_ROOT}/data/processed"
export PAPER_DATA_FILES="${PROJECT_ROOT}/data/source_datasets"
export PYTHONPATH="${NOTEBOOK_DIR}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/experiment.py"
