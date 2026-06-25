#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/paper_env.sh"
OUTPUT_DIR="${PAPER_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
WORK_DIR="${PAPER_WORKDIR:-${SCRIPT_DIR}/work}"
SOURCE_ROOT="${SCRIPT_DIR}/source"
NOTEBOOK_DIR="${SOURCE_ROOT}/figure_notebooks_rev"

paper_find_aligned_alisim_fastas() {
  local search_dir="$1"
  find -L "${search_dir}" -type f \
    \( -iname '*.fa' -o -iname '*.fasta' -o -iname '*.fas' -o -iname '*.faa' \) \
    ! -iname '*unaligned*' \
    | LC_ALL=C sort
}

paper_resolve_alisim_results_dir() {
  local -a candidates=()
  if [[ -n "${PAPER_ALISIM_RESULTS:-}" ]]; then
    candidates+=("${PAPER_ALISIM_RESULTS}")
  fi
  candidates+=(
    "${PROJECT_ROOT}/data/alisim_results"
    "${PROJECT_ROOT}/data/raw/alisim_results"
    "${PROJECT_ROOT}/alisim_results"
  )

  local candidate
  local first_match
  local -a checked=()
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    checked+=("${candidate}")
    [[ -d "${candidate}" ]] || continue
    first_match="$(find -L "${candidate}" -type f \
      \( -iname '*.fa' -o -iname '*.fasta' -o -iname '*.fas' -o -iname '*.faa' \) \
      ! -iname '*unaligned*' -print -quit 2>/dev/null || true)"
    if [[ -n "${first_match}" ]]; then
      cd "${candidate}" && pwd -P
      return 0
    fi
  done

  {
    echo "[paper-exp] Could not locate aligned AliSim FASTA files."
    echo "[paper-exp] Checked:"
    printf '  - %s\n' "${checked[@]}"
    echo
    echo "Set PAPER_ALISIM_RESULTS=/path/to/alisim_results or unpack the data bundle at data/alisim_results."
  } >&2
  return 1
}

paper_replicate_number() {
  local fasta_path="$1"
  local fallback_index="$2"
  local base
  base="$(basename "${fasta_path}")"

  if [[ "${base}" =~ ([Rr][Ee][Pp][Ll][Ii][Cc][Aa][Tt][Ee]|[Rr][Ee][Pp]|[Ss][Ii][Mm])[_-]?0*([0-9]+) ]]; then
    printf '%03d\n' "$((10#${BASH_REMATCH[2]}))"
  elif [[ "${base}" =~ ([0-9]+) ]]; then
    printf '%03d\n' "$((10#${BASH_REMATCH[1]}))"
  else
    printf '%03d\n' "${fallback_index}"
  fi
}

paper_prepare_alisim_compat_dir() {
  local actual_dir="$1"
  local compat_dir="$2"
  local tmp_dir="${compat_dir}.tmp.$$"
  local count=0
  local fasta_path
  local rep_num
  local link_name

  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"

  while IFS= read -r fasta_path; do
    count=$((count + 1))
    rep_num="$(paper_replicate_number "${fasta_path}" "${count}")"
    link_name="replicate_${rep_num}.fa"
    if [[ -e "${tmp_dir}/${link_name}" || -L "${tmp_dir}/${link_name}" ]]; then
      link_name="replicate_$(printf '%03d' "${count}").fa"
    fi
    ln -s "${fasta_path}" "${tmp_dir}/${link_name}"
  done < <(paper_find_aligned_alisim_fastas "${actual_dir}")

  if [[ "${count}" -eq 0 ]]; then
    rm -rf "${tmp_dir}"
    echo "[paper-exp] No aligned AliSim FASTA files found in ${actual_dir}" >&2
    return 1
  fi

  if [[ -L "${compat_dir}" || -f "${compat_dir}" ]]; then
    rm -f "${compat_dir}"
  elif [[ -d "${compat_dir}" ]]; then
    rm -rf "${compat_dir}"
  fi
  mv "${tmp_dir}" "${compat_dir}"

  echo "[paper-exp] Prepared ${count} aligned AliSim FASTA links in ${compat_dir} from ${actual_dir}"
}

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
export PAPER_DATA_FILES="${PAPER_DATA_FILES:-${PROJECT_ROOT}/data/source_datasets}"
export PAPER_ALISIM_RESULTS
PAPER_ALISIM_RESULTS="$(paper_resolve_alisim_results_dir)"
export PAPER_ALISIM_RESULTS
export PYTHONPATH="${NOTEBOOK_DIR}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

paper_prepare_alisim_compat_dir "${PAPER_ALISIM_RESULTS}" "${SOURCE_ROOT}/alisim_results"

PYTHON_BIN="$(paper_resolve_python "${PROJECT_ROOT}")"
paper_require_fitness_landscape "${PYTHON_BIN}" "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/experiment.py"
