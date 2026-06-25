#!/usr/bin/env bash

paper_resolve_python() {
  local project_root="$1"

  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "${PYTHON}"
  elif [[ -x "${project_root}/.venv/bin/python" ]]; then
    printf '%s\n' "${project_root}/.venv/bin/python"
  else
    printf '%s\n' "python3"
  fi
}

paper_require_fitness_landscape() {
  local python_bin="$1"
  local project_root="$2"

  if "${python_bin}" - <<'PY' >/dev/null 2>&1
import fitness_landscape
PY
  then
    return 0
  fi

  cat >&2 <<EOF
[paper-env] Required Python module "fitness_landscape" is not available for:
  ${python_bin}

Create the reproduction environment from the repository root:
  cd ${project_root}
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

If dependencies are installed in another environment, rerun with:
  PYTHON=/path/to/python bash scripts/run_all_experiments.sh
EOF
  exit 1
}

paper_env_find_aligned_alisim_fastas() {
  local search_dir="$1"
  find -L "${search_dir}" -type f \
    \( -iname '*.fa' -o -iname '*.fasta' -o -iname '*.fas' -o -iname '*.faa' \) \
    ! -iname '*unaligned*' \
    | LC_ALL=C sort
}

paper_env_resolve_alisim_results_dir() {
  local project_root="$1"
  local -a candidates=()
  if [[ -n "${PAPER_ALISIM_RESULTS:-}" ]]; then
    candidates+=("${PAPER_ALISIM_RESULTS}")
  fi
  candidates+=(
    "${project_root}/data/alisim_results"
    "${project_root}/data/raw/alisim_results"
    "${project_root}/alisim_results"
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
    echo "[paper-env] Could not locate aligned AliSim FASTA files."
    echo "[paper-env] Checked:"
    printf '  - %s\n' "${checked[@]}"
    echo
    echo "Set PAPER_ALISIM_RESULTS=/path/to/alisim_results or unpack the data bundle at data/alisim_results."
  } >&2
  return 1
}

paper_env_replicate_number() {
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

paper_env_prepare_alisim_compat_dir() {
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
    rep_num="$(paper_env_replicate_number "${fasta_path}" "${count}")"
    link_name="replicate_${rep_num}.fa"
    if [[ -e "${tmp_dir}/${link_name}" || -L "${tmp_dir}/${link_name}" ]]; then
      link_name="replicate_$(printf '%03d' "${count}").fa"
    fi
    ln -s "${fasta_path}" "${tmp_dir}/${link_name}"
  done < <(paper_env_find_aligned_alisim_fastas "${actual_dir}")

  if [[ "${count}" -eq 0 ]]; then
    rm -rf "${tmp_dir}"
    echo "[paper-env] No aligned AliSim FASTA files found in ${actual_dir}" >&2
    return 1
  fi

  if [[ -L "${compat_dir}" || -f "${compat_dir}" ]]; then
    rm -f "${compat_dir}"
  elif [[ -d "${compat_dir}" ]]; then
    rm -rf "${compat_dir}"
  fi
  mv "${tmp_dir}" "${compat_dir}"

  echo "[paper-env] Prepared ${count} aligned AliSim FASTA links in ${compat_dir} from ${actual_dir}"
}

paper_prepare_sparse_alisim_compat_dir() {
  local project_root="$1"
  local actual_dir
  local compat_dir

  actual_dir="$(paper_env_resolve_alisim_results_dir "${project_root}")"
  export PAPER_ALISIM_RESULTS="${actual_dir}"
  compat_dir="${project_root}/experiments/figure-1_sparse-phylogenetic-diffusion-scale-validation/source/alisim_results"
  paper_env_prepare_alisim_compat_dir "${actual_dir}" "${compat_dir}"
}
