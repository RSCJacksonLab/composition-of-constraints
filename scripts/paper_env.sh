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
