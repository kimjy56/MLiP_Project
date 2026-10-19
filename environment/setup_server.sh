#!/usr/bin/env bash
# Linux x86_64 setup only. Does not train models or create submit.zip.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
uv python install 3.11.17 --no-bin
uv venv --python 3.11.17 "$project_root/.venv-server"
uv pip install --python "$project_root/.venv-server/bin/python" \
  -r "$project_root/environment/server-requirements.txt"
uv pip check --python "$project_root/.venv-server/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_root/.cache/matplotlib}"
"$project_root/.venv-server/bin/python" "$project_root/environment/check_environment.py" \
  --output "$project_root/reports/server_environment.json"
