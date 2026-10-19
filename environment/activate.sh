# Source this file: source /workspace/MLiP_Project/environment/activate.sh
mlip_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$mlip_project_root/.venv-server/bin/activate" ]]; then
    echo 'Server environment missing; run environment/setup_server.sh first.' >&2
    return 1
fi
source "$mlip_project_root/.venv-server/bin/activate"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$mlip_project_root/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$mlip_project_root/.cache}"
unset mlip_project_root
