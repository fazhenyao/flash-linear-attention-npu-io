#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/data/fazhenyao/cann/3_23/ascend-toolkit/set_env.sh}"
CONDA_ENV="${CONDA_ENV:-fla_dump}"
HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8787}"
DOCS_PORT="${DOCS_PORT:-8080}"

if [[ -f /data/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /data/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

if [[ -f "${CANN_SET_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_SET_ENV}"
fi

export PERF_RUN_MODE=local
export PERF_LOCAL_SCRIPT="${PERF_LOCAL_SCRIPT:-scripts/flash_gated_delta_rule.py}"
export PERF_NPU_DEVICE="${PERF_NPU_DEVICE:-2}"
export PERF_CHIP="${PERF_CHIP:-A2}"
export PERF_PROF_OUTPUT="${PERF_PROF_OUTPUT:-data/prof_gdr}"
export PERF_OP_OUTPUT="${PERF_OP_OUTPUT:-data/prof_op}"
export PERF_SOC_VERSION="${PERF_SOC_VERSION:-Ascend910B}"

stop_port() {
  local port="$1"
  local pids
  pids="$(ss -ltnp 2>/dev/null | awk -v port=":${port}" '$4 ~ port {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 1
  fi
}

stop_port "${BACKEND_PORT}"
stop_port "${DOCS_PORT}"

cd "${ROOT}/docs"
nohup python3 -m http.server "${DOCS_PORT}" --bind "${HOST}" >"${ROOT}/data/docs-server.log" 2>&1 &
echo "docs server: http://${HOST}:${DOCS_PORT}/"

cd "${ROOT}"
nohup python3 backend/app.py --host "${HOST}" --port "${BACKEND_PORT}" >"${ROOT}/data/backend-server.log" 2>&1 &
echo "backend api: http://${HOST}:${BACKEND_PORT}/io"
echo "perf runner mode: ${PERF_RUN_MODE}"
echo "python: $(command -v python3)"
echo "msopprof: $(command -v msopprof || echo 'not found')"
