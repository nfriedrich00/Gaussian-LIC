#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -Eeuo pipefail

# Replay an existing mapper-contract bag through the native CUDA mapper, save
# the resulting Gaussian map, and compute render metrics. All outputs default
# to /tmp so validation never mutates checked-in experiment artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "${SCRIPT_DIR}/.." && pwd)"
BAG="${GL2_MAPPER_BAG:-${WS}/run_lio/data/CBD_Building_01_frontend_raw_offset_time_full_mapper_contract}"
PARAMS="${GL2_MAPPER_PARAMS:-${WS}/run_lio/config/cbd_mapper.yaml}"
OUTDIR="${GL2_STEP1_OUTPUT:-/tmp/gaussian_lic2_step1_psnr}"
MAPBIN="${GL2_MAPPING_NODE:-${WS}/install/gaussian_lic_mapping/lib/gaussian_lic_mapping/mapping_node}"
[[ -x "${MAPBIN}" ]] || MAPBIN="${WS}/build/gaussian_lic_mapping/mapping_node"
OUTDIR="$(realpath -m -- "${OUTDIR}")"
case "${OUTDIR}" in
  /|/tmp|/home|"${HOME}"|"${WS}")
    echo "refusing unsafe GL2 output directory: ${OUTDIR}" >&2
    exit 2
    ;;
esac
if [[ "${OUTDIR#/}" != */* ]]; then
  echo "GL2 output directory must have at least two path components: ${OUTDIR}" >&2
  exit 2
fi
LOG="${OUTDIR}/mapping_node.log"
LIBTORCH_ROOT="${LIBTORCH_ROOT:-${HOME}/Software/libtorch}"

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${WS}/install/setup.bash" ]]; then
  source "${WS}/install/setup.bash"
fi
set -u

if [[ -d "${LIBTORCH_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${LIBTORCH_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments}"

[[ -e "${BAG}" ]] || { echo "mapper-contract bag is unavailable: ${BAG}" >&2; exit 2; }
[[ -f "${PARAMS}" ]] || { echo "mapper parameter file is unavailable: ${PARAMS}" >&2; exit 2; }
[[ -x "${MAPBIN}" ]] || { echo "mapping_node is unavailable: ${MAPBIN}" >&2; exit 2; }

MAP_PID=""
MAPPER_LAST_EXIT_CODE=""

safe_remove_output() {
  local target
  target="$(realpath -m -- "$1")"
  if [[ "${target}" != "${OUTDIR}" || "${target}" == "/" ]]; then
    echo "refusing unsafe recursive removal: ${target}" >&2
    return 2
  fi
  rm -rf -- "${target}"
}

wait_mapper() {
  local context="$1"
  local pid="${MAP_PID}"
  local exit_code
  [[ -n "${pid}" ]] || return 0
  if wait "${pid}"; then exit_code=0; else exit_code=$?; fi
  MAP_PID=""
  MAPPER_LAST_EXIT_CODE="${exit_code}"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "mapping_node exited with code ${exit_code} (${context})" >&2
    return 1
  fi
  return 0
}

require_mapper_alive() {
  local context="$1"
  if [[ -n "${MAP_PID}" ]] && kill -0 "${MAP_PID}" 2>/dev/null; then
    return 0
  fi
  wait_mapper "${context}" || true
  echo "mapping_node is not alive (${context})" >&2
  return 1
}

cleanup() {
  if [[ -z "${MAP_PID}" ]]; then
    return 0
  fi
  if ! kill -0 "${MAP_PID}" 2>/dev/null; then
    wait_mapper "before requested shutdown" || true
    echo "mapping_node exited before shutdown was requested" >&2
    return 1
  fi
  kill -INT "${MAP_PID}" 2>/dev/null || true
  for _ in $(seq 1 50); do
    if ! kill -0 "${MAP_PID}" 2>/dev/null; then
      wait_mapper "requested SIGINT shutdown"
      return
    fi
    sleep 0.1
  done
  kill -TERM "${MAP_PID}" 2>/dev/null || true
  wait_mapper "forced SIGTERM shutdown"
}

cleanup_on_exit() {
  local original_status=$?
  local cleanup_failed=0
  trap - EXIT INT TERM
  if [[ -n "${MAP_PID}" ]] && ! cleanup; then
    cleanup_failed=1
  fi
  if [[ "${original_status}" -eq 0 && "${cleanup_failed}" -ne 0 ]]; then
    original_status=1
  fi
  exit "${original_status}"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

safe_remove_output "${OUTDIR}"
mkdir -p "${OUTDIR}"

echo "[1/5] starting native mapping_node"
"${MAPBIN}" --ros-args --params-file "${PARAMS}" \
  -p save_map_render_evaluation:=true >"${LOG}" 2>&1 &
MAP_PID=$!

sleep 1
if ! kill -0 "${MAP_PID}" 2>/dev/null; then
  echo "mapping_node exited during the initial startup check" >&2
  tail -50 "${LOG}" >&2
  wait_mapper "initial startup" || true
  exit 1
fi

echo "[2/5] waiting for /gaussian_lic/save_map"
ready=false
for second in $(seq 1 90); do
  if ! kill -0 "${MAP_PID}" 2>/dev/null; then
    echo "mapping_node exited during startup" >&2
    tail -50 "${LOG}" >&2
    wait_mapper "service startup" || true
    exit 1
  fi
  if ros2 service list 2>/dev/null | grep -q '^/gaussian_lic/save_map$'; then
    echo "mapper ready (${second}s)"
    ready=true
    break
  fi
  sleep 1
done
[[ "${ready}" == true ]] || { echo "mapper readiness timeout; see ${LOG}" >&2; exit 1; }

echo "[3/5] replaying mapper-contract bag"
ros2 bag play "${BAG}" --rate "${GL2_PLAYBACK_RATE:-0.5}" \
  --disable-keyboard-controls
require_mapper_alive "after bag replay"

echo "[4/5] draining the mapping queue"
previous=-1
stable=0
for second in $(seq 1 90); do
  require_mapper_alive "while draining the mapping queue"
  status="$(timeout 5 ros2 topic echo /gaussian_lic/status \
    gaussian_lic_msgs/msg/MappingStatus --once 2>/dev/null || true)"
  frames="$(awk '/^num_mapping_frames:/ {print $2; exit}' <<<"${status}")"
  optimizations="$(awk '/^gaussian_optimization_count:/ {print $2; exit}' <<<"${status}")"
  echo "drain=${second}s mapping_frames=${frames:-?} optimization_count=${optimizations:-?}"
  if [[ -n "${frames}" && "${frames}" == "${previous}" ]]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  if [[ "${stable}" -ge 4 ]]; then
    break
  fi
  previous="${frames}"
  sleep 1
done
[[ "${stable}" -ge 4 ]] || { echo "mapping queue did not drain; see ${LOG}" >&2; exit 1; }

echo "[5/5] saving map and evaluating renders"
require_mapper_alive "before SaveMap"
response="$(timeout 300 ros2 service call /gaussian_lic/save_map \
  gaussian_lic_msgs/srv/SaveMap \
  "{path: '${OUTDIR}/map.ply', include_skybox: false}")"
grep -q 'success: true' <<<"${response}" || {
  echo "SaveMap failed: ${response}" >&2
  exit 1
}

[[ -f "${OUTDIR}/map.ply" ]] || { echo "SaveMap did not create map.ply" >&2; exit 1; }
[[ -d "${OUTDIR}/renders" && -d "${OUTDIR}/gt" ]] || {
  echo "render evaluation directories were not produced; see ${LOG}" >&2
  exit 1
}

/usr/bin/python3 "${WS}/scripts/eval_render_quality.py" \
  --result-dir "${OUTDIR}" \
  --render-dir "${OUTDIR}/renders" \
  --reference-dir "${OUTDIR}/gt"

cleanup
echo "GL2 mapper-contract PSNR validation completed: ${OUTDIR}"
