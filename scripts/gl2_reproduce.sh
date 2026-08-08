#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -Eeuo pipefail

# Reproduce the released GL2 baseline/coupling experiments without writing into
# the source tree or accepting abnormal teardown as success.

MODE="${1:-baseline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${GL2_OUTPUT_ROOT:-/tmp/gaussian_lic2_reproduce}"
RUN_ROOT="${OUTPUT_ROOT}/${MODE}"
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

REF="${GL2_REFERENCE_TRAJECTORY:-${WS}/baseline/fastlivo2/CBD_Building_01/native_reference/cocolic_livo_reference_10hz.tum}"
NODE="${GL2_COCOLIC_NODE:-${WS}/install/cocolic/lib/cocolic/odometry_node}"
MAPBIN="${GL2_MAPPING_NODE:-${WS}/install/gaussian_lic_mapping/lib/gaussian_lic_mapping/mapping_node}"
[[ -x "${MAPBIN}" ]] || MAPBIN="${WS}/build/gaussian_lic_mapping/mapping_node"

OUTPUT_ROOT="$(realpath -m -- "${OUTPUT_ROOT}")"
RUN_ROOT="$(realpath -m -- "${RUN_ROOT}")"
if [[ "${OUTPUT_ROOT}" == "/" || "${RUN_ROOT}" != "${OUTPUT_ROOT}/"* ]]; then
  echo "refusing unsafe GL2 output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"
MAP_PID=""
MAPPER_LAST_EXIT_CODE=""

safe_remove_tree() {
  local target
  target="$(realpath -m -- "$1")"
  if [[ "${target}" == "/" || "${target}" == "${OUTPUT_ROOT}" || "${target}" != "${OUTPUT_ROOT}/"* ]]; then
    echo "refusing unsafe recursive removal outside ${OUTPUT_ROOT}: ${target}" >&2
    return 2
  fi
  rm -rf -- "${target}"
}

wait_mapper() {
  local context="$1"
  local pid="${MAP_PID}"
  local exit_code
  [[ -n "${pid}" ]] || return 0
  if wait "${pid}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  MAP_PID=""
  MAPPER_LAST_EXIT_CODE="${exit_code}"
  if [[ "${exit_code}" -ne 0 ]]; then
    echo "mapping_node exited with code ${exit_code} (${context})" >&2
    return 1
  fi
  return 0
}

cleanup_mapper() {
  if [[ -z "${MAP_PID}" ]]; then
    return
  fi
  if ! kill -0 "${MAP_PID}" 2>/dev/null; then
    wait_mapper "before requested shutdown"
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
  if [[ -n "${MAP_PID}" ]] && ! cleanup_mapper; then
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

require_path() {
  [[ -e "$1" ]] || { echo "required path is unavailable: $1" >&2; exit 2; }
}

ate() {
  local trajectory="$1"
  local report="$2"
  /usr/bin/python3 "${WS}/scripts/trajectory_compare.py" \
    --max-association-dt 0.1 --min-coverage 0.2 --min-matches 30 \
    --baseline "${REF}" --current "${trajectory}" --output "${report}"
  /usr/bin/python3 - "${report}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
translation = report["translation"]
print(
    f"ATE rmse={translation['rmse_m'] * 100:.2f}cm "
    f"mean={translation['mean_m'] * 100:.2f}cm cov={report['coverage']:.2f}"
)
PY
}

run_track_a() {
  local source_config="$1"
  local label="$2"
  local run_dir="${RUN_ROOT}/${label}"
  local log="${run_dir}/track_a.log"

  require_path "${NODE}"
  require_path "${source_config}"
  safe_remove_tree "${run_dir}"
  mkdir -p "${run_dir}"

  "${NODE}" "${source_config}" --ros-args \
    -p output_directory:="${run_dir}" >"${log}" 2>&1 &
  local track_pid=$!

  sleep 1
  if ! kill -0 "${track_pid}" 2>/dev/null; then
    local early_exit
    if wait "${track_pid}"; then early_exit=0; else early_exit=$?; fi
    echo "track A exited during startup with code ${early_exit}; see ${log}" >&2
    return 1
  fi
  if [[ -n "${MAP_PID}" ]] && ! kill -0 "${MAP_PID}" 2>/dev/null; then
    wait_mapper "immediately after track A startup" || true
    kill -TERM "${track_pid}" 2>/dev/null || true
    wait "${track_pid}" 2>/dev/null || true
    echo "mapping_node did not remain alive after track A startup" >&2
    return 1
  fi

  local track_exit_code
  if wait "${track_pid}"; then track_exit_code=0; else track_exit_code=$?; fi
  if [[ "${track_exit_code}" -ne 0 ]]; then
    echo "track A exited with code ${track_exit_code}; see ${log}" >&2
    return 1
  fi
  if [[ -n "${MAP_PID}" ]] && ! kill -0 "${MAP_PID}" 2>/dev/null; then
    wait_mapper "while track A was running" || true
    echo "mapping_node exited before track A completed" >&2
    return 1
  fi

  mapfile -t trajectories < <(find "${run_dir}" -maxdepth 1 -type f -name '*_LICO.txt' -print)
  if [[ "${#trajectories[@]}" -ne 1 ]]; then
    echo "track A did not produce exactly one LICO trajectory; see ${log}" >&2
    return 1
  fi
  TRACK_A_TRAJECTORY="${trajectories[0]}"
  export TRACK_A_TRAJECTORY
  echo "track A completed cleanly: ${TRACK_A_TRAJECTORY}"
}

start_mapper() {
  local save_render_eval="${1:-false}"
  local log="${RUN_ROOT}/mapper.log"
  require_path "${MAPBIN}"
  "${MAPBIN}" --ros-args \
    --params-file "${WS}/run_lio/config/cbd_mapper_coupled.yaml" \
    -p save_map_render_evaluation:="${save_render_eval}" \
    -p depth_completion:="${GL2_DEPTH_COMPLETION:-true}" >"${log}" 2>&1 &
  MAP_PID=$!

  for second in $(seq 1 90); do
    if ! kill -0 "${MAP_PID}" 2>/dev/null; then
      echo "mapper exited before advertising SaveMap; see ${log}" >&2
      wait_mapper "during startup" || true
      return 1
    fi
    if ros2 service list 2>/dev/null | grep -q '^/gaussian_lic/save_map$'; then
      echo "mapper ready (${second}s)"
      return 0
    fi
    sleep 1
  done
  echo "mapper readiness timeout; see ${log}" >&2
  return 1
}

require_path "${REF}"

case "${MODE}" in
  baseline)
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_full_baseline.yaml" baseline
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    ;;
  coupled)
    start_mapper false
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_gs_live_full.yaml" coupled
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    cleanup_mapper
    ;;
  degraded-baseline)
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_degraded_baseline.yaml" degraded_baseline
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    ;;
  degraded-coupled)
    start_mapper false
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_degraded_coupled.yaml" degraded_coupled
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    cleanup_mapper
    ;;
  asym-baseline)
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_asym_baseline.yaml" asym_baseline
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    ;;
  asym-se3pose)
    start_mapper false
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_asym_se3pose.yaml" asym_se3pose
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    cleanup_mapper
    ;;
  degraded-se3pose)
    # Render-SE3(论文 Camera Factor Option 2 位姿级):光度耦合关,SE3 位姿因子开。
    start_mapper false
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_degraded_se3pose.yaml" degraded_se3pose
    ate "${TRACK_A_TRAJECTORY}" "${RUN_ROOT}/ate.json"
    cleanup_mapper
    ;;
  psnr)
    MAP_OUT="${RUN_ROOT}/map"
    safe_remove_tree "${MAP_OUT}"
    mkdir -p "${MAP_OUT}"
    start_mapper true
    run_track_a "${WS}/run_lio/config/ct_odometry_lico_gs_live_full.yaml" psnr
    response="$(ros2 service call /gaussian_lic/save_map gaussian_lic_msgs/srv/SaveMap \
      "{path: '${MAP_OUT}/map.ply', include_skybox: false}")"
    grep -q 'success: true' <<<"${response}" || {
      echo "SaveMap failed: ${response}" >&2
      exit 1
    }
    cleanup_mapper
    require_path "${MAP_OUT}/renders"
    require_path "${MAP_OUT}/gt"
    /usr/bin/python3 "${WS}/scripts/eval_render_quality.py" \
      --result-dir "${MAP_OUT}" \
      --render-dir "${MAP_OUT}/renders" \
      --reference-dir "${MAP_OUT}/gt"
    ;;
  *)
    echo "usage: $0 {baseline|coupled|degraded-baseline|degraded-coupled|psnr}" >&2
    exit 2
    ;;
esac

echo "GL2 reproduction completed successfully: ${RUN_ROOT}"
