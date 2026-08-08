#!/usr/bin/env bash
# LiDAR_Degenerate 真退化 A/B:两臂同带在线 mapper(PSNR 公平),唯一变量 = enable_render_se3_pose。
# 用法: degen_ab.sh {baseline|se3pose}
set -Eeo pipefail
MODE="${1:?usage: degen_ab.sh baseline-or-se3pose}"
WS=/home/frank/gaussian_lic_ros2
OUT=/tmp/gl2_degen/${MODE}
rm -rf "${OUT}"; mkdir -p "${OUT}/map"

set +u
source /opt/ros/jazzy/setup.bash
source "${WS}/install/setup.bash"
set -u
export LD_LIBRARY_PATH="${HOME}/Software/libtorch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments}"

MAPBIN="${WS}/install/gaussian_lic_mapping/lib/gaussian_lic_mapping/mapping_node"
[[ -x "${MAPBIN}" ]] || MAPBIN="${WS}/build/gaussian_lic_mapping/mapping_node"
NODE="${WS}/install/cocolic/lib/cocolic/odometry_node"

"${MAPBIN}" --ros-args \
  --params-file "${WS}/run_lio/config/cbd_mapper_coupled.yaml" \
  -p save_map_render_evaluation:=true \
  -p depth_completion:=false > "${OUT}/mapper.log" 2>&1 &
MAP_PID=$!
trap 'kill -9 ${MAP_PID} 2>/dev/null || true' EXIT

for s in $(seq 1 90); do
  kill -0 "${MAP_PID}" 2>/dev/null || { echo "mapper died at startup"; exit 1; }
  ros2 service list 2>/dev/null | grep -q '^/gaussian_lic/save_map$' && break
  sleep 1
done
echo "mapper ready"

set +e
stdbuf -oL -eL "${NODE}" "${WS}/run_lio/config/ct_odometry_degen_noPnP_${MODE}.yaml" --ros-args \
  -p output_directory:="${OUT}" > "${OUT}/track.log" 2>&1
TRACK_EXIT=$?
set -e
echo "track done exit=${TRACK_EXIT}"
if [[ ${TRACK_EXIT} -ne 0 ]]; then tail -15 "${OUT}/track.log"; exit ${TRACK_EXIT}; fi

ros2 service call /gaussian_lic/save_map gaussian_lic_msgs/srv/SaveMap \
  "{path: '${OUT}/map/map.ply', include_skybox: false}" | grep -q 'success: true' \
  || { echo "SaveMap failed"; exit 1; }
kill -TERM "${MAP_PID}" 2>/dev/null || true; wait "${MAP_PID}" 2>/dev/null || true

if [[ -d "${OUT}/map/renders" && -d "${OUT}/map/gt" ]]; then
  /usr/bin/python3 "${WS}/scripts/eval_render_quality.py" \
    --result-dir "${OUT}/map" --render-dir "${OUT}/map/renders" \
    --reference-dir "${OUT}/map/gt" > "${OUT}/psnr.txt" 2>&1 || true
  tail -5 "${OUT}/psnr.txt"
fi
echo "DEGEN_${MODE}_DONE"
