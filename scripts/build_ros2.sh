#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

usage() {
  cat <<'EOF'
Usage: ./scripts/build_ros2.sh [profile] [colcon build arguments]

Profiles:
  --full             Build the real libtorch + CUDA Gaussian backend.
  --cpu-only         Build middleware and the diagnostic CPU renderer only.
  --with-tensorrt    Add TensorRT/SPNet support to --full.

Without a profile, the GAUSSIAN_LIC_ENABLE_* environment variables retain
their legacy behavior and default to OFF. Real dataset profiles require
--full; CPU builds are rejected at launch when rasterizer mode is requested.
EOF
}

build_profile="legacy"
with_tensorrt="false"
colcon_args=()
for arg in "$@"; do
  case "${arg}" in
    --full)
      if [[ "${build_profile}" == "cpu-only" ]]; then
        echo "error: --full and --cpu-only are mutually exclusive" >&2
        exit 2
      fi
      build_profile="full"
      ;;
    --cpu-only)
      if [[ "${build_profile}" == "full" ]]; then
        echo "error: --full and --cpu-only are mutually exclusive" >&2
        exit 2
      fi
      build_profile="cpu-only"
      ;;
    --with-tensorrt)
      with_tensorrt="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      colcon_args+=("${arg}")
      ;;
  esac
done

if [[ "${with_tensorrt}" == "true" && "${build_profile}" != "full" ]]; then
  echo "error: --with-tensorrt requires --full" >&2
  exit 2
fi

case "${build_profile}" in
  full)
    enable_torch="ON"
    enable_cuda="ON"
    if [[ "${with_tensorrt}" == "true" ]]; then
      enable_tensorrt="ON"
    else
      enable_tensorrt="OFF"
    fi
    ;;
  cpu-only)
    enable_torch="OFF"
    enable_cuda="OFF"
    enable_tensorrt="OFF"
    ;;
  *)
    enable_torch="${GAUSSIAN_LIC_ENABLE_TORCH:-OFF}"
    enable_cuda="${GAUSSIAN_LIC_ENABLE_CUDA:-OFF}"
    enable_tensorrt="${GAUSSIAN_LIC_ENABLE_TENSORRT:-OFF}"
    ;;
esac

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
cd "${ROOT_DIR}"

cmake_args=(
  --no-warn-unused-cli
  -DPython3_EXECUTABLE=/usr/bin/python3
  -DPYTHON_EXECUTABLE=/usr/bin/python3
  -DGAUSSIAN_LIC_ENABLE_TORCH="${enable_torch}"
  -DGAUSSIAN_LIC_ENABLE_CUDA="${enable_cuda}"
  -DGAUSSIAN_LIC_ENABLE_TENSORRT="${enable_tensorrt}"
)

if [[ "${enable_torch}" == "ON" || "${enable_cuda}" == "ON" || "${enable_tensorrt}" == "ON" ]]; then
  export PATH="${CUDA_HOME}/bin:/usr/local/cuda/bin:${PATH}"
  cmake_args+=(
    -DCUDAToolkit_ROOT="${CUDA_HOME}"
  )
fi

if [[ "${enable_tensorrt}" == "ON" ]]; then
  default_tensorrt_root="${HOME}/Software/TensorRT-10.9.0.34-cuda12.8"
  if [[ ! -d "${default_tensorrt_root}" ]]; then
    default_tensorrt_root="${HOME}/Software/TensorRT-8.6.1.6"
  fi
  cmake_args+=(
    -DTENSORRT_ROOT="${TENSORRT_ROOT:-${default_tensorrt_root}}"
  )
fi

if [[ "${enable_torch}" == "ON" || "${enable_cuda}" == "ON" ]]; then
  torch_dir="${TORCH_DIR:-${HOME}/Software/libtorch/share/cmake/Torch}"
  cuda_arch="${CMAKE_CUDA_ARCHITECTURES:-}"
  if [[ -z "${cuda_arch}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
  fi
  cuda_arch="${cuda_arch:-native}"
  cuda_compiler="${CMAKE_CUDA_COMPILER:-${CUDA_HOME}/bin/nvcc}"

  if [[ ! -f "${torch_dir}/TorchConfig.cmake" ]]; then
    echo "error: TorchConfig.cmake not found under ${torch_dir}; set TORCH_DIR" >&2
    exit 2
  fi
  if [[ "${enable_cuda}" == "ON" && ! -x "${cuda_compiler}" ]]; then
    echo "error: CUDA compiler not found at ${cuda_compiler}; set CUDA_HOME or CMAKE_CUDA_COMPILER" >&2
    exit 2
  fi

  cmake_args+=(
    -DGAUSSIAN_LIC_ENABLE_TORCH=ON
    -DGAUSSIAN_LIC_ENABLE_CUDA="${enable_cuda}"
    -DTorch_DIR="${torch_dir}"
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}"
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
  )
fi

colcon_command=(colcon)
if [[ -n "${GAUSSIAN_LIC_LOG_BASE:-}" ]]; then
  colcon_command+=(--log-base "${GAUSSIAN_LIC_LOG_BASE}")
fi

"${colcon_command[@]}" build --symlink-install \
  "${colcon_args[@]}" \
  --cmake-args "${cmake_args[@]}"
