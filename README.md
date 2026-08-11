# Gaussian-LIC2 ROS2 Jazzy

Native ROS2 Jazzy engineering port of
[Gaussian-LIC / Gaussian-LIC2](https://github.com/APRIL-ZJU/Gaussian-LIC) for
LiDAR-Inertial-Camera Gaussian Splatting SLAM.

This repository is not a ROS1 bridge wrapper. It keeps the ROS2 middleware,
the ported Coco-LIC front-end, the mapper contract, the CUDA/Torch Gaussian
mapping path, offline artifact tooling, and validation scripts in one
workspace so each part can be ported and tested independently.

## Forked / Ported From

This work is a ROS2 engineering port based on
[`APRIL-ZJU/Gaussian-LIC`](https://github.com/APRIL-ZJU/Gaussian-LIC). The
original Gaussian-LIC/Gaussian-LIC2 algorithms, paper lineage, and upstream
implementation credit belong to the APRIL-ZJU authors. This repository focuses
on the ROS2 Jazzy port, middleware contracts, CUDA/Torch integration, validation
tooling, and reproducibility packaging around that upstream work.

The upstream authors have been informed about this ROS2 port.


## Current Status

The public tree is an executable ROS2 porting checkpoint, not a packaged
one-command dataset release. Source, configs, scripts, CI checks, and validation
reports are versioned in git. Large datasets, generated rosbag2 artifacts,
render outputs, point clouds, TensorRT engines, and long-form experiment bundles
must be supplied separately.

Highlights:

- ROS2 Jazzy workspace with native message, launch, frontend-adapter, the
  ported Coco-LIC front-end, mapping, and offline tooling packages.
- Mapper input contract for `/points_for_gs`, `/pose_for_gs`,
  `/image_for_gs`, `/camera_info_for_gs`, `/depth_for_gs`, and `/imu_for_gs`.
- Optional CUDA/libtorch Gaussian mapping path with rendered feedback for
  closed-loop tracking experiments.
- Camera Factor Option 2 pose-level render feedback inside the Coco-LIC
  continuous-time factor graph, alongside the earlier patch-photometric
  coupling (see the section below).
- Dataset profiles and conversion scripts for FAST-LIVO, FAST-LIVO2, M2DGR,
  MCD, and R3LIVE style inputs.
- Validation reports kept under `docs/` for strict parity, paper-completion
  audit, GL2 closed-loop evidence, and known upstream/reference limitations.

## Quick Start

The CPU-only profile builds the native ROS2 middleware and diagnostic renderer;
it does not require CUDA or libtorch:

```bash
git clone https://github.com/KaiFeng-Frank/Gaussian-LIC2-ROS2-Jazzy.git
cd Gaussian-LIC2-ROS2-Jazzy

source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install -y ros-dev-tools ripgrep
# Initialize rosdep only when this machine has no default source list yet.
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths src --ignore-src -r -y
./scripts/build_ros2.sh --cpu-only
source install/setup.bash
./scripts/smoke_test.sh --tf
```

This smoke test checks the ROS2 nodes, topics, services, diagnostic CPU preview,
and TF output. It does not claim to exercise the CUDA Gaussian rasterizer.

For the full Gaussian backend, first install a CUDA 12.x toolkit and a
CUDA-enabled libtorch 2.x distribution that are compatible with each other.
Then fetch the upstream LPIPS asset, point the build at those installations,
and run the CUDA probes plus the CUDA-device smoke test:

```bash
./scripts/fetch_upstreams.sh
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_DIR="$HOME/Software/libtorch/share/cmake/Torch"
test -x "$CUDA_HOME/bin/nvcc"
test -f "$TORCH_DIR/TorchConfig.cmake"
./scripts/build_ros2.sh --full --cmake-clean-cache
source install/setup.bash
ros2 run gaussian_lic_mapping rasterizer_probe
ros2 run gaussian_lic_mapping torch_cuda_backend_probe
./scripts/smoke_test.sh --torch --torch-device cuda --render-mode rasterizer --tf
```

The final three commands exercise the CUDA rasterizer and CUDA Gaussian backend,
then validate their ROS2 topic/service integration. Override `CUDA_HOME` and
`TORCH_DIR` when the toolkit or libtorch is installed elsewhere.

Use `./scripts/build_ros2.sh --full --with-tensorrt` for any dataset profile
whose `depth_completion` setting is `true`. The launch file accepts
`depth_completion_engine_path:=...`; when that value is empty it checks
`GAUSSIAN_LIC_SPNET_ENGINE`, then the matching
`~/Software/TensorRT-engines/spnet_<height>_<width>_fp16.engine`. A requested
SPNet path now fails at startup if TensorRT or the engine is unavailable instead
of silently running a different sparse-depth pipeline. Explicitly set
`depth_completion:=false` when that fallback is intentional.

On the verified local machine, both `spnet_512_640_fp16.engine` and
`spnet_480_640_fp16.engine` already exist in that directory. TensorRT plans are
GPU/runtime-specific; see [`docs/SPNET_RUNTIME_STATUS.md`](docs/SPNET_RUNTIME_STATUS.md)
for their hashes and the local generation commands instead of copying a plan to
different hardware.

The real dataset profiles also retain the upstream end-of-run visual-quality
stage. The bundled `lpips_alex.pt` is installed with the mapper and may be
overridden with `lpips_model_path:=...` or `GAUSSIAN_LIC_LPIPS_MODEL`. Final
output contains both the ROS2 `renders/` name used by the validation scripts and
an upstream-compatible `render/` alias, plus `gt/`, `render_depth/`, and the
metrics manifest. Evaluation-owned directories are replaced on each save so
stale frames cannot contaminate a result.

CI and middleware-only development use
`./scripts/build_ros2.sh --cpu-only`; attempting to launch the real rasterizer
from that build fails immediately with an actionable rebuild message.

The mapper's ROS1/ROS2 algorithm boundary, including SPNet sampling, CUDA
optimization, exposure, density-control opt-ins, final evaluation, and
inactivity finalization, is documented in
[`docs/GAUSSIAN_MAPPER_PARITY.md`](docs/GAUSSIAN_MAPPER_PARITY.md).

Run the workspace test suite:

```bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

For real datasets, place bags and generated artifacts outside git and point the
provided scripts/configs at those local paths.

## Render Feedback (Camera Factor Option 2)

The Coco-LIC front-end can consume the mapper's `RenderedFeedback` stream live
(`gs_live_publish` with lockstep pacing) and add one pose-level factor per
image frame: a fixed-reference warp Gauss-Newton alignment of the current
image against the latest rendered keyframe (rendered image, observed depth,
source pose), whose converged pose enters the continuous-time factor graph as
an absolute `IMUPoseFactorNURBS` measurement. The coupling is controlled by
`enable_render_se3_pose` and the `render_se3_*` keys; per-exit-path gate
counters make a silently inert configuration visible.

A/B harnesses:

```bash
./scripts/gl2_reproduce.sh degraded-baseline   # weakened-LiDAR pair on CBD
./scripts/gl2_reproduce.sh degraded-se3pose
./scripts/gl2_reproduce.sh asym-baseline       # transient good->bad->good window
./scripts/gl2_reproduce.sh asym-se3pose
./scripts/degen_ab.sh baseline                 # FAST-LIVO LiDAR_Degenerate pair
./scripts/degen_ab.sh se3pose
```

Measured boundary across these testbeds (repeated runs, not single trials):
the factor stays active on 95%+ of frames yet produces no measurable accuracy
benefit in any tested configuration — healthy, weakened-LiDAR, transient
degradation, or true geometric degeneracy with or without competing visual
anchors. On the unanchored degenerate testbed both arms scatter wildly across
repeats (loop-closure drift 9-26% baseline, 10-50% with the factor), so
single-run differences there are noise; an earlier commit message describing
a shape-recovery effect predates this repetition study and is superseded.

## Evidence Snapshot

The validation method is evidence-driven: compare against valid upstream
references or ground truth where available, and explicitly mark datasets whose
archived upstream references are defective.

| Evidence surface | Public report |
|---|---|
| Strict multi-dataset parity matrix | [`docs/strict_parity_matrix_report.md`](docs/strict_parity_matrix_report.md) |
| Paper-completion audit | [`docs/paper_completion_report.md`](docs/paper_completion_report.md) |
| GL2 closed-loop tracking and 3DGS mapping results | [`docs/GL2_RESULTS.md`](docs/GL2_RESULTS.md) |
| Input data audit contract | [`docs/strict_data_status.md`](docs/strict_data_status.md) |
| Historical rejected diagnostics | [`docs/HISTORICAL_DIAGNOSTICS.md`](docs/HISTORICAL_DIAGNOSTICS.md) |

Current archived reports record a passing strict matrix for FAST-LIVO,
FAST-LIVO2, M2DGR, MCD, and R3LIVE coverage. The strongest closed-loop GL2
result is on FAST-LIVO2 `CBD_Building_01`: cm-class baseline tracking, full
frame live mapper feedback, and measured PSNR/ATE ablations documented in
`docs/GL2_RESULTS.md`.

## Repository Layout

```text
src/                         ROS2 packages for messages, the Coco-LIC front-end, mapping, and tools
src/gaussian_lic_bringup/config/
                             Mapper dataset and runtime configuration profiles
src/cocolic/config/          Coco-LIC continuous-time odometry configurations
run_lio/config/              Additional replay and validation configurations
scripts/                     Build, replay, conversion, validation, and audit utilities
docs/                        Public validation reports and engineering notes
docker/                      Jazzy container environment
external/                    Upstream source inventory
```

## Reproducibility Boundary

This repository intentionally does not commit:

- raw datasets or rosbag2/MCAP conversions
- bulk rendered images, point clouds, maps, or uncurated experiment bundles
- local TensorRT engines or GPU-specific binary artifacts
- large upstream baseline bundles

Small reference/baseline trajectories and selected validation outputs under
`baseline/`, `run_lio/data/`, and `results/` are versioned when they are needed
to substantiate a report. The public contract is therefore source plus this
curated validation evidence, not the raw datasets or complete generated output.
To reproduce a dataset-level report, obtain the relevant dataset/baseline
artifacts, configure the local paths, and run the scripts referenced in the
matching `docs/` report.

## Platform

Primary development target:

```text
Ubuntu 24.04
ROS2 Jazzy
CUDA 12.x
libtorch 2.x
```

Other ROS2 distributions are not part of the required CI or public support
surface for this port.

## Scope and Limitations

- This is an unofficial research/engineering port, not an APRIL-ZJU release.
- Dataset-level claims depend on valid references; broken zero-trajectory
  upstream references are treated as evidence defects, not silently accepted.
- Closed-loop render feedback (patch-photometric and the pose-level Camera
  Factor Option 2) measures as producing no repeatable accuracy benefit in
  any tested configuration; the unanchored degenerate testbed is chaotic
  across repeats, so single-run effects there must not be read as signal.
  The mechanism stays active and numerically healthy — the finding is a
  boundary statement about its utility in this pipeline, not a defect.

## Detailed Notes

The previous long-form README has been retained as
[`docs/DETAILED_STATUS_ARCHIVE.md`](docs/DETAILED_STATUS_ARCHIVE.md) for audit
history. New readers should start with this README and the validation reports
linked above.
