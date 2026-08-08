# Gaussian Mapper ROS1/ROS2 Parity Contract

This document records the algorithm boundary implemented by
`gaussian_lic_mapping`. The reference is `external/Gaussian-LIC/src/gaussian.cpp`;
ROS2 middleware changes are allowed, while the standard dataset profiles on a
full Torch/CUDA build keep the CUDA mapping behavior equivalent to the ROS1
implementation. The naked node's conservative parameter defaults and the
CPU/debug renderer are diagnostic paths, not this parity claim.

## Upstream-equivalent standard-profile path

| Stage | ROS2 behavior |
|---|---|
| Frame selection | SPNet and Gaussian insertion run on keyframes only. |
| SPNet validation | Mean completed-minus-known sparse depth must have absolute value below 0.1 m. The bias is removed before selection. |
| SPNet sampling | Sobel edges are rejected; patches containing any source depth are skipped; the minimum positive completion depth seeds at most one point per empty patch. |
| Depth supervision | Completion adds colored 3D points but never replaces the original sparse training depth image. |
| `max_depth` | Filters only SPNet-synthesized points, never source LiDAR points. |
| Initialization/extension | Foreground log-scale is `log(scaling_scale * depth / focal)` on CPU and CUDA. Skybox initialization keeps the upstream CUDA KNN scale branch. |
| Visibility extension | Alpha-hole testing renders the complete foreground plus skybox map before appending pending points. |
| Optimization | CUDA rasterization, L1 plus fused SSIM, sparse depth loss, visibility-masked sparse Adam, skybox, and upstream random camera sampling are active in the full build. |
| Iteration decay | Motion over 120 m replaces the ordinary list with a halved iteration budget, sampled equally from the first two-thirds and last third. With upstream budget 100, both 80- and 150-camera sets select 25+25 cameras. |
| Save/evaluate | Gaussian PLY is binary little-endian with upstream property order. Final evaluation writes train/test renders, depth, GT, PSNR, SSIM, and optional AlexNet LPIPS. `lpips_model_path` accepts either the model file or its directory. |

PointCloud2 conversion honors organized-cloud `row_step`, per-field bounds and
datatype/count, source endianness, packed RGB endianness, and scalar colors in
either 0..1 or integer ranges. Non-finite XYZ is rejected, non-finite scalar
color/intensity is replaced with zero, and invalid pose/extrinsic quaternions
are rejected before normalization. Each aligned frame stores the single
CameraInfo/parameter intrinsics snapshot used for point projection; SPNet,
Torch camera construction, sampled-frame optimization, preview rendering, and
final evaluation reuse that frame's snapshot even if a newer CameraInfo arrives.

## Explicit non-parity options

- The upstream renderer accepts `apply_exposure` but does not use its boolean
  argument in the public implementation. This port makes that dormant feature
  functional with an optimized 3x4 affine exposure transform. It is therefore
  an intentional opt-in behavior fix, not bit-exact execution. All standard
  profiles keep `apply_exposure: false`.
- Pruning, densification, count capping, and opacity reset added by the earlier
  ROS2 work are not part of the released upstream mapping loop. They run only
  when `enable_non_upstream_density_control: true` is set together with their
  individual options. Standard profiles leave this gate false; only the
  experimental densify profile opts in.
- CPU Torch mode is a diagnostic tensor/backend path. Strict mapper parity
  requires the Torch/CUDA rasterizer build; middleware-only builds fail clearly
  when a real rasterizer is requested.

## ROS2 execution and completion

Sensor subscriptions remain in the default mutually-exclusive group, GPU
mapping plus SaveMap use a separate mutually-exclusive group, and status
snapshots use a third group. The standalone node and composed launch use a
multi-threaded executor, so status or map saving waiting on GPU state cannot
prevent DDS sensor input from entering bounded queues.

`auto_finalize_on_inactivity` is off by default for safe live operation. When
enabled, the mapper waits for both input and converted-frame inactivity, drains
and accounts for every orphaned point/pose/image/depth and feedback queue, saves
the map, writes evaluation/finalization manifests, and optionally shuts down.

`MappingStatus.msg` gained additive SPNet completion counters. Because ROS2
message type hashes include the full schema, all publishers/subscribers must be
rebuilt together; this is source-level additive compatibility, not wire
compatibility with a binary built from the previous message definition.

The deterministic probes are registered with CTest as
`frame_data_parity`, `optimization_sampling_parity`, and
`mapping_concurrency_contract`.
