# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure helpers used by the launch file to preserve selected YAML profiles."""

import json
import os
from pathlib import Path

import yaml


PROFILE_INHERIT_SENTINEL = "__profile__"

# These are mapper defaults used only when an older profile omits a parameter.
# A launch argument whose value is PROFILE_INHERIT_SENTINEL first takes the
# corresponding mapping_node.ros__parameters value from the selected YAML.
MAPPING_OVERRIDE_FALLBACKS = {
    "depth_completion": False,
    "depth_completion_engine_path": "",
    "enable_torch_camera_conversion": True,
    "enable_torch_gaussian_init": True,
    "enable_torch_gaussian_extend": True,
    "enable_torch_gaussian_optimization": True,
    "torch_gaussian_optimization_steps": 100,
    "torch_gaussian_optimization_max_samples": 4096,
    "torch_gaussian_optimization_sampling": "upstream_random",
    "torch_gaussian_optimization_seed": 20260505,
    "enable_torch_gaussian_pruning": True,
    "enable_torch_gaussian_densification": False,
    "enable_non_upstream_density_control": False,
    "torch_gaussian_prune_min_opacity": 0.005,
    "torch_gaussian_max_foreground": 1500000,
    "torch_gaussian_prune_count_policy": "uniform",
    "torch_gaussian_prune_max_world_scale": 0.0,
    "enable_torch_gaussian_extend_visibility_filter": True,
    "torch_gaussian_extend_alpha_threshold": 0.99,
    "torch_gaussian_opacity_reset_interval": 0,
    "torch_gaussian_device": "cuda",
    "publish_gaussian_map": True,
    "gaussian_map_publish_min_interval_sec": 0.0,
    "gaussian_map_publish_on_empty_extend": True,
    "save_map_render_evaluation": False,
    "lpips_model_path": "",
    "test_frame_stride": 1,
    # EOF from the rosbag process is authoritative.  Keeping inactivity off by
    # default avoids finalizing in the middle of a bag with a legitimate gap.
    "auto_finalize_on_inactivity": False,
    "auto_finalize_inactivity_sec": 5.0,
    "auto_finalize_output_path": "gaussian_lic_result",
    "auto_finalize_include_skybox": False,
    # A successful one-shot EOF should make the standalone mapper exit after
    # saving; live and looping launches retain the node.
    "auto_finalize_exit": None,
    "sensor_qos_reliability": "best_effort",
    "sensor_qos_history": "keep_last",
    "sensor_qos_depth": 5,
    "require_depth_topic": True,
    "sync_anchor_stream": "pointcloud",
    "render_mode": "rasterizer",
    "rendered_feedback_topic": "/gaussian_lic/rendered_feedback",
    "rendered_feedback_qos_reliability": "reliable",
    "rendered_feedback_qos_durability": "volatile",
    "rendered_feedback_qos_depth": 128,
    "rendered_feedback_source_stream": "aligned_frame",
    "rendered_feedback_image_topic": "__inherit__",
    "rendered_feedback_pose_topic": "__inherit__",
    "rendered_feedback_image_qos_reliability": "best_effort",
    "rendered_feedback_image_qos_history": "keep_last",
    "rendered_feedback_image_qos_depth": 64,
    "rendered_feedback_pose_qos_reliability": "best_effort",
    "rendered_feedback_pose_qos_history": "keep_last",
    "rendered_feedback_pose_qos_depth": 64,
    "publish_rendered_feedback_before_update": False,
    "rendered_image_mode": "",
    "publish_tf": False,
}

SENSOR_QOS_STREAMS = (
    "pointcloud",
    "pose",
    "image",
    "camera_info",
    "depth",
    "imu",
    "raw_image",
    "raw_camera_info",
    "raw_depth",
    "raw_pointcloud",
    "raw_imu",
    "pose_stamped",
    "raw_odometry",
    "frontend_odometry",
)
SENSOR_QOS_SUFFIXES = ("reliability", "history", "depth")

for _stream in SENSOR_QOS_STREAMS:
    MAPPING_OVERRIDE_FALLBACKS[f"{_stream}_qos_reliability"] = "best_effort"
    MAPPING_OVERRIDE_FALLBACKS[f"{_stream}_qos_history"] = "keep_last"
    MAPPING_OVERRIDE_FALLBACKS[f"{_stream}_qos_depth"] = 5

STREAM_QOS_GLOBAL_OVERRIDES = {
    f"{stream}_qos_{suffix}": f"sensor_qos_{suffix}"
    for stream in SENSOR_QOS_STREAMS
    for suffix in SENSOR_QOS_SUFFIXES
}


def load_mapping_profile(config_path):
    """Return merged wildcard and mapping_node ROS parameters from a ROS2 YAML."""
    path = Path(config_path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot read mapping profile {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeError(f"Mapping profile {path} must contain a YAML mapping")

    merged = {}
    wildcard = document.get("/**", {})
    if isinstance(wildcard, dict):
        wildcard_parameters = wildcard.get("ros__parameters", {})
        if isinstance(wildcard_parameters, dict):
            merged.update(wildcard_parameters)

    node_sections = [
        section
        for name, section in document.items()
        if str(name).rstrip("/").split("/")[-1] == "mapping_node"
        and isinstance(section, dict)
    ]
    if len(node_sections) > 1:
        raise RuntimeError(f"Mapping profile {path} defines mapping_node more than once")
    if node_sections:
        node_parameters = node_sections[0].get("ros__parameters", {})
        if not isinstance(node_parameters, dict):
            raise RuntimeError(
                f"mapping_node.ros__parameters in {path} must contain a YAML mapping"
            )
        merged.update(node_parameters)
    if not merged:
        raise RuntimeError(f"Mapping profile {path} has no mapping_node ROS parameters")
    return merged


def launch_text(value):
    """Serialize a YAML scalar/container for launch substitution type inference."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def parse_launch_bool(value, name):
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false, got {value!r}")


def resolve_depth_completion_engine_path(
    configured, width, height, *, environ=None, home=None
):
    """Resolve an external SPNet engine without baking machine paths into YAML."""
    configured_path = str(configured).strip()
    if configured_path:
        return str(Path(configured_path).expanduser())

    environment = os.environ if environ is None else environ
    environment_path = str(environment.get("GAUSSIAN_LIC_SPNET_ENGINE", "")).strip()
    if environment_path:
        return str(Path(environment_path).expanduser())

    try:
        width_value = int(width)
        height_value = int(height)
    except (TypeError, ValueError):
        return ""
    if width_value <= 0 or height_value <= 0:
        return ""

    home_path = Path.home() if home is None else Path(home)
    engine_dir = home_path / "Software" / "TensorRT-engines"
    candidates = (
        engine_dir / f"spnet_{height_value}_{width_value}_fp16.engine",
        engine_dir / f"spnet_{height_value}_{width_value}.engine",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def resolve_lpips_model_path(configured, *, environ=None, candidates=()):
    """Resolve the LPIPS TorchScript asset required by upstream-style evaluation."""
    configured_path = str(configured).strip()
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.is_dir():
            path /= "lpips_alex.pt"
        return str(path)

    environment = os.environ if environ is None else environ
    environment_path = str(environment.get("GAUSSIAN_LIC_LPIPS_MODEL", "")).strip()
    if environment_path:
        path = Path(environment_path).expanduser()
        if path.is_dir():
            path /= "lpips_alex.pt"
        return str(path)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_dir():
            path /= "lpips_alex.pt"
        if path.is_file():
            return str(path)
    return ""


def resolve_mapping_overrides(
    current_values, profile_parameters, play_bag, loop_bag="false"
):
    """Resolve profile sentinels while retaining explicit CLI values.

    Stream QoS follows the ROS2 launch precedence expected by callers:
    explicit stream CLI, explicit global CLI, stream profile, global profile,
    then the built-in fallback.
    """
    resolved = {}
    for name, fallback in MAPPING_OVERRIDE_FALLBACKS.items():
        current = current_values[name]
        if current != PROFILE_INHERIT_SENTINEL:
            resolved[name] = current
            continue

        global_qos_name = STREAM_QOS_GLOBAL_OVERRIDES.get(name)
        if (
            global_qos_name is not None
            and current_values[global_qos_name] != PROFILE_INHERIT_SENTINEL
        ):
            value = current_values[global_qos_name]
        elif name in profile_parameters:
            value = profile_parameters[name]
        elif global_qos_name is not None and global_qos_name in profile_parameters:
            value = profile_parameters[global_qos_name]
        elif name == "auto_finalize_exit" and fallback is None:
            value = parse_launch_bool(play_bag, "play_bag") and not parse_launch_bool(
                loop_bag, "loop_bag"
            )
        else:
            value = fallback
        resolved[name] = launch_text(value)
    return resolved


def resolve_pointcloud_coordinates(
    requested, frontend_adapter, imu_pose_fallback, rotate_with_imu_pose
):
    """Resolve auto coordinate semantics at launch time."""
    token = str(requested).strip().lower()
    if token not in {"auto", "world", "sensor"}:
        raise RuntimeError(
            "pointcloud_coordinates must be one of auto, world, or sensor, "
            f"got {requested!r}"
        )
    if token != "auto":
        return token
    adapter_enabled = parse_launch_bool(frontend_adapter, "frontend_adapter")
    imu_fallback_enabled = parse_launch_bool(imu_pose_fallback, "adapter_imu_pose_fallback")
    rotates_to_world = parse_launch_bool(
        rotate_with_imu_pose, "adapter_rotate_pointcloud_with_imu_pose"
    )
    if not adapter_enabled or (imu_fallback_enabled and rotates_to_world):
        return "world"
    return "sensor"


def bag_play_command(bag, loop=False):
    command = ["ros2", "bag", "play", bag, "--clock", "--read-ahead-queue-size", "100"]
    if loop:
        command.append("--loop")
    return command
