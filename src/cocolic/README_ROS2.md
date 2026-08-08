# Coco-LIC ROS 2 package notes

The workspace-level audit that distinguishes this embedded compatibility port
from Gaussian-LIC2's new adapter/tracking implementations is
[`docs/ROS1_TO_ROS2_CHANGES_CN.md`](../../docs/ROS1_TO_ROS2_CHANGES_CN.md).

Run the installed package with:

```bash
ros2 launch cocolic odometry.launch.py \
  config_path:=/path/to/ct_odometry_fastlivo2.yaml \
  bag_path:=/path/to/rosbag2
```

`config_path` is a ROS parameter and remains compatible with the legacy first
positional argument. Sensor YAML paths beginning with `/` are interpreted
relative to the directory containing the main profile, matching the upstream
profile layout without depending on the current working directory.

`output_directory` defaults to `$ROS_HOME/cocolic` (or `$HOME/.ros/cocolic`).
Frame IDs and topic names can be adapted with `world_frame`, `global_frame`,
`lidar_frame`, `camera_frame`, `image_frame`, and `topic_prefix`. The launch
file can start the installed RViz profile with `use_rviz:=true`.

Livox topics may use `sensor_msgs/msg/PointCloud2` with mandatory `x`, `y`,
`z` (`FLOAT32`), `offset_time` (`UINT32`, nanoseconds), `tag` (`UINT8`) and
`line` (`UINT8`) fields. Invalid
layouts fail before entering the estimator. If `livox_ros_driver2` is present
when this package is configured, native `livox_ros_driver2/msg/CustomMsg`
deserialization is enabled automatically. A bag containing that type produces
a clear rebuild instruction when the optional driver was unavailable.

The package installs FAST-LIVO, FAST-LIVO2, LVI-SAM, M2DGR, R3LIVE and VIRAL
profiles. Dataset paths are intentionally supplied through `bag_path` rather
than baked into launch files.

The native ROS 2 visualization surface covers every publisher reachable from
the current estimator: TF, lidar/camera odometry, spline paths, dense/debug
clouds and tracking images. Gaussian mapper-contract topics have one owner in
`OdometryManager`, including camera info and feedback lockstep; the viewer does
not advertise duplicate GS topics. Their names are configurable through
`gs_image_topic`, `gs_depth_topic`, `gs_camera_info_topic`, `gs_pose_topic`,
`gs_points_topic`, and `rendered_feedback_topic`.
