from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("cocolic"), "config", "ct_odometry_fastlivo2.yaml"]
    )
    config_path = LaunchConfiguration("config_path")
    bag_path = LaunchConfiguration("bag_path")
    output_directory = LaunchConfiguration("output_directory")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    world_frame = LaunchConfiguration("world_frame")
    global_frame = LaunchConfiguration("global_frame")
    lidar_frame = LaunchConfiguration("lidar_frame")
    camera_frame = LaunchConfiguration("camera_frame")
    image_frame = LaunchConfiguration("image_frame")
    topic_prefix = LaunchConfiguration("topic_prefix")
    debug_input_image_topic = LaunchConfiguration("debug_input_image_topic")
    gs_image_topic = LaunchConfiguration("gs_image_topic")
    gs_depth_topic = LaunchConfiguration("gs_depth_topic")
    gs_camera_info_topic = LaunchConfiguration("gs_camera_info_topic")
    gs_pose_topic = LaunchConfiguration("gs_pose_topic")
    gs_points_topic = LaunchConfiguration("gs_points_topic")
    rendered_feedback_topic = LaunchConfiguration("rendered_feedback_topic")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument(
                "bag_path",
                default_value="",
                description="Optional rosbag2 path overriding the selected profile",
            ),
            DeclareLaunchArgument(
                "output_directory",
                default_value="",
                description="Writable result root; empty uses $ROS_HOME/cocolic",
            ),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("world_frame", default_value="map"),
            DeclareLaunchArgument("global_frame", default_value="global"),
            DeclareLaunchArgument("lidar_frame", default_value="lidar"),
            DeclareLaunchArgument("camera_frame", default_value="camera"),
            DeclareLaunchArgument("image_frame", default_value="image_frame"),
            DeclareLaunchArgument("topic_prefix", default_value=""),
            DeclareLaunchArgument(
                "debug_input_image_topic",
                default_value="vio/test_img",
                description="Decoded input-image debug topic (relative topics honor topic_prefix)",
            ),
            DeclareLaunchArgument("gs_image_topic", default_value="/image_for_gs"),
            DeclareLaunchArgument("gs_depth_topic", default_value="/depth_for_gs"),
            DeclareLaunchArgument("gs_camera_info_topic", default_value="/camera_info_for_gs"),
            DeclareLaunchArgument("gs_pose_topic", default_value="/pose_for_gs"),
            DeclareLaunchArgument("gs_points_topic", default_value="/points_for_gs"),
            DeclareLaunchArgument(
                "rendered_feedback_topic",
                default_value="/gaussian_lic/rendered_feedback",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("cocolic"), "config", "coco.rviz"]
                ),
            ),
            Node(
                package="cocolic",
                executable="odometry_node",
                name="cocolic_odometry",
                output="screen",
                parameters=[{
                    "config_path": config_path,
                    "bag_path": bag_path,
                    "output_directory": output_directory,
                    "world_frame": world_frame,
                    "global_frame": global_frame,
                    "lidar_frame": lidar_frame,
                    "camera_frame": camera_frame,
                    "image_frame": image_frame,
                    "topic_prefix": topic_prefix,
                    "debug_input_image_topic": debug_input_image_topic,
                    "gs_image_topic": gs_image_topic,
                    "gs_depth_topic": gs_depth_topic,
                    "gs_camera_info_topic": gs_camera_info_topic,
                    "gs_pose_topic": gs_pose_topic,
                    "gs_points_topic": gs_points_topic,
                    "rendered_feedback_topic": rendered_feedback_topic,
                }],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="cocolic_rviz",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
