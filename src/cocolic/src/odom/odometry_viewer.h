/*
 * Coco-LIC ROS 2 visualization and TF bridge.
 *
 * This keeps the publisher surface used by the estimator while using native
 * ROS 2 publishers, messages and tf2.  Visualization remains observational:
 * it never feeds data back into estimation.
 */
#pragma once

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <climits>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <lidar/lidar_feature.h>
#include <spline/trajectory.h>
#include <utils/mypcl_cloud_type.h>

namespace cocolic
{

enum SplineViewerType
{
  Init = 0,
  Loop = 2,
};

template <typename MessageT>
class PublisherHandle
{
public:
  using Publisher = rclcpp::Publisher<MessageT>;

  void reset(typename Publisher::SharedPtr publisher)
  {
    publisher_ = std::move(publisher);
  }

  size_t getNumSubscribers() const
  {
    return publisher_ ? publisher_->get_subscription_count() : 0U;
  }

  void publish(const MessageT &message) const
  {
    if (publisher_) publisher_->publish(message);
  }

  void publish(const std::shared_ptr<MessageT> &message) const
  {
    if (publisher_ && message) publisher_->publish(*message);
  }

private:
  typename Publisher::SharedPtr publisher_;
};

class OdometryViewer
{
public:
  PublisherHandle<sensor_msgs::msg::Image> pub_track_img_;
  PublisherHandle<sensor_msgs::msg::Image> pub_undistort_scan_in_cur_img_;
  PublisherHandle<sensor_msgs::msg::Image> pub_old_and_new_added_points_in_cur_img_;
  PublisherHandle<sensor_msgs::msg::PointCloud2> pub_sub_visual_map_;

  void SetPublisher(const rclcpp::Node::SharedPtr &node)
  {
    if (!node) throw std::invalid_argument("OdometryViewer requires a ROS 2 node");
    node_ = node;
    topic_prefix_ = DeclareOrGet("topic_prefix", std::string());
    world_frame_ = DeclareOrGet("world_frame", std::string("map"));
    global_frame_ = DeclareOrGet("global_frame", std::string("global"));
    lidar_frame_ = DeclareOrGet("lidar_frame", std::string("lidar"));
    camera_frame_ = DeclareOrGet("camera_frame", std::string("camera"));
    image_frame_ = DeclareOrGet("image_frame", std::string("image_frame"));
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*node_);
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    pub_track_img_.reset(node_->create_publisher<sensor_msgs::msg::Image>(Topic("track_img"), qos));
    pub_undistort_scan_in_cur_img_.reset(
        node_->create_publisher<sensor_msgs::msg::Image>(Topic("undistort_scan_in_cur_img"), qos));
    pub_old_and_new_added_points_in_cur_img_.reset(
        node_->create_publisher<sensor_msgs::msg::Image>(Topic("old_and_new_added_points_in_cur_img"), qos));
    pub_sub_visual_map_.reset(
        node_->create_publisher<sensor_msgs::msg::PointCloud2>(Topic("sub_visual_map"), qos));

    pub_spline_trajectory_ = node_->create_publisher<nav_msgs::msg::Path>(
        Topic("spline/trajectory"), qos);
    pub_spline_ctrl_ = node_->create_publisher<nav_msgs::msg::Path>(
        Topic("spline/ctrl_path"), qos);
    pub_spline_ctrl_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        Topic("spline/ctrl_cloud"), qos);
    pub_target_dense_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        Topic("lio/target_dense_cloud"), qos);
    pub_source_dense_cloud_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        Topic("lio/source_dense_cloud"), qos);
    pub_lidar_odom_ = node_->create_publisher<nav_msgs::msg::Odometry>(
        Topic("odometry/lidar"), qos);
    pub_camera_odom_ = node_->create_publisher<nav_msgs::msg::Odometry>(
        Topic("odometry/camera"), qos);

  }

  void PublishTrackImg(const sensor_msgs::msg::Image::SharedPtr &image)
  {
    if (!image) return;
    image->header.frame_id = image_frame_;
    pub_track_img_.publish(image);
  }

  void PublishUndistortScanInCurImg(
      const sensor_msgs::msg::Image::SharedPtr &image)
  {
    if (!image) return;
    image->header.frame_id = image_frame_;
    pub_undistort_scan_in_cur_img_.publish(image);
  }

  void PublishOldAndNewAddedPointsInCurImg(
      const sensor_msgs::msg::Image::SharedPtr &image)
  {
    if (!image) return;
    image->header.frame_id = image_frame_;
    pub_old_and_new_added_points_in_cur_img_.publish(image);
  }

  void PublishSubVisualMap(const VPointCloud &cloud)
  {
    if (pub_sub_visual_map_.getNumSubscribers() == 0 || cloud.empty()) return;
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(cloud, message);
    message.header.stamp = node_->now();
    message.header.frame_id = world_frame_;
    pub_sub_visual_map_.publish(message);
  }

  void PublishTF(const Eigen::Quaterniond &quaternion,
                 const Eigen::Vector3d &position,
                 const std::string &child_frame,
                 const std::string &parent_frame)
  {
    if (!node_ || !tf_broadcaster_) return;
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = node_->now();
    transform.header.frame_id = ResolveFrame(parent_frame);
    transform.child_frame_id = ResolveFrame(child_frame);
    transform.transform.translation.x = position.x();
    transform.transform.translation.y = position.y();
    transform.transform.translation.z = position.z();
    transform.transform.rotation.x = quaternion.x();
    transform.transform.rotation.y = quaternion.y();
    transform.transform.rotation.z = quaternion.z();
    transform.transform.rotation.w = quaternion.w();
    tf_broadcaster_->sendTransform(transform);

    if (child_frame == "lidar" || child_frame == "camera")
    {
      nav_msgs::msg::Odometry odometry;
      odometry.header = transform.header;
      odometry.child_frame_id = transform.child_frame_id;
      odometry.pose.pose.position.x = position.x();
      odometry.pose.pose.position.y = position.y();
      odometry.pose.pose.position.z = position.z();
      odometry.pose.pose.orientation = transform.transform.rotation;
      if (child_frame == "lidar") pub_lidar_odom_->publish(odometry);
      else pub_camera_odom_->publish(odometry);
    }
  }

  void PublishSplineTrajectory(const Trajectory::Ptr &trajectory,
                               double min_time, double max_time, double dt)
  {
    if (!node_ || !trajectory || dt <= 0.0) return;
    if (pub_spline_trajectory_->get_subscription_count() != 0)
    {
      nav_msgs::msg::Path path;
      path.header.stamp = node_->now();
      path.header.frame_id = world_frame_;
      for (double time = min_time; time < max_time - 0.08; time += dt)
      {
        const SE3d pose = trajectory->GetIMUPoseNURBS(time);
        geometry_msgs::msg::PoseStamped item;
        item.header.frame_id = world_frame_;
        item.header.stamp = rclcpp::Time(
            static_cast<int64_t>(std::max(0.0, time) * 1e9));
        item.pose.position.x = pose.translation().x();
        item.pose.position.y = pose.translation().y();
        item.pose.position.z = pose.translation().z();
        item.pose.orientation.x = pose.unit_quaternion().x();
        item.pose.orientation.y = pose.unit_quaternion().y();
        item.pose.orientation.z = pose.unit_quaternion().z();
        item.pose.orientation.w = pose.unit_quaternion().w();
        path.poses.push_back(item);
      }
      pub_spline_trajectory_->publish(path);
    }

    if (pub_spline_ctrl_->get_subscription_count() != 0)
    {
      nav_msgs::msg::Path controls;
      controls.header.stamp = node_->now();
      controls.header.frame_id = world_frame_;
      for (size_t index = 0; index < trajectory->numKnots(); ++index)
      {
        geometry_msgs::msg::PoseStamped item;
        item.header = controls.header;
        const auto &position = trajectory->getKnotPos(index);
        const auto quaternion = trajectory->getKnotSO3(index).unit_quaternion();
        item.pose.position.x = position.x();
        item.pose.position.y = position.y();
        item.pose.position.z = position.z();
        item.pose.orientation.x = quaternion.x();
        item.pose.orientation.y = quaternion.y();
        item.pose.orientation.z = quaternion.z();
        item.pose.orientation.w = quaternion.w();
        controls.poses.push_back(item);
      }
      pub_spline_ctrl_->publish(controls);
    }

    // Preserve the upstream visualization contract used by coco.rviz: each
    // spline knot is published in the world frame and its optimization state
    // is carried in the PointXYZI intensity channel.
    if (pub_spline_ctrl_cloud_->get_subscription_count() != 0)
    {
      VPointCloud control_cloud;
      control_cloud.reserve(trajectory->numKnots());
      for (size_t index = 0; index < trajectory->numKnots(); ++index)
      {
        const Eigen::Vector3d &position = trajectory->getKnotPos(index);
        VPoint control_point;
        control_point.x = position.x();
        control_point.y = position.y();
        control_point.z = position.z();
        control_point.intensity = trajectory->intensity_map[index];
        control_cloud.push_back(control_point);
      }

      sensor_msgs::msg::PointCloud2 message;
      pcl::toROSMsg(control_cloud, message);
      message.header.stamp = node_->now();
      message.header.frame_id = world_frame_;
      pub_spline_ctrl_cloud_->publish(message);
    }
  }

  void PublishDenseCloud(const Trajectory::Ptr &trajectory,
                         const LiDARFeature &target_feature,
                         const LiDARFeature &source_feature)
  {
    if (!node_ || !trajectory) return;
    if (pub_target_dense_cloud_->get_subscription_count() != 0 &&
        (++target_decimation_ % 20U == 0U))
    {
      sensor_msgs::msg::PointCloud2 message;
      pcl::toROSMsg(*target_feature.surface_features, message);
      message.header.stamp = node_->now();
      message.header.frame_id = world_frame_;
      pub_target_dense_cloud_->publish(message);
    }

    if (pub_source_dense_cloud_->get_subscription_count() == 0) return;
    VPointCloud transformed;
    transformed.reserve(source_feature.surface_features->size());
    for (const auto &point : source_feature.surface_features->points)
    {
      if (point.timestamp >= trajectory->maxTimeNsNURBS()) continue;
      const SE3d pose = trajectory->GetLidarPoseNURBS(point.timestamp);
      const Eigen::Vector3d local(point.x, point.y, point.z);
      const Eigen::Vector3d world = pose * local;
      VPoint output;
      output.x = world.x(); output.y = world.y(); output.z = world.z();
      output.intensity = point.intensity;
      transformed.push_back(output);
    }
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(transformed, message);
    message.header.stamp = node_->now();
    message.header.frame_id = world_frame_;
    pub_source_dense_cloud_->publish(message);
  }

private:
  template <typename T>
  T DeclareOrGet(const std::string &name, const T &default_value)
  {
    if (!node_->has_parameter(name)) return node_->declare_parameter<T>(name, default_value);
    return node_->get_parameter(name).get_value<T>();
  }

  std::string Topic(const std::string &suffix) const
  {
    if (topic_prefix_.empty()) return "/" + suffix;
    std::string prefix = topic_prefix_;
    if (prefix.front() != '/') prefix.insert(prefix.begin(), '/');
    if (prefix.back() != '/') prefix.push_back('/');
    return prefix + suffix;
  }

  std::string ResolveFrame(const std::string &frame) const
  {
    if (frame == "map") return world_frame_;
    if (frame == "global") return global_frame_;
    if (frame == "lidar") return lidar_frame_;
    if (frame == "camera") return camera_frame_;
    if (frame == "image_frame") return image_frame_;
    return frame;
  }

  rclcpp::Node::SharedPtr node_;
  std::string topic_prefix_;
  std::string world_frame_ = "map";
  std::string global_frame_ = "global";
  std::string lidar_frame_ = "lidar";
  std::string camera_frame_ = "camera";
  std::string image_frame_ = "image_frame";
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  size_t target_decimation_ = 0;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_spline_trajectory_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_spline_ctrl_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_spline_ctrl_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_target_dense_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_source_dense_cloud_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_lidar_odom_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_odom_;
};

} // namespace cocolic
