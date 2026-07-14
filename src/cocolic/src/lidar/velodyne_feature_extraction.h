/*
 * Coco-LIC: Continuous-Time Tightly-Coupled LiDAR-Inertial-Camera Odometry using Non-Uniform B-spline
 * Copyright (C) 2023 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

// ROS2 Jazzy port: dropped <ros/ros.h>; ROS1 sensor_msgs → ROS2 message.
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "lidar_feature.h"

#include <utils/cloud_tool.h>
#include <utils/mypcl_cloud_type.h>
#include <utils/parameter_struct.h>

// #include <livox_ros_driver/CustomMsg.h>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <cocolic/msg/feature_cloud.hpp>

namespace cocolic
{

  struct smoothness_t
  {
    float value;
    size_t ind;
  };

  struct by_value
  {
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
      return left.value < right.value;
    }
  };

  class VelodyneFeatureExtraction
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<VelodyneFeatureExtraction> Ptr;

    VelodyneFeatureExtraction(const YAML::Node &node,
                              const rclcpp::Node::SharedPtr &ros_node = nullptr);

    void LidarHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar_msg);

    void LidarHandler(const RTPointCloud::Ptr raw_cloud, int64_t stamp_ns = -1);

    //
    bool ParsePointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar_msg,
                         RTPointCloud::Ptr out_cloud) const;
    bool ParsePointCloudNoFeature(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar_msg,
        RTPointCloud::Ptr out_cloud);

    static bool CheckMsgFields(const sensor_msgs::msg::PointCloud2 &cloud_msg,
                               std::string fields_name = "time");

    inline RTPointCloud::Ptr GetCornerFeature() const { return p_corner_cloud; }

    inline RTPointCloud::Ptr GetSurfaceFeature() const { return p_surface_cloud; }

  private:
    void AllocateMemory();

    void ResetParameters();

    //
    void OrganizedCloudToRangeImage(const RTPointCloud::Ptr cur_cloud,
                                    cv::Mat &dist_image,
                                    RTPointCloud::Ptr &corresponding_cloud) const;

    //
    void RTCloudToRangeImage(const RTPointCloud::Ptr cur_cloud,
                             cv::Mat &dist_image,
                             RTPointCloud::Ptr &corresponding_cloud) const;

    void CloudExtraction();

    //
    void CaculateSmoothness();

    //
    void MarkOccludedPoints();

    //
    void ExtractFeatures();

    //
    void PublishCloud(int64_t stamp_ns);

  private:
    rclcpp::Node::SharedPtr ros_node_;
    std::string debug_frame_ = "lidar";
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_corner_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_surface_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_full_cloud_;
    rclcpp::Publisher<cocolic::msg::FeatureCloud>::SharedPtr pub_feature_cloud_;

    LiDARFeatureParam fea_param_;

    //
    int n_scan;
    int horizon_scan;
    cv::Mat range_mat;
    RTPointCloud::Ptr p_full_cloud;

    RTPointCloud::Ptr p_extracted_cloud;

    std::vector<float> point_range_list;
    std::vector<int> point_column_id;
    std::vector<int> start_ring_index;
    std::vector<int> end_ring_index;

    std::vector<smoothness_t> cloud_smoothness;
    std::vector<float> cloud_curvature;
    std::vector<int> cloud_neighbor_picked;
    std::vector<int> cloud_label;

    ///
    RTPointCloud::Ptr p_corner_cloud;
    RTPointCloud::Ptr p_surface_cloud;

    VoxelFilter<RTPoint> down_size_filter;

    MODE work_mode_;
  };

} // namespace cocolic
