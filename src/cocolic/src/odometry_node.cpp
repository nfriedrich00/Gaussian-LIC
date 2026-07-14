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

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <glog/logging.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include <odom/odometry_manager.h>

using namespace cocolic;

namespace
{
class ExecutorGuard
{
public:
  explicit ExecutorGuard(const rclcpp::Node::SharedPtr &node)
      : executor_(std::make_shared<rclcpp::executors::SingleThreadedExecutor>())
  {
    executor_->add_node(node);
    thread_ = std::thread([executor = executor_]() { executor->spin(); });
  }

  ~ExecutorGuard()
  {
    executor_->cancel();
    if (thread_.joinable()) thread_.join();
  }

  ExecutorGuard(const ExecutorGuard &) = delete;
  ExecutorGuard &operator=(const ExecutorGuard &) = delete;

private:
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread thread_;
};
} // namespace

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  rclcpp::init(argc, argv);
  try
  {
    const auto non_ros_args = rclcpp::remove_ros_arguments(argc, argv);
    std::string legacy_path;
    if (non_ros_args.size() > 1) legacy_path = non_ros_args[1];

    auto ros_node = rclcpp::Node::make_shared("cocolic_odometry");
    std::string config_path =
        ros_node->declare_parameter<std::string>("config_path", legacy_path);
    if (config_path.empty())
    {
      config_path = ament_index_cpp::get_package_share_directory("cocolic") +
                    "/config/ct_odometry_fastlivo2.yaml";
    }
    config_path = std::filesystem::absolute(config_path).lexically_normal().string();
    std::cout << "Odometry load " << config_path << ".\n";

    YAML::Node config_node = YAML::LoadFile(config_path);
    // Optional ROS parameter override is useful for launch and CI without
    // mutating the installed dataset profile.
    const std::string bag_override =
        ros_node->declare_parameter<std::string>("bag_path", "");
    if (!bag_override.empty()) config_node["bag_path"] = bag_override;
    const char *ros_home = std::getenv("ROS_HOME");
    const char *home = std::getenv("HOME");
    std::string default_output = ros_home ? ros_home :
        (home ? std::string(home) + "/.ros" : std::string(".ros"));
    default_output += "/cocolic";
    std::string output_directory =
        ros_node->declare_parameter<std::string>("output_directory", default_output);
    if (output_directory.empty()) output_directory = default_output;

    std::cout << "\n🥥 Start Coco-LIC Odometry (ROS 2) 🥥\n";
    {
      OdometryManager odom_manager(
          config_node, std::filesystem::path(config_path).parent_path().string(),
          ros_node, output_directory);
      ExecutorGuard executor_guard(ros_node);
      odom_manager.RunBag();
      if (!odom_manager.SaveOdometry())
        throw std::runtime_error(
            "bag replay completed without an exportable trajectory");
    } // all worker threads are joined before rclcpp shutdown
    std::cout << "\n✨ All Done.\n\n";
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "Coco-LIC failed: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
}
