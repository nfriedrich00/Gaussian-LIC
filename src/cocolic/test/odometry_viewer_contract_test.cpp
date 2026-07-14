#include <odom/odometry_viewer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/executors/single_threaded_executor.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = rclcpp::Node::make_shared("odometry_viewer_contract_test");
    node->declare_parameter<std::string>("world_frame", "viewer_world");

    sensor_msgs::msg::PointCloud2::SharedPtr received;
    const auto subscription = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/spline/ctrl_cloud", rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
        [&received](sensor_msgs::msg::PointCloud2::SharedPtr message)
        {
          received = std::move(message);
        });
    (void)subscription;

    cocolic::OdometryViewer viewer;
    viewer.SetPublisher(node);

    auto trajectory = std::make_shared<cocolic::Trajectory>(0.1);
    for (size_t index = 0; index < trajectory->numKnots(); ++index)
    {
      trajectory->setKnotPos(
          Eigen::Vector3d(static_cast<double>(index) + 0.25,
                          -static_cast<double>(index) - 0.5,
                          2.0 * static_cast<double>(index) + 0.75),
          static_cast<int>(index));
      trajectory->intensity_map[index] = 100 + static_cast<int>(index);
    }

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    for (int attempt = 0; attempt < 200 && !received; ++attempt)
    {
      viewer.PublishSplineTrajectory(trajectory, 0.0, 0.0, 0.01);
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!received)
      throw std::runtime_error("/spline/ctrl_cloud was not published");
    if (received->header.frame_id != "viewer_world")
      throw std::runtime_error("control cloud did not use the configured world frame");
    if (received->header.stamp.sec == 0 && received->header.stamp.nanosec == 0)
      throw std::runtime_error("control cloud has a zero publication timestamp");

    VPointCloud cloud;
    pcl::fromROSMsg(*received, cloud);
    if (cloud.size() != trajectory->numKnots())
      throw std::runtime_error("control cloud knot count differs from the trajectory");
    for (size_t index = 0; index < cloud.size(); ++index)
    {
      const auto &point = cloud[index];
      const auto &expected = trajectory->getKnotPos(index);
      if (std::abs(point.x - expected.x()) > 1.0e-6 ||
          std::abs(point.y - expected.y()) > 1.0e-6 ||
          std::abs(point.z - expected.z()) > 1.0e-6 ||
          std::abs(point.intensity - trajectory->intensity_map[index]) > 1.0e-6)
        throw std::runtime_error("control cloud lost knot coordinates or intensity");
    }

    executor.remove_node(node);
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "odometry_viewer_contract_test: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
}
