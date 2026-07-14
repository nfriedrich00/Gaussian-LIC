#include <odom/msg_manager.h>

#include <opencv2/imgcodecs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_field.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
sensor_msgs::msg::PointCloud2 ValidCloud()
{
  using PF = sensor_msgs::msg::PointField;
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = 10;
  cloud.width = 2;
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.point_step = 20;
  cloud.row_step = 40;
  cloud.data.resize(40);
  const auto add = [&](const char *name, uint32_t offset, uint8_t datatype) {
    PF field;
    field.name = name; field.offset = offset; field.datatype = datatype; field.count = 1;
    cloud.fields.push_back(field);
  };
  add("x", 0, PF::FLOAT32); add("y", 4, PF::FLOAT32);
  add("z", 8, PF::FLOAT32); add("intensity", 12, PF::UINT8);
  add("tag", 13, PF::UINT8); add("line", 14, PF::UINT8);
  add("offset_time", 16, PF::UINT32);
  for (size_t i = 0; i < 2; ++i)
  {
    const float xyz[3] = {float(i + 1), float(i + 2), float(i + 3)};
    const uint32_t offset = static_cast<uint32_t>(i * 50000000U);
    auto *point = cloud.data.data() + i * cloud.point_step;
    std::memcpy(point, xyz, sizeof(xyz));
    point[12] = static_cast<uint8_t>(20 + i);
    point[13] = 0x10; point[14] = static_cast<uint8_t>(i);
    std::memcpy(point + 16, &offset, sizeof(offset));
  }
  return cloud;
}
} // namespace

int main()
{
  try
  {
    const auto cloud = ValidCloud();
    const auto converted = cocolic::MsgManager::PointCloud2ToCustomMsg(cloud);
    if (converted->point_num != 2 || converted->points[1].offset_time != 50000000U ||
        std::abs(converted->points[1].x - 2.0F) > 1e-6F)
      throw std::runtime_error("valid PointCloud2 conversion changed values");

    auto missing_time = cloud;
    missing_time.fields.pop_back();
    bool rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(missing_time); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("missing offset_time was accepted");

    auto missing_tag = cloud;
    missing_tag.fields.erase(missing_tag.fields.begin() + 4);
    rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(missing_tag); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("missing Livox tag was accepted");

    auto missing_line = cloud;
    missing_line.fields.erase(missing_line.fields.begin() + 5);
    rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(missing_line); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("missing Livox line was accepted");

    auto short_data = cloud;
    short_data.data.resize(10);
    rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(short_data); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("short PointCloud2 data was accepted");

    auto bad_height = cloud;
    bad_height.height = 0;
    rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(bad_height); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("non-empty PointCloud2 with height=0 was accepted");

    auto nan_intensity = cloud;
    nan_intensity.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(nan_intensity.data.data() + 12, &nan_value, sizeof(nan_value));
    rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(nan_intensity); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("NaN PointCloud2 intensity was accepted");

    auto nan_xyz = cloud;
    std::memcpy(nan_xyz.data.data(), &nan_value, sizeof(nan_value));
    rejected = false;
    try { (void)cocolic::MsgManager::PointCloud2ToCustomMsg(nan_xyz); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("NaN PointCloud2 xyz was accepted");

#ifdef COCOLIC_HAS_LIVOX_ROS_DRIVER2
    livox_ros_driver2::msg::CustomMsg native_livox;
    native_livox.timebase = 1234567890ULL;
    native_livox.point_num = 2;
    native_livox.points.resize(2);
    native_livox.points[0].x = 1.0F;
    native_livox.points[0].y = 2.0F;
    native_livox.points[0].z = 3.0F;
    native_livox.points[0].reflectivity = 42;
    native_livox.points[0].tag = 0x10;
    native_livox.points[0].line = 3;
    native_livox.points[0].offset_time = 25000000U;
    native_livox.points[1] = native_livox.points[0];
    native_livox.points[1].x = 4.0F;
    native_livox.points[1].offset_time = 75000000U;
    const auto native_converted =
        cocolic::MsgManager::LivoxCustomMsgToLite(native_livox);
    if (native_converted->timebase != 1234567890LL ||
        native_converted->point_num != 2 ||
        native_converted->points[0].reflectivity != 42 ||
        native_converted->points[0].tag != 0x10 ||
        native_converted->points[0].line != 3 ||
        native_converted->points[1].offset_time != 75000000U ||
        std::abs(native_converted->points[1].x - 4.0F) > 1e-6F)
      throw std::runtime_error("valid native Livox conversion changed values");

    auto mismatched_native = native_livox;
    mismatched_native.point_num = 1;
    rejected = false;
    try { (void)cocolic::MsgManager::LivoxCustomMsgToLite(mismatched_native); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("native Livox point_num mismatch was accepted");

    native_livox.point_num = 2;
    native_livox.points[0].x = nan_value;
    rejected = false;
    try { (void)cocolic::MsgManager::LivoxCustomMsgToLite(native_livox); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("NaN native Livox xyz was accepted");
#endif

    cv::Mat image(3, 4, CV_8UC3, cv::Scalar(12, 34, 56));
    auto compressed = std::make_shared<sensor_msgs::msg::CompressedImage>();
    compressed->format = "jpeg";
    if (!cv::imencode(".jpg", image, compressed->data))
      throw std::runtime_error("test JPEG encoding failed");
    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serializer;
    rclcpp::SerializedMessage cdr;
    serializer.serialize_message(compressed.get(), &cdr);
    auto cdr_roundtrip = std::make_shared<sensor_msgs::msg::CompressedImage>();
    serializer.deserialize_message(&cdr, cdr_roundtrip.get());
    const cv::Mat decoded = cocolic::MsgManager::DecodeCompressedImage(cdr_roundtrip);
    if (decoded.rows != image.rows || decoded.cols != image.cols || decoded.type() != CV_8UC3)
      throw std::runtime_error("CompressedImage decode contract failed");

    if (cocolic::MsgManager::ResolveTopic("robot", "vio/test_img") !=
            "/robot/vio/test_img" ||
        cocolic::MsgManager::ResolveTopic("/robot/", "/fixed/image") !=
            "/fixed/image" ||
        cocolic::MsgManager::ResolveTopic("", "vio/test_img") !=
            "/vio/test_img")
      throw std::runtime_error("ROS topic prefix contract failed");

    auto imu = std::make_shared<sensor_msgs::msg::Imu>();
    imu->header.stamp.sec = 3;
    imu->header.stamp.nanosec = 5;
    imu->angular_velocity.x = 1.0;
    imu->linear_acceleration.z = 1.0;
    imu->orientation.w = 1.0;
    cocolic::IMUData imu_data;
    bool orientation_valid = false;
    std::string imu_error;
    if (!cocolic::MsgManager::IMUMsgToIMUData(
            imu, imu_data, true, &orientation_valid, &imu_error) ||
        !orientation_valid || imu_data.timestamp != 3000000005LL ||
        std::abs(imu_data.accel.z() - 9.81) > 1e-12)
      throw std::runtime_error("valid IMU conversion contract failed");

    imu->orientation.w = 0.0;
    if (!cocolic::MsgManager::IMUMsgToIMUData(
            imu, imu_data, false, &orientation_valid, &imu_error) ||
        orientation_valid ||
        std::abs(imu_data.orientation.unit_quaternion().w() - 1.0) > 1e-12)
      throw std::runtime_error("invalid IMU orientation did not fall back to identity");
    imu->angular_velocity.y = std::numeric_limits<double>::quiet_NaN();
    if (cocolic::MsgManager::IMUMsgToIMUData(
            imu, imu_data, false, &orientation_valid, &imu_error))
      throw std::runtime_error("non-finite IMU sample was accepted");
    imu->angular_velocity.y = 0.0;
    imu->header.stamp.nanosec = 1000000000U;
    if (cocolic::MsgManager::IMUMsgToIMUData(
            imu, imu_data, false, &orientation_valid, &imu_error))
      throw std::runtime_error("invalid IMU nanosecond field was accepted");

    const YAML::Node lidar_config = YAML::Load(R"(
VLP16:
  edge_threshold: 1.0
  surf_threshold: 0.1
  odometry_surface_leaf_size: 0.2
  min_distance: 0.1
  max_distance: 200.0
  N_SCAN: 16
  Horizon_SCAN: 1800
)");
    cocolic::VelodyneFeatureExtraction velodyne(lidar_config);
    RTPointCloudTmp velodyne_cloud;
    for (int i = 0; i < 2; ++i)
    {
      RTPointTmp p{}; p.x = 5.0F + i; p.ring = i; p.time = 0.01F * i;
      velodyne_cloud.push_back(p);
    }
    sensor_msgs::msg::PointCloud2 velodyne_msg;
    pcl::toROSMsg(velodyne_cloud, velodyne_msg);
    auto velodyne_ptr =
        std::make_shared<const sensor_msgs::msg::PointCloud2>(velodyne_msg);
    RTPointCloud::Ptr parsed(new RTPointCloud);
    if (!velodyne.ParsePointCloud(velodyne_ptr, parsed) || parsed->size() != 2)
      throw std::runtime_error("Velodyne time schema was rejected");

    OusterPointCloudTmp ouster_cloud;
    for (int i = 0; i < 2; ++i)
    {
      OusterPointTmp p{}; p.x = 5.0F + i; p.ring = i; p.t = 20000000U * i;
      ouster_cloud.push_back(p);
    }
    sensor_msgs::msg::PointCloud2 ouster_msg;
    pcl::toROSMsg(ouster_cloud, ouster_msg);
    parsed->clear();
    if (!velodyne.ParsePointCloud(
            std::make_shared<const sensor_msgs::msg::PointCloud2>(ouster_msg),
            parsed) ||
        parsed->size() != 2 || parsed->points[1].time != 20000000LL)
      throw std::runtime_error("per-message Ouster t schema switching failed");

    RTPointCloudTmpHesai hesai_cloud;
    for (int i = 0; i < 2; ++i)
    {
      RTPointTmpHesai p{}; p.x = 5.0F + i; p.ring = i;
      p.timestamp = 100.0 + 0.03 * i; hesai_cloud.push_back(p);
    }
    sensor_msgs::msg::PointCloud2 hesai_msg;
    pcl::toROSMsg(hesai_cloud, hesai_msg);
    parsed->clear();
    if (!velodyne.ParsePointCloud(
            std::make_shared<const sensor_msgs::msg::PointCloud2>(hesai_msg),
            parsed) ||
        parsed->size() != 2 ||
        std::llabs(parsed->points[1].time - 30000000LL) > 1000LL)
      throw std::runtime_error("per-message Hesai timestamp schema switching failed");

    auto missing_ring = velodyne_msg;
    missing_ring.fields.erase(
        std::remove_if(missing_ring.fields.begin(), missing_ring.fields.end(),
                       [](const auto &field) { return field.name == "ring"; }),
        missing_ring.fields.end());
    if (velodyne.ParsePointCloud(
            std::make_shared<const sensor_msgs::msg::PointCloud2>(missing_ring),
            parsed))
      throw std::runtime_error("rotating LiDAR cloud without ring was accepted");
    velodyne.LidarHandler(RTPointCloud::Ptr(new RTPointCloud));
    RTPointCloud::Ptr one_point(new RTPointCloud);
    RTPoint tiny{}; tiny.x = 5.0F; tiny.ring = 0; one_point->push_back(tiny);
    velodyne.LidarHandler(one_point);
  }
  catch (const std::exception &error)
  {
    std::cerr << "input_contract_test: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
