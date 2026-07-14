// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_mapping/frame_data.hpp>
#include <gaussian_lic_mapping/mapper_dataset.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace
{

void require(const bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<typename T>
void write_big_endian(std::vector<uint8_t> & data, const size_t offset, const T & value)
{
  uint8_t bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  for (size_t i = 0; i < sizeof(T); ++i) {
    data.at(offset + i) = bytes[sizeof(T) - 1U - i];
  }
}

sensor_msgs::msg::PointField field(
  const std::string & name, const uint32_t offset, const uint8_t datatype)
{
  sensor_msgs::msg::PointField result;
  result.name = name;
  result.offset = offset;
  result.datatype = datatype;
  result.count = 1;
  return result;
}

sensor_msgs::msg::Image::SharedPtr make_rgb_image(const uint32_t width, const uint32_t height)
{
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->width = width;
  image->height = height;
  image->encoding = sensor_msgs::image_encodings::RGB8;
  image->step = width * 3U;
  image->data.resize(static_cast<size_t>(image->step) * height);
  for (uint32_t v = 0; v < height; ++v) {
    for (uint32_t u = 0; u < width; ++u) {
      const size_t offset = static_cast<size_t>(v) * image->step + u * 3U;
      image->data[offset + 0U] = static_cast<uint8_t>(20U + u);
      image->data[offset + 1U] = static_cast<uint8_t>(40U + v);
      image->data[offset + 2U] = 60U;
    }
  }
  return image;
}

sensor_msgs::msg::Image::SharedPtr make_depth_image(
  const uint32_t width, const uint32_t height, const float value)
{
  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  depth->width = width;
  depth->height = height;
  depth->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  depth->step = width * sizeof(float);
  depth->data.resize(static_cast<size_t>(depth->step) * height);
  for (size_t offset = 0; offset < depth->data.size(); offset += sizeof(float)) {
    std::memcpy(depth->data.data() + offset, &value, sizeof(float));
  }
  return depth;
}

sensor_msgs::msg::PointCloud2::SharedPtr make_padded_big_endian_cloud()
{
  auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  cloud->width = 2;
  cloud->height = 2;
  cloud->point_step = 16;
  cloud->row_step = 40;  // Eight bytes of padding after each organized row.
  cloud->is_bigendian = true;
  cloud->fields = {
    field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4, sensor_msgs::msg::PointField::FLOAT32),
    field("z", 8, sensor_msgs::msg::PointField::FLOAT32),
    field("rgb", 12, sensor_msgs::msg::PointField::UINT32)};
  cloud->data.assign(static_cast<size_t>(cloud->row_step) * cloud->height, 0xA5U);
  const float xyz[4][3] = {
    {0.0F, 0.0F, 30.0F}, {1.0F, 0.0F, 2.0F},
    {0.0F, 1.0F, 3.0F}, {1.0F, 1.0F, 4.0F}};
  const uint32_t rgb[4] = {0x00112233U, 0x00445566U, 0x00778899U, 0x00AABBCCU};
  for (size_t i = 0; i < 4U; ++i) {
    const size_t row = i / 2U;
    const size_t column = i % 2U;
    const size_t offset = row * cloud->row_step + column * cloud->point_step;
    write_big_endian(cloud->data, offset + 0U, xyz[i][0]);
    write_big_endian(cloud->data, offset + 4U, xyz[i][1]);
    write_big_endian(cloud->data, offset + 8U, xyz[i][2]);
    write_big_endian(cloud->data, offset + 12U, rgb[i]);
  }
  return cloud;
}

gaussian_lic_mapping::AlignedRosFrame make_aligned_frame()
{
  gaussian_lic_mapping::AlignedRosFrame frame;
  frame.pointcloud = make_padded_big_endian_cloud();
  frame.image = make_rgb_image(4, 4);
  frame.depth = make_depth_image(4, 4, 0.0F);
  auto pose = std::make_shared<geometry_msgs::msg::PoseStamped>();
  pose->pose.orientation.w = 1.0;
  frame.pose = pose;
  return frame;
}

void test_pointcloud_layout_endianness_and_source_depth_policy()
{
  auto frame = make_aligned_frame();
  const auto converted = gaussian_lic_mapping::convert_aligned_frame(
    frame, 0, 1, gaussian_lic_mapping::CameraIntrinsics{2.0, 2.0, 0.0, 0.0},
    gaussian_lic_mapping::PointCloudCoordinates::kWorld,
    gaussian_lic_mapping::CameraExtrinsics{},
    1.0,  // Must not truncate the source cloud; ROS1 uses this only for SPNet points.
    false, false);
  require(converted.points.size() == 4U, "padded organized cloud did not yield four points");
  require(std::abs(converted.points[0].depth_m - 30.0F) < 1.0e-6F,
    "big-endian FLOAT32 or source max_depth policy is wrong");
  require(std::abs(converted.points[2].xyz_world.y() - 1.0F) < 1.0e-6F,
    "row_step padding was parsed as point data");
  require(std::abs(converted.points[0].color_rgb.x() - 17.0F / 255.0F) < 1.0e-6F,
    "big-endian packed RGB was decoded incorrectly");
  require(converted.skipped_points_max_depth == 0U,
    "source cloud points were incorrectly counted as max-depth rejects");
  require(
    converted.intrinsics.fx == 2.0 && converted.intrinsics.fy == 2.0 &&
    converted.intrinsics.cx == 0.0 && converted.intrinsics.cy == 0.0,
    "frame did not preserve the point-conversion intrinsics snapshot");

  gaussian_lic_mapping::MapperFrameData stored_frame;
  stored_frame.is_keyframe = true;
  stored_frame.intrinsics = gaussian_lic_mapping::CameraIntrinsics{17.0, 18.0, 3.0, 4.0};
  gaussian_lic_mapping::MapperDataset dataset;
  const auto & stored_record = dataset.add_frame(std::move(stored_frame));
  require(
    stored_record.intrinsics.fx == 17.0 && stored_record.intrinsics.fy == 18.0 &&
    stored_record.intrinsics.cx == 3.0 && stored_record.intrinsics.cy == 4.0,
    "MapperDataset did not preserve per-frame intrinsics");

  auto float_color_frame = make_aligned_frame();
  auto float_color_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  float_color_cloud->width = 1;
  float_color_cloud->height = 1;
  float_color_cloud->point_step = 24;
  float_color_cloud->row_step = 24;
  float_color_cloud->fields = {
    field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4, sensor_msgs::msg::PointField::FLOAT32),
    field("z", 8, sensor_msgs::msg::PointField::FLOAT32),
    field("r", 12, sensor_msgs::msg::PointField::FLOAT32),
    field("g", 16, sensor_msgs::msg::PointField::FLOAT32),
    field("b", 20, sensor_msgs::msg::PointField::FLOAT32)};
  float_color_cloud->data.resize(24);
  float float_fields[6] = {0.0F, 0.0F, 2.0F, 0.25F, 0.5F, 0.75F};
  std::memcpy(float_color_cloud->data.data(), float_fields, sizeof(float_fields));
  float_color_frame.pointcloud = float_color_cloud;
  const auto float_color_converted = gaussian_lic_mapping::convert_aligned_frame(
    float_color_frame, 0, 1, gaussian_lic_mapping::CameraIntrinsics{},
    gaussian_lic_mapping::PointCloudCoordinates::kWorld,
    gaussian_lic_mapping::CameraExtrinsics{}, 0.0, false, false);
  require(float_color_converted.points.size() == 1U &&
    std::abs(float_color_converted.points[0].color_rgb.x() - 0.25F) < 1.0e-6F &&
    std::abs(float_color_converted.points[0].color_rgb.z() - 0.75F) < 1.0e-6F,
    "FLOAT32 0..1 RGB fields were incorrectly divided by 255");

  float_fields[3] = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(float_color_cloud->data.data(), float_fields, sizeof(float_fields));
  const auto nan_color_converted = gaussian_lic_mapping::convert_aligned_frame(
    float_color_frame, 0, 1, gaussian_lic_mapping::CameraIntrinsics{},
    gaussian_lic_mapping::PointCloudCoordinates::kWorld,
    gaussian_lic_mapping::CameraExtrinsics{}, 0.0, false, false);
  require(
    nan_color_converted.points.size() == 1U &&
    nan_color_converted.points[0].color_rgb.allFinite() &&
    nan_color_converted.points[0].color_rgb.x() == 0.0F,
    "non-finite FLOAT32 RGB was allowed into the optimizer");

  auto intensity_frame = make_aligned_frame();
  auto intensity_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  intensity_cloud->width = 1;
  intensity_cloud->height = 1;
  intensity_cloud->point_step = 16;
  intensity_cloud->row_step = 16;
  intensity_cloud->fields = {
    field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4, sensor_msgs::msg::PointField::FLOAT32),
    field("z", 8, sensor_msgs::msg::PointField::FLOAT32),
    field("intensity", 12, sensor_msgs::msg::PointField::FLOAT32)};
  intensity_cloud->data.resize(16);
  const float intensity_fields[4] = {
    0.0F, 0.0F, 2.0F, std::numeric_limits<float>::infinity()};
  std::memcpy(intensity_cloud->data.data(), intensity_fields, sizeof(intensity_fields));
  intensity_frame.pointcloud = intensity_cloud;
  const auto nan_intensity_converted = gaussian_lic_mapping::convert_aligned_frame(
    intensity_frame, 0, 1, gaussian_lic_mapping::CameraIntrinsics{0.0, 0.0, 0.0, 0.0},
    gaussian_lic_mapping::PointCloudCoordinates::kWorld,
    gaussian_lic_mapping::CameraExtrinsics{}, 0.0, false, false);
  require(
    nan_intensity_converted.points.size() == 1U &&
    nan_intensity_converted.points[0].color_rgb.allFinite() &&
    nan_intensity_converted.points[0].color_rgb.isZero(0.0F),
    "non-finite FLOAT32 intensity was allowed into the optimizer");

  auto malformed = make_aligned_frame();
  auto mutable_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(*malformed.pointcloud);
  mutable_cloud->row_step = 31;
  malformed.pointcloud = mutable_cloud;
  bool threw = false;
  try {
    (void)gaussian_lic_mapping::convert_aligned_frame(malformed, 0, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "invalid row_step was accepted");

  malformed = make_aligned_frame();
  mutable_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(*malformed.pointcloud);
  mutable_cloud->fields[2].offset = 15;
  malformed.pointcloud = mutable_cloud;
  threw = false;
  try {
    (void)gaussian_lic_mapping::convert_aligned_frame(malformed, 0, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "field extending beyond point_step was accepted");

  malformed = make_aligned_frame();
  mutable_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(*malformed.pointcloud);
  mutable_cloud->fields[0].count = 2;
  malformed.pointcloud = mutable_cloud;
  threw = false;
  try {
    (void)gaussian_lic_mapping::convert_aligned_frame(malformed, 0, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "multi-count scalar x field was accepted");

  malformed = make_aligned_frame();
  mutable_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(*malformed.pointcloud);
  mutable_cloud->data.pop_back();
  malformed.pointcloud = mutable_cloud;
  threw = false;
  try {
    (void)gaussian_lic_mapping::convert_aligned_frame(malformed, 0, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "truncated PointCloud2 data was accepted");

  malformed = make_aligned_frame();
  auto invalid_pose = std::make_shared<geometry_msgs::msg::PoseStamped>(*malformed.pose);
  invalid_pose->pose.orientation.w = 0.0;
  malformed.pose = invalid_pose;
  threw = false;
  try {
    (void)gaussian_lic_mapping::convert_aligned_frame(malformed, 0, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "zero-norm pose quaternion was accepted");

  malformed = make_aligned_frame();
  invalid_pose = std::make_shared<geometry_msgs::msg::PoseStamped>(*malformed.pose);
  invalid_pose->pose.orientation.x = std::numeric_limits<double>::quiet_NaN();
  malformed.pose = invalid_pose;
  threw = false;
  try {
    (void)gaussian_lic_mapping::convert_aligned_frame(malformed, 0, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "non-finite pose quaternion was accepted");
}

gaussian_lic_mapping::MapperFrameData make_completion_frame(const bool is_keyframe)
{
  gaussian_lic_mapping::MapperFrameData frame;
  frame.is_keyframe = is_keyframe;
  frame.width = 4;
  frame.height = 4;
  frame.image_rgb_float = cv::Mat(4, 4, CV_32FC3, cv::Scalar(0.2F, 0.4F, 0.6F));
  frame.depth_m_float = cv::Mat::zeros(4, 4, CV_32FC1);
  frame.depth_m_float.at<float>(0, 0) = 2.0F;
  frame.r_wc = Eigen::Matrix3d::Identity();
  frame.t_wc = Eigen::Vector3d{1.0, 2.0, 3.0};
  return frame;
}

void test_ros1_depth_completion_semantics()
{
  auto frame = make_completion_frame(true);
  const cv::Mat sparse_before = frame.depth_m_float.clone();
  const cv::Mat completed(4, 4, CV_32FC1, cv::Scalar(2.05F));
  const auto result = gaussian_lic_mapping::append_depth_completion_points(
    frame, completed, gaussian_lic_mapping::CameraIntrinsics{2.0, 2.0, 0.0, 0.0}, 2, 2.1);
  require(result.attempted && result.accepted, "valid keyframe completion was rejected");
  require(result.selected_patch_count == 3U && result.appended_point_count == 3U,
    "completion did not select one point per empty patch");
  require(frame.points.size() == 3U, "completed points were not appended to the frame");
  require(cv::countNonZero(frame.depth_m_float != sparse_before) == 0,
    "completion replaced the sparse optimization depth");
  require(std::abs(frame.points.front().depth_m - 2.0F) < 1.0e-5F,
    "known-depth bias was not removed from completed points");
  require(std::abs(frame.points.front().xyz_world.z() - 5.0F) < 1.0e-5F,
    "completed camera point was not transformed into world coordinates");

  frame = make_completion_frame(true);
  const auto max_depth_result = gaussian_lic_mapping::append_depth_completion_points(
    frame, completed, gaussian_lic_mapping::CameraIntrinsics{2.0, 2.0, 0.0, 0.0}, 2, 1.5);
  require(max_depth_result.selected_patch_count == 3U &&
    max_depth_result.rejected_max_depth_count == 3U && frame.points.empty(),
    "max_depth was not limited to synthesized completion points");

  frame = make_completion_frame(true);
  const cv::Mat biased(4, 4, CV_32FC1, cv::Scalar(3.0F));
  const auto bias_result = gaussian_lic_mapping::append_depth_completion_points(
    frame, biased, gaussian_lic_mapping::CameraIntrinsics{2.0, 2.0, 0.0, 0.0}, 2, 20.0);
  require(bias_result.attempted && !bias_result.accepted && frame.points.empty(),
    "completion with excessive known-depth mean error was accepted");

  frame = make_completion_frame(false);
  const auto non_keyframe_result = gaussian_lic_mapping::append_depth_completion_points(
    frame, completed, gaussian_lic_mapping::CameraIntrinsics{2.0, 2.0, 0.0, 0.0}, 2, 20.0);
  require(!non_keyframe_result.attempted && frame.points.empty(),
    "SPNet completion ran on a non-keyframe");
}

}  // namespace

int main()
{
  try {
    test_pointcloud_layout_endianness_and_source_depth_policy();
    test_ros1_depth_completion_semantics();
    std::cout << "frame_data_parity_probe: PASS\n";
    return 0;
  } catch (const std::exception & ex) {
    std::cerr << "frame_data_parity_probe: FAIL: " << ex.what() << "\n";
    return 1;
  }
}
