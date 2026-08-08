// SPDX-License-Identifier: GPL-3.0-or-later

// ROS2_PORT_NOTE [ROS_API][SAFETY]: this hand parser intentionally handles
// organized rows, padding, datatype/count/bounds, source endianness and packed
// or scalar colors instead of assuming a PCL host layout.  See the exact input
// contract in docs/ROS1_TO_ROS2_CHANGES_CN.md.

#include <gaussian_lic_mapping/frame_data.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace gaussian_lic_mapping
{
namespace
{

const sensor_msgs::msg::PointField * find_field(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const char * name)
{
  const auto it = std::find_if(
    cloud.fields.begin(), cloud.fields.end(),
    [name](const sensor_msgs::msg::PointField & field) {
      return field.name == name;
    });
  return it == cloud.fields.end() ? nullptr : &(*it);
}

bool host_is_big_endian()
{
  const uint16_t value = 0x0102U;
  return *reinterpret_cast<const uint8_t *>(&value) == 0x01U;
}

template<typename T>
T read_value(const uint8_t * ptr, const bool source_is_big_endian)
{
  T value{};
  if (sizeof(T) == 1U || source_is_big_endian == host_is_big_endian()) {
    std::memcpy(&value, ptr, sizeof(T));
    return value;
  }

  std::array<uint8_t, sizeof(T)> bytes{};
  std::reverse_copy(ptr, ptr + sizeof(T), bytes.begin());
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

size_t point_field_datatype_size(const uint8_t datatype)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
    case sensor_msgs::msg::PointField::UINT8:
      return 1U;
    case sensor_msgs::msg::PointField::INT16:
    case sensor_msgs::msg::PointField::UINT16:
      return 2U;
    case sensor_msgs::msg::PointField::INT32:
    case sensor_msgs::msg::PointField::UINT32:
    case sensor_msgs::msg::PointField::FLOAT32:
      return 4U;
    case sensor_msgs::msg::PointField::FLOAT64:
      return 8U;
    default:
      throw std::runtime_error("unsupported PointCloud2 field datatype");
  }
}

void validate_point_field(
  const sensor_msgs::msg::PointField & field,
  const uint32_t point_step)
{
  if (field.count != 1U) {
    throw std::runtime_error("PointCloud2 field '" + field.name + "' must have count == 1");
  }
  const size_t element_size = point_field_datatype_size(field.datatype);
  const size_t field_size = element_size * static_cast<size_t>(field.count);
  if (
    static_cast<size_t>(field.offset) > static_cast<size_t>(point_step) ||
    field_size > static_cast<size_t>(point_step) - static_cast<size_t>(field.offset))
  {
    throw std::runtime_error(
            "PointCloud2 field '" + field.name + "' exceeds point_step");
  }
}

double read_numeric_field(
  const uint8_t * base,
  const sensor_msgs::msg::PointField & field,
  const bool source_is_big_endian)
{
  const uint8_t * ptr = base + field.offset;
  switch (field.datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return static_cast<double>(read_value<int8_t>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::UINT8:
      return static_cast<double>(read_value<uint8_t>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::INT16:
      return static_cast<double>(read_value<int16_t>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::UINT16:
      return static_cast<double>(read_value<uint16_t>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::INT32:
      return static_cast<double>(read_value<int32_t>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::UINT32:
      return static_cast<double>(read_value<uint32_t>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::FLOAT32:
      return static_cast<double>(read_value<float>(ptr, source_is_big_endian));
    case sensor_msgs::msg::PointField::FLOAT64:
      return read_value<double>(ptr, source_is_big_endian);
    default:
      throw std::runtime_error("unsupported PointCloud2 field datatype");
  }
}

uint32_t read_rgb_bits(
  const uint8_t * base,
  const sensor_msgs::msg::PointField & field,
  const bool source_is_big_endian)
{
  const uint8_t * ptr = base + field.offset;
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    const float packed = read_value<float>(ptr, source_is_big_endian);
    uint32_t bits = 0;
    std::memcpy(&bits, &packed, sizeof(bits));
    return bits;
  }
  if (field.datatype == sensor_msgs::msg::PointField::UINT32 ||
    field.datatype == sensor_msgs::msg::PointField::INT32)
  {
    return read_value<uint32_t>(ptr, source_is_big_endian);
  }
  throw std::runtime_error("rgb/rgba PointCloud2 field must be FLOAT32 or UINT32");
}

bool has_valid_projection_intrinsics(const CameraIntrinsics & intrinsics)
{
  return std::isfinite(intrinsics.fx) && std::isfinite(intrinsics.fy) &&
    std::isfinite(intrinsics.cx) && std::isfinite(intrinsics.cy) &&
    intrinsics.fx > 0.0 && intrinsics.fy > 0.0;
}

bool project_to_image_pixel(
  const int width,
  const int height,
  const CameraIntrinsics & intrinsics,
  const Eigen::Vector3d & xyz_cam,
  int & u,
  int & v)
{
  if (width <= 0 || height <= 0 || !has_valid_projection_intrinsics(intrinsics) || xyz_cam.z() <= 0.0)
  {
    return false;
  }

  const double u_float = intrinsics.fx * xyz_cam.x() / xyz_cam.z() + intrinsics.cx;
  const double v_float = intrinsics.fy * xyz_cam.y() / xyz_cam.z() + intrinsics.cy;
  if (!std::isfinite(u_float) || !std::isfinite(v_float)) {
    return false;
  }

  u = static_cast<int>(std::floor(u_float));
  v = static_cast<int>(std::floor(v_float));
  return u >= 0 && v >= 0 && u < width && v < height;
}

bool sample_projected_image_color(
  const cv::Mat & image_rgb_float,
  const CameraIntrinsics & intrinsics,
  const Eigen::Vector3d & xyz_cam,
  Eigen::Vector3f & color_rgb,
  int * projected_u = nullptr,
  int * projected_v = nullptr)
{
  if (image_rgb_float.empty() || image_rgb_float.type() != CV_32FC3) {
    return false;
  }

  int u = 0;
  int v = 0;
  if (!project_to_image_pixel(image_rgb_float.cols, image_rgb_float.rows, intrinsics, xyz_cam, u, v)) {
    return false;
  }
  if (projected_u) {
    *projected_u = u;
  }
  if (projected_v) {
    *projected_v = v;
  }

  const cv::Vec3f & rgb = image_rgb_float.at<cv::Vec3f>(v, u);
  if (!std::isfinite(rgb[0]) || !std::isfinite(rgb[1]) || !std::isfinite(rgb[2])) {
    return false;
  }
  color_rgb.x() = std::clamp(rgb[0], 0.0F, 1.0F);
  color_rgb.y() = std::clamp(rgb[1], 0.0F, 1.0F);
  color_rgb.z() = std::clamp(rgb[2], 0.0F, 1.0F);
  return true;
}

float normalize_intensity_color(
  const double value,
  const sensor_msgs::msg::PointField & field)
{
  if (!std::isfinite(value)) {
    return 0.0F;
  }
  double normalized = value;
  switch (field.datatype) {
    case sensor_msgs::msg::PointField::UINT8:
      normalized = value / 255.0;
      break;
    case sensor_msgs::msg::PointField::UINT16:
      normalized = value / 65535.0;
      break;
    case sensor_msgs::msg::PointField::UINT32:
      normalized = value / 4294967295.0;
      break;
    default:
      if (value > 1.0) {
        normalized = value / 255.0;
      }
      break;
  }
  return static_cast<float>(std::clamp(normalized, 0.0, 1.0));
}

bool sample_intensity_color(
  const uint8_t * base,
  const sensor_msgs::msg::PointField * intensity_field,
  const bool source_is_big_endian,
  Eigen::Vector3f & color_rgb)
{
  if (!intensity_field) {
    return false;
  }
  const float intensity = normalize_intensity_color(
    read_numeric_field(base, *intensity_field, source_is_big_endian), *intensity_field);
  color_rgb = Eigen::Vector3f::Constant(intensity);
  return true;
}

cv::Mat make_projected_depth_image(
  const std::vector<MapperPoint> & points,
  const int width,
  const int height,
  const CameraIntrinsics & intrinsics,
  const Eigen::Quaterniond & q_wc,
  const Eigen::Vector3d & t_wc)
{
  cv::Mat depth_m(height, width, CV_32FC1, cv::Scalar(0.0F));
  if (width <= 0 || height <= 0 || !has_valid_projection_intrinsics(intrinsics)) {
    return depth_m;
  }

  const Eigen::Matrix3d r_cw = q_wc.toRotationMatrix().transpose();
  for (const MapperPoint & point : points) {
    const Eigen::Vector3d xyz_world = point.xyz_world.cast<double>();
    const Eigen::Vector3d xyz_cam = r_cw * (xyz_world - t_wc);
    if (xyz_cam.z() <= 0.0) {
      continue;
    }

    const double u_float = intrinsics.fx * xyz_cam.x() / xyz_cam.z() + intrinsics.cx;
    const double v_float = intrinsics.fy * xyz_cam.y() / xyz_cam.z() + intrinsics.cy;
    if (!std::isfinite(u_float) || !std::isfinite(v_float)) {
      continue;
    }

    // Match scripts/frontend_raw_to_ros1_mapper_contract.py::make_projected_depth.
    // RGB sampling intentionally stays floor-based because the ROS1 converter uses floor there.
    const int u = static_cast<int>(std::nearbyint(u_float));
    const int v = static_cast<int>(std::nearbyint(v_float));
    if (u < 0 || v < 0 || u >= width || v >= height) {
      continue;
    }

    float & current_depth = depth_m.at<float>(v, u);
    const float candidate_depth = static_cast<float>(xyz_cam.z());
    if (current_depth <= 0.0F || candidate_depth < current_depth) {
      current_depth = candidate_depth;
    }
  }

  return depth_m;
}

}  // namespace

cv::Mat convert_image_to_rgb_float(const sensor_msgs::msg::Image & image_msg)
{
  namespace enc = sensor_msgs::image_encodings;

  cv_bridge::CvImagePtr cv_ptr;
  if (image_msg.encoding == enc::BGR8) {
    cv_ptr = cv_bridge::toCvCopy(image_msg, enc::BGR8);
    cv::Mat rgb;
    cv::cvtColor(cv_ptr->image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
  }

  if (image_msg.encoding == enc::RGB8) {
    cv_ptr = cv_bridge::toCvCopy(image_msg, enc::RGB8);
    cv::Mat rgb;
    cv_ptr->image.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
  }

  if (image_msg.encoding == enc::MONO8) {
    cv_ptr = cv_bridge::toCvCopy(image_msg, enc::MONO8);
    cv::Mat rgb_u8;
    cv::cvtColor(cv_ptr->image, rgb_u8, cv::COLOR_GRAY2RGB);
    cv::Mat rgb;
    rgb_u8.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
  }

  throw std::runtime_error("unsupported image encoding for Gaussian-LIC mapper: " + image_msg.encoding);
}

namespace
{

cv::Mat convert_depth_to_float_m(const sensor_msgs::msg::Image & depth_msg)
{
  namespace enc = sensor_msgs::image_encodings;

  if (depth_msg.encoding == enc::TYPE_32FC1) {
    return cv_bridge::toCvCopy(depth_msg, enc::TYPE_32FC1)->image;
  }

  if (depth_msg.encoding == enc::TYPE_16UC1 || depth_msg.encoding == enc::MONO16) {
    cv::Mat depth_m;
    cv_bridge::toCvCopy(depth_msg, depth_msg.encoding)->image.convertTo(depth_m, CV_32FC1, 0.001);
    return depth_m;
  }

  throw std::runtime_error("unsupported depth encoding for Gaussian-LIC mapper: " + depth_msg.encoding);
}

std::vector<MapperPoint> convert_pointcloud(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const Eigen::Quaterniond & q_w_pose,
  const Eigen::Vector3d & t_w_pose,
  const Eigen::Quaterniond & q_wc,
  const Eigen::Vector3d & t_wc,
  const cv::Mat & image_rgb_float,
  const CameraIntrinsics & intrinsics,
  const PointCloudCoordinates pointcloud_coordinates,
  const CameraExtrinsics & camera_extrinsics,
  const double max_depth_m,
  const bool require_projected_color,
  const bool zbuffer_projected_points,
  size_t & skipped_nonpositive_depth,
  size_t & skipped_max_depth,
  size_t & skipped_unprojected,
  size_t & skipped_occluded)
{
  skipped_nonpositive_depth = 0;
  skipped_max_depth = 0;
  skipped_unprojected = 0;
  skipped_occluded = 0;
  // ROS1 Gaussian-LIC applies max_depth only to points synthesized by SPNet.
  // Keep this argument for source compatibility, but never truncate the input
  // LiDAR/source cloud here.
  (void)max_depth_m;
  const size_t point_count = static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
  std::vector<MapperPoint> points;
  points.reserve(point_count);

  if (point_count == 0) {
    return points;
  }
  if (cloud.point_step == 0U) {
    throw std::runtime_error("PointCloud2 point_step must be positive");
  }
  const size_t minimum_row_step = static_cast<size_t>(cloud.width) * cloud.point_step;
  if (cloud.row_step < minimum_row_step) {
    throw std::runtime_error("PointCloud2 row_step is smaller than width * point_step");
  }
  if (
    cloud.height != 0U &&
    static_cast<size_t>(cloud.row_step) >
    std::numeric_limits<size_t>::max() / static_cast<size_t>(cloud.height))
  {
    throw std::runtime_error("PointCloud2 data size calculation overflow");
  }
  const size_t required_data_size = static_cast<size_t>(cloud.row_step) * cloud.height;
  if (cloud.data.size() < required_data_size) {
    throw std::runtime_error("PointCloud2 data buffer is smaller than row_step * height");
  }

  const auto * x_field = find_field(cloud, "x");
  const auto * y_field = find_field(cloud, "y");
  const auto * z_field = find_field(cloud, "z");
  if (!x_field || !y_field || !z_field) {
    throw std::runtime_error("PointCloud2 must contain x/y/z fields");
  }

  const auto * rgb_field = find_field(cloud, "rgb");
  if (!rgb_field) {
    rgb_field = find_field(cloud, "rgba");
  }
  const auto * r_field = find_field(cloud, "r");
  const auto * g_field = find_field(cloud, "g");
  const auto * b_field = find_field(cloud, "b");
  const auto * intensity_field = find_field(cloud, "intensity");
  if (!intensity_field) {
    intensity_field = find_field(cloud, "reflectivity");
  }
  const bool has_explicit_color = rgb_field || (r_field && g_field && b_field);
  for (const auto * field : {
      x_field, y_field, z_field, rgb_field, r_field, g_field, b_field, intensity_field})
  {
    if (field) {
      validate_point_field(*field, cloud.point_step);
    }
  }
  const bool use_projected_zbuffer =
    zbuffer_projected_points && require_projected_color && !has_explicit_color && !image_rgb_float.empty() &&
    image_rgb_float.type() == CV_32FC3 && has_valid_projection_intrinsics(intrinsics);
  std::vector<int> nearest_point_by_pixel;
  std::vector<double> nearest_depth_by_pixel;
  if (use_projected_zbuffer) {
    const size_t pixel_count =
      static_cast<size_t>(image_rgb_float.cols) * static_cast<size_t>(image_rgb_float.rows);
    nearest_point_by_pixel.assign(pixel_count, -1);
    nearest_depth_by_pixel.assign(pixel_count, std::numeric_limits<double>::infinity());
  }

  const Eigen::Matrix3d r_cw = q_wc.toRotationMatrix().transpose();
  const Eigen::Vector3d t_cw = -r_cw * t_wc;
  const Eigen::Matrix3d r_w_pose = q_w_pose.toRotationMatrix();
  const Eigen::Quaterniond q_camera_pose = camera_extrinsics.q_pose_camera.inverse();

  for (size_t i = 0; i < point_count; ++i) {
    const size_t row = i / static_cast<size_t>(cloud.width);
    const size_t column = i % static_cast<size_t>(cloud.width);
    const uint8_t * base = cloud.data.data() + row * cloud.row_step + column * cloud.point_step;

    const Eigen::Vector3d xyz_input{
      read_numeric_field(base, *x_field, cloud.is_bigendian),
      read_numeric_field(base, *y_field, cloud.is_bigendian),
      read_numeric_field(base, *z_field, cloud.is_bigendian)};

    if (!xyz_input.allFinite()) {
      ++skipped_nonpositive_depth;
      continue;
    }

    Eigen::Vector3d xyz_world = xyz_input;
    Eigen::Vector3d xyz_cam = xyz_input;
    if (pointcloud_coordinates == PointCloudCoordinates::kSensor) {
      xyz_world = r_w_pose * xyz_input + t_w_pose;
      xyz_cam = q_camera_pose * (xyz_input - camera_extrinsics.p_pose_camera);
    } else {
      xyz_cam = r_cw * xyz_world + t_cw;
    }
    if (xyz_cam.z() <= 0.0) {
      ++skipped_nonpositive_depth;
      continue;
    }
    Eigen::Vector3f color_rgb{1.0F, 1.0F, 1.0F};
    int projected_u = -1;
    int projected_v = -1;
    if (rgb_field) {
      const uint32_t rgb = read_rgb_bits(base, *rgb_field, cloud.is_bigendian);
      color_rgb.x() = static_cast<float>((rgb >> 16U) & 0xFFU) / 255.0F;
      color_rgb.y() = static_cast<float>((rgb >> 8U) & 0xFFU) / 255.0F;
      color_rgb.z() = static_cast<float>(rgb & 0xFFU) / 255.0F;
    } else if (r_field && g_field && b_field) {
      color_rgb.x() = normalize_intensity_color(
        read_numeric_field(base, *r_field, cloud.is_bigendian), *r_field);
      color_rgb.y() = normalize_intensity_color(
        read_numeric_field(base, *g_field, cloud.is_bigendian), *g_field);
      color_rgb.z() = normalize_intensity_color(
        read_numeric_field(base, *b_field, cloud.is_bigendian), *b_field);
    } else if (!sample_projected_image_color(
        image_rgb_float, intrinsics, xyz_cam, color_rgb, &projected_u, &projected_v))
    {
      if (require_projected_color) {
        ++skipped_unprojected;
        continue;
      }
      (void)sample_intensity_color(base, intensity_field, cloud.is_bigendian, color_rgb);
    }

    MapperPoint point{
      xyz_world.cast<float>(),
      color_rgb,
      static_cast<float>(xyz_cam.z())};
    if (use_projected_zbuffer) {
      if (projected_u < 0 || projected_v < 0) {
        ++skipped_unprojected;
        continue;
      }
      const size_t pixel_index =
        static_cast<size_t>(projected_v) * static_cast<size_t>(image_rgb_float.cols) +
        static_cast<size_t>(projected_u);
      if (pixel_index >= nearest_point_by_pixel.size()) {
        ++skipped_unprojected;
        continue;
      }
      const int previous_index = nearest_point_by_pixel[pixel_index];
      if (previous_index >= 0 && xyz_cam.z() >= nearest_depth_by_pixel[pixel_index]) {
        ++skipped_occluded;
        continue;
      }
      if (previous_index >= 0) {
        points[static_cast<size_t>(previous_index)] = point;
        nearest_depth_by_pixel[pixel_index] = xyz_cam.z();
        ++skipped_occluded;
        continue;
      }
      nearest_point_by_pixel[pixel_index] = static_cast<int>(points.size());
      nearest_depth_by_pixel[pixel_index] = xyz_cam.z();
    }
    points.push_back(point);
  }

  return points;
}

}  // namespace

DepthCompletionResult append_depth_completion_points(
  MapperFrameData & frame,
  const cv::Mat & completed_depth_m,
  const CameraIntrinsics & intrinsics,
  const int patch_size,
  const double max_depth_m,
  const double known_depth_mean_tolerance_m,
  const double sobel_edge_threshold)
{
  DepthCompletionResult result;
  if (!frame.is_keyframe) {
    return result;
  }
  result.attempted = true;
  if (patch_size <= 0) {
    throw std::runtime_error("depth completion patch_size must be positive");
  }
  if (
    frame.depth_m_float.empty() || completed_depth_m.empty() ||
    frame.depth_m_float.type() != CV_32FC1 || completed_depth_m.type() != CV_32FC1 ||
    frame.depth_m_float.size() != completed_depth_m.size())
  {
    throw std::runtime_error("sparse and completed depth must be same-sized CV_32FC1 images");
  }
  if (
    frame.image_rgb_float.empty() || frame.image_rgb_float.type() != CV_32FC3 ||
    frame.image_rgb_float.size() != completed_depth_m.size())
  {
    throw std::runtime_error("depth completion requires a same-sized CV_32FC3 RGB image");
  }
  if (!has_valid_projection_intrinsics(intrinsics)) {
    throw std::runtime_error("depth completion requires positive finite camera intrinsics");
  }

  const cv::Mat known_mask = frame.depth_m_float > 0.0F;
  cv::Mat completed_at_known;
  completed_depth_m.copyTo(completed_at_known, known_mask);
  const cv::Mat depth_difference = completed_at_known - frame.depth_m_float;
  result.mean_known_depth_difference_m = cv::mean(depth_difference, known_mask)[0];
  if (
    !std::isfinite(result.mean_known_depth_difference_m) ||
    std::abs(result.mean_known_depth_difference_m) >= known_depth_mean_tolerance_m)
  {
    return result;
  }

  cv::Mat gradient_x;
  cv::Mat gradient_y;
  cv::Sobel(completed_depth_m, gradient_x, CV_32F, 1, 0, 3);
  cv::Sobel(completed_depth_m, gradient_y, CV_32F, 0, 1, 3);
  cv::Mat gradient_magnitude;
  cv::magnitude(gradient_x, gradient_y, gradient_magnitude);

  cv::Mat bias_corrected = completed_depth_m -
    static_cast<float>(result.mean_known_depth_difference_m);
  const cv::Mat wanted_mask =
    (bias_corrected > 0.0F) & (gradient_magnitude < static_cast<float>(sobel_edge_threshold));
  cv::Mat wanted_depth = cv::Mat::zeros(completed_depth_m.size(), CV_32FC1);
  bias_corrected.copyTo(wanted_depth, wanted_mask);

  result.accepted = true;
  const int height = frame.depth_m_float.rows;
  const int width = frame.depth_m_float.cols;
  for (int patch_v = 0; patch_v < height; patch_v += patch_size) {
    for (int patch_u = 0; patch_u < width; patch_u += patch_size) {
      const int v_end = std::min(patch_v + patch_size, height);
      const int u_end = std::min(patch_u + patch_size, width);
      bool has_sparse_depth = false;
      float minimum_depth = std::numeric_limits<float>::max();
      int selected_u = -1;
      int selected_v = -1;
      for (int v = patch_v; v < v_end && !has_sparse_depth; ++v) {
        const float * sparse_row = frame.depth_m_float.ptr<float>(v);
        const float * wanted_row = wanted_depth.ptr<float>(v);
        for (int u = patch_u; u < u_end; ++u) {
          if (sparse_row[u] > 0.0F) {
            has_sparse_depth = true;
            break;
          }
          if (wanted_row[u] > 0.0F && wanted_row[u] < minimum_depth) {
            minimum_depth = wanted_row[u];
            selected_u = u;
            selected_v = v;
          }
        }
      }
      if (has_sparse_depth || selected_u < 0) {
        continue;
      }

      ++result.selected_patch_count;
      if (minimum_depth > max_depth_m) {
        ++result.rejected_max_depth_count;
        continue;
      }
      const cv::Vec3f & rgb = frame.image_rgb_float.at<cv::Vec3f>(selected_v, selected_u);
      const Eigen::Vector3f color_rgb{
        std::isfinite(rgb[0]) ? std::clamp(rgb[0], 0.0F, 1.0F) : 0.0F,
        std::isfinite(rgb[1]) ? std::clamp(rgb[1], 0.0F, 1.0F) : 0.0F,
        std::isfinite(rgb[2]) ? std::clamp(rgb[2], 0.0F, 1.0F) : 0.0F};
      const Eigen::Vector3d point_camera{
        (static_cast<double>(selected_u) - intrinsics.cx) * minimum_depth / intrinsics.fx,
        (static_cast<double>(selected_v) - intrinsics.cy) * minimum_depth / intrinsics.fy,
        minimum_depth};
      const Eigen::Vector3d point_world = frame.r_wc * point_camera + frame.t_wc;
      frame.points.push_back(MapperPoint{
        point_world.cast<float>(),
        color_rgb,
        minimum_depth});
      ++result.appended_point_count;
    }
  }
  return result;
}

MapperFrameData convert_aligned_frame(
  const AlignedRosFrame & frame,
  const uint64_t frame_index,
  const int select_every_k_frame,
  const CameraIntrinsics & intrinsics,
  const PointCloudCoordinates pointcloud_coordinates,
  const CameraExtrinsics & camera_extrinsics,
  const double max_depth_m,
  const bool require_projected_color,
  const bool zbuffer_projected_points)
{
  if (!frame.pointcloud || !frame.pose || !frame.image) {
    throw std::runtime_error("cannot convert incomplete aligned ROS frame");
  }
  if (select_every_k_frame <= 0) {
    throw std::runtime_error("select_every_k_frame must be positive");
  }

  MapperFrameData out;
  if (frame.has_stamp) {
    out.stamp = frame.stamp;
  } else {
    out.stamp = frame.pointcloud->header.stamp;
  }
  out.pointcloud_stamp = frame.pointcloud->header.stamp;
  out.pose_stamp = frame.pose->header.stamp;
  out.image_stamp = frame.image->header.stamp;
  if (frame.depth) {
    out.depth_stamp = frame.depth->header.stamp;
    out.has_depth_stamp = true;
  }
  out.frame_index = frame_index;
  out.is_keyframe = ((frame_index + 1U) % static_cast<uint64_t>(select_every_k_frame)) == 0U;
  out.image_rgb_float = convert_image_to_rgb_float(*frame.image);
  out.width = out.image_rgb_float.cols;
  out.height = out.image_rgb_float.rows;
  out.intrinsics = intrinsics;

  out.q_wc = Eigen::Quaterniond{
    frame.pose->pose.orientation.w,
    frame.pose->pose.orientation.x,
    frame.pose->pose.orientation.y,
    frame.pose->pose.orientation.z};
  const double pose_quaternion_squared_norm = out.q_wc.squaredNorm();
  if (!std::isfinite(pose_quaternion_squared_norm) || pose_quaternion_squared_norm <= 1.0e-24) {
    throw std::runtime_error("pose orientation must be finite and have non-zero norm");
  }
  out.q_wc.normalize();
  out.t_wc = Eigen::Vector3d{
    frame.pose->pose.position.x,
    frame.pose->pose.position.y,
    frame.pose->pose.position.z};
  if (!out.t_wc.allFinite()) {
    throw std::runtime_error("pose position must be finite");
  }
  const Eigen::Quaterniond q_w_pose = out.q_wc;
  const Eigen::Vector3d t_w_pose = out.t_wc;
  const double extrinsic_quaternion_squared_norm = camera_extrinsics.q_pose_camera.squaredNorm();
  if (
    !std::isfinite(extrinsic_quaternion_squared_norm) ||
    extrinsic_quaternion_squared_norm <= 1.0e-24 ||
    !camera_extrinsics.p_pose_camera.allFinite())
  {
    throw std::runtime_error("camera extrinsics must be finite with a non-zero quaternion");
  }
  const Eigen::Quaterniond q_pose_camera = camera_extrinsics.q_pose_camera.normalized();
  out.q_wc = (q_w_pose * q_pose_camera).normalized();
  out.t_wc = t_w_pose + q_w_pose * camera_extrinsics.p_pose_camera;
  out.r_wc = out.q_wc.toRotationMatrix();

  out.points = convert_pointcloud(
    *frame.pointcloud, q_w_pose, t_w_pose, out.q_wc, out.t_wc, out.image_rgb_float, intrinsics,
    pointcloud_coordinates, camera_extrinsics, max_depth_m, require_projected_color,
    zbuffer_projected_points,
    out.skipped_points_nonpositive_depth, out.skipped_points_max_depth,
    out.skipped_points_unprojected, out.skipped_points_occluded);
  if (frame.depth) {
    out.depth_m_float = convert_depth_to_float_m(*frame.depth);
  } else {
    out.depth_m_float = make_projected_depth_image(
      out.points, out.width, out.height, intrinsics, out.q_wc, out.t_wc);
  }

  return out;
}

}  // namespace gaussian_lic_mapping
