// SPDX-License-Identifier: GPL-3.0-or-later

// ROS2_PORT_NOTE [NEW][SAFETY]: ROS2-native PointCloud2 schema/overflow/
// row-padding validation for the adapter.  It has no Gaussian-LIC ROS1 source
// counterpart and does not estimate motion.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace gaussian_lic_frontend
{

struct PointCloudXyzLayout
{
  const sensor_msgs::msg::PointField * x{nullptr};
  const sensor_msgs::msg::PointField * y{nullptr};
  const sensor_msgs::msg::PointField * z{nullptr};
  size_t point_count{0U};
  size_t packed_row_size{0U};
  size_t required_data_size{0U};
};

inline size_t checked_multiply(const size_t lhs, const size_t rhs, const char * label)
{
  if (lhs != 0U && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::runtime_error(std::string("PointCloud2 ") + label + " overflows size_t");
  }
  return lhs * rhs;
}

inline size_t point_field_datatype_size(const uint8_t datatype)
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
      return 0U;
  }
}

inline const sensor_msgs::msg::PointField * require_unique_xyz_field(
  const sensor_msgs::msg::PointCloud2 & cloud, const char * name)
{
  const sensor_msgs::msg::PointField * result = nullptr;
  for (const auto & field : cloud.fields) {
    if (field.name != name) {
      continue;
    }
    if (result != nullptr) {
      throw std::runtime_error(std::string("PointCloud2 contains duplicate '") + name + "' fields");
    }
    result = &field;
  }
  if (result == nullptr) {
    throw std::runtime_error(std::string("PointCloud2 is missing the '") + name + "' field");
  }
  if (result->count != 1U) {
    throw std::runtime_error(std::string("PointCloud2 '") + name + "' field count must be 1");
  }
  if (
    result->datatype != sensor_msgs::msg::PointField::FLOAT32 &&
    result->datatype != sensor_msgs::msg::PointField::FLOAT64)
  {
    throw std::runtime_error(
            std::string("PointCloud2 '") + name + "' field must be FLOAT32 or FLOAT64");
  }
  const size_t field_size = point_field_datatype_size(result->datatype);
  if (
    result->offset > cloud.point_step ||
    field_size > static_cast<size_t>(cloud.point_step) - result->offset)
  {
    throw std::runtime_error(
            std::string("PointCloud2 '") + name + "' field extends past point_step");
  }
  return result;
}

inline PointCloudXyzLayout validate_pointcloud_xyz_layout(
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  if (cloud.point_step == 0U) {
    throw std::runtime_error("PointCloud2 point_step must be positive");
  }
  if (cloud.height == 0U && cloud.width != 0U) {
    throw std::runtime_error("PointCloud2 height is zero while width is non-zero");
  }

  PointCloudXyzLayout layout;
  layout.packed_row_size = checked_multiply(
    static_cast<size_t>(cloud.width), static_cast<size_t>(cloud.point_step), "row size");
  if (static_cast<size_t>(cloud.row_step) < layout.packed_row_size) {
    throw std::runtime_error("PointCloud2 row_step is smaller than width * point_step");
  }
  layout.required_data_size = checked_multiply(
    static_cast<size_t>(cloud.height), static_cast<size_t>(cloud.row_step), "data size");
  if (cloud.data.size() != layout.required_data_size) {
    throw std::runtime_error(
            "PointCloud2 data length must equal height * row_step (including row padding)");
  }
  layout.point_count = checked_multiply(
    static_cast<size_t>(cloud.width), static_cast<size_t>(cloud.height), "point count");
  for (const auto & field : cloud.fields) {
    const size_t datatype_size = point_field_datatype_size(field.datatype);
    if (datatype_size == 0U) {
      throw std::runtime_error(
              "PointCloud2 field '" + field.name + "' has an unknown datatype");
    }
    if (field.count == 0U) {
      throw std::runtime_error(
              "PointCloud2 field '" + field.name + "' count must be positive");
    }
    const size_t field_size = checked_multiply(
      datatype_size, static_cast<size_t>(field.count), "field size");
    if (
      field.offset > cloud.point_step ||
      field_size > static_cast<size_t>(cloud.point_step) - field.offset)
    {
      throw std::runtime_error(
              "PointCloud2 field '" + field.name + "' extends past point_step");
    }
  }
  layout.x = require_unique_xyz_field(cloud, "x");
  layout.y = require_unique_xyz_field(cloud, "y");
  layout.z = require_unique_xyz_field(cloud, "z");
  return layout;
}

inline size_t point_offset(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const PointCloudXyzLayout & layout,
  const size_t row,
  const size_t column)
{
  if (row >= cloud.height || column >= cloud.width) {
    throw std::out_of_range("PointCloud2 row/column is outside the validated layout");
  }
  const size_t offset = row * static_cast<size_t>(cloud.row_step) +
    column * static_cast<size_t>(cloud.point_step);
  if (offset + cloud.point_step > layout.required_data_size) {
    throw std::runtime_error("PointCloud2 point extends past validated data length");
  }
  return offset;
}

inline bool host_is_big_endian()
{
  const uint16_t marker = 0x0102U;
  return reinterpret_cast<const uint8_t *>(&marker)[0] == 0x01U;
}

template<typename T>
inline T read_endian_value(const uint8_t * ptr, const bool data_is_big_endian)
{
  std::array<uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), ptr, bytes.size());
  if (data_is_big_endian != host_is_big_endian()) {
    std::reverse(bytes.begin(), bytes.end());
  }
  T value{};
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

template<typename T>
inline void write_endian_value(uint8_t * ptr, const T value, const bool data_is_big_endian)
{
  std::array<uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  if (data_is_big_endian != host_is_big_endian()) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::memcpy(ptr, bytes.data(), bytes.size());
}

inline double read_float_field(
  const uint8_t * base,
  const sensor_msgs::msg::PointField & field,
  const bool data_is_big_endian)
{
  const uint8_t * ptr = base + field.offset;
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    return static_cast<double>(read_endian_value<float>(ptr, data_is_big_endian));
  }
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT64) {
    return read_endian_value<double>(ptr, data_is_big_endian);
  }
  throw std::runtime_error("PointCloud2 xyz field is not FLOAT32/FLOAT64 after validation");
}

inline void write_float_field(
  uint8_t * base,
  const sensor_msgs::msg::PointField & field,
  const double value,
  const bool data_is_big_endian)
{
  uint8_t * ptr = base + field.offset;
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    write_endian_value<float>(ptr, static_cast<float>(value), data_is_big_endian);
    return;
  }
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT64) {
    write_endian_value<double>(ptr, value, data_is_big_endian);
    return;
  }
  throw std::runtime_error("PointCloud2 xyz field is not FLOAT32/FLOAT64 after validation");
}

}  // namespace gaussian_lic_frontend
