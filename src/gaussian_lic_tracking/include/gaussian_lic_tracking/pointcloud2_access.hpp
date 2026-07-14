// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace gaussian_lic_tracking::pointcloud2
{

struct Layout
{
  std::size_t width{0U};
  std::size_t height{0U};
  std::size_t point_step{0U};
  std::size_t row_step{0U};
  std::size_t point_count{0U};
};

inline bool checked_multiply(
  const std::size_t lhs, const std::size_t rhs, std::size_t & result)
{
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

inline bool checked_add(
  const std::size_t lhs, const std::size_t rhs, std::size_t & result)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

inline std::size_t scalar_size(const sensor_msgs::msg::PointField & field)
{
  switch (field.datatype) {
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

inline bool validate_layout(
  const sensor_msgs::msg::PointCloud2 & msg,
  Layout & layout,
  std::string * reason = nullptr)
{
  layout = Layout{};
  layout.width = static_cast<std::size_t>(msg.width);
  layout.height = static_cast<std::size_t>(msg.height);
  layout.point_step = static_cast<std::size_t>(msg.point_step);
  layout.row_step = static_cast<std::size_t>(msg.row_step);

  auto fail = [reason](const char * text) {
      if (reason != nullptr) {
        *reason = text;
      }
      return false;
    };

  if (!checked_multiply(layout.width, layout.height, layout.point_count)) {
    return fail("width * height overflows size_t");
  }
  if (layout.point_count == 0U) {
    if (!msg.data.empty()) {
      return fail("an empty cloud must not carry point data");
    }
    return true;
  }
  if (layout.point_step == 0U) {
    return fail("point_step must be positive for a non-empty cloud");
  }
  std::size_t packed_row_size = 0U;
  if (!checked_multiply(layout.width, layout.point_step, packed_row_size)) {
    return fail("width * point_step overflows size_t");
  }
  if (layout.row_step < packed_row_size) {
    return fail("row_step is smaller than width * point_step");
  }
  std::size_t required_data_size = 0U;
  if (!checked_multiply(layout.row_step, layout.height, required_data_size)) {
    return fail("row_step * height overflows size_t");
  }
  if (msg.data.size() < required_data_size) {
    return fail("data is shorter than row_step * height");
  }
  return true;
}

inline bool validate_scalar_field(
  const sensor_msgs::msg::PointField & field,
  const Layout & layout,
  std::string * reason = nullptr)
{
  auto fail = [reason](const char * text) {
      if (reason != nullptr) {
        *reason = text;
      }
      return false;
    };
  if (field.count != 1U) {
    return fail("field count must be exactly one");
  }
  const std::size_t size = scalar_size(field);
  if (size == 0U) {
    return fail("field datatype is not numeric");
  }
  const std::size_t offset = static_cast<std::size_t>(field.offset);
  if (offset > layout.point_step || size > layout.point_step - offset) {
    return fail("field extends past point_step");
  }
  return true;
}

inline bool point_offset(
  const Layout & layout, const std::size_t index, std::size_t & offset)
{
  if (index >= layout.point_count || layout.width == 0U) {
    return false;
  }
  const std::size_t row = index / layout.width;
  const std::size_t column = index % layout.width;
  std::size_t row_offset = 0U;
  std::size_t column_offset = 0U;
  if (!checked_multiply(row, layout.row_step, row_offset) ||
    !checked_multiply(column, layout.point_step, column_offset) ||
    !checked_add(row_offset, column_offset, offset))
  {
    return false;
  }
  return true;
}

inline bool host_is_big_endian()
{
  const std::uint16_t value = 0x0102U;
  std::uint8_t bytes[sizeof(value)]{};
  std::memcpy(bytes, &value, sizeof(value));
  return bytes[0] == 0x01U;
}

template<typename ScalarT>
inline ScalarT load_scalar(const std::uint8_t * bytes, const bool source_is_big_endian)
{
  std::uint8_t native_bytes[sizeof(ScalarT)]{};
  if (source_is_big_endian == host_is_big_endian() || sizeof(ScalarT) == 1U) {
    std::memcpy(native_bytes, bytes, sizeof(ScalarT));
  } else {
    std::reverse_copy(bytes, bytes + sizeof(ScalarT), native_bytes);
  }
  ScalarT value{};
  std::memcpy(&value, native_bytes, sizeof(value));
  return value;
}

template<typename ScalarT>
inline void store_scalar(
  std::uint8_t * bytes, const ScalarT value, const bool destination_is_big_endian)
{
  std::uint8_t native_bytes[sizeof(ScalarT)]{};
  std::memcpy(native_bytes, &value, sizeof(value));
  if (destination_is_big_endian == host_is_big_endian() || sizeof(ScalarT) == 1U) {
    std::memcpy(bytes, native_bytes, sizeof(ScalarT));
  } else {
    std::reverse_copy(native_bytes, native_bytes + sizeof(ScalarT), bytes);
  }
}

inline bool read_numeric(
  const sensor_msgs::msg::PointCloud2 & msg,
  const Layout & layout,
  const std::size_t index,
  const sensor_msgs::msg::PointField & field,
  double & value)
{
  if (!validate_scalar_field(field, layout)) {
    return false;
  }
  std::size_t base = 0U;
  if (!point_offset(layout, index, base)) {
    return false;
  }
  std::size_t field_offset = 0U;
  const std::size_t size = scalar_size(field);
  if (!checked_add(base, static_cast<std::size_t>(field.offset), field_offset) ||
    field_offset > msg.data.size() || size > msg.data.size() - field_offset)
  {
    return false;
  }
  const std::uint8_t * bytes = msg.data.data() + field_offset;
  switch (field.datatype) {
    case sensor_msgs::msg::PointField::INT8:
      value = static_cast<double>(load_scalar<std::int8_t>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::UINT8:
      value = static_cast<double>(load_scalar<std::uint8_t>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::INT16:
      value = static_cast<double>(load_scalar<std::int16_t>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::UINT16:
      value = static_cast<double>(load_scalar<std::uint16_t>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::INT32:
      value = static_cast<double>(load_scalar<std::int32_t>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::UINT32:
      value = static_cast<double>(load_scalar<std::uint32_t>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::FLOAT32:
      value = static_cast<double>(load_scalar<float>(bytes, msg.is_bigendian));
      return true;
    case sensor_msgs::msg::PointField::FLOAT64:
      value = load_scalar<double>(bytes, msg.is_bigendian);
      return true;
    default:
      return false;
  }
}

inline bool write_float32(
  sensor_msgs::msg::PointCloud2 & msg,
  const Layout & layout,
  const std::size_t index,
  const sensor_msgs::msg::PointField & field,
  const float value)
{
  if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    !validate_scalar_field(field, layout))
  {
    return false;
  }
  std::size_t base = 0U;
  std::size_t field_offset = 0U;
  if (!point_offset(layout, index, base) ||
    !checked_add(base, static_cast<std::size_t>(field.offset), field_offset) ||
    field_offset > msg.data.size() || sizeof(float) > msg.data.size() - field_offset)
  {
    return false;
  }
  store_scalar<float>(msg.data.data() + field_offset, value, msg.is_bigendian);
  return true;
}

}  // namespace gaussian_lic_tracking::pointcloud2
