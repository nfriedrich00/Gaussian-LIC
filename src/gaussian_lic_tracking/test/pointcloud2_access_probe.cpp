// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include <gaussian_lic_tracking/pointcloud2_access.hpp>

namespace
{

sensor_msgs::msg::PointField field(
  const std::string & name, const std::uint32_t offset, const std::uint8_t datatype)
{
  sensor_msgs::msg::PointField result;
  result.name = name;
  result.offset = offset;
  result.datatype = datatype;
  result.count = 1U;
  return result;
}

bool close(const double lhs, const double rhs)
{
  return std::abs(lhs - rhs) < 1.0e-6;
}

}  // namespace

int main()
{
  using gaussian_lic_tracking::pointcloud2::Layout;
  using gaussian_lic_tracking::pointcloud2::read_numeric;
  using gaussian_lic_tracking::pointcloud2::store_scalar;
  using gaussian_lic_tracking::pointcloud2::validate_layout;
  using gaussian_lic_tracking::pointcloud2::validate_scalar_field;
  using gaussian_lic_tracking::pointcloud2::write_float32;

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.width = 2U;
  cloud.height = 2U;
  cloud.point_step = 16U;
  cloud.row_step = 40U;  // Eight bytes of padding after each organized row.
  cloud.is_bigendian = true;
  cloud.fields = {
    field("x", 0U, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4U, sensor_msgs::msg::PointField::FLOAT32),
    field("z", 8U, sensor_msgs::msg::PointField::FLOAT32),
    field("offset_time", 12U, sensor_msgs::msg::PointField::UINT32)};
  cloud.data.assign(80U, 0xA5U);

  Layout layout;
  std::string reason;
  if (!validate_layout(cloud, layout, &reason) || layout.point_count != 4U) {
    std::cerr << "valid padded layout rejected: " << reason << '\n';
    return 1;
  }
  for (std::size_t index = 0U; index < layout.point_count; ++index) {
    std::size_t base = 0U;
    if (!gaussian_lic_tracking::pointcloud2::point_offset(layout, index, base)) {
      return 2;
    }
    store_scalar<float>(cloud.data.data() + base, static_cast<float>(index + 1U), true);
    store_scalar<float>(cloud.data.data() + base + 4U, static_cast<float>(index + 11U), true);
    store_scalar<float>(cloud.data.data() + base + 8U, static_cast<float>(index + 21U), true);
    store_scalar<std::uint32_t>(
      cloud.data.data() + base + 12U, static_cast<std::uint32_t>(index * 100U), true);
  }

  double value = 0.0;
  if (!read_numeric(cloud, layout, 2U, cloud.fields[0], value) || !close(value, 3.0)) {
    std::cerr << "row_step-aware big-endian x decode failed\n";
    return 3;
  }
  if (!read_numeric(cloud, layout, 2U, cloud.fields[3], value) || !close(value, 200.0)) {
    std::cerr << "row_step-aware big-endian time decode failed\n";
    return 4;
  }
  if (!write_float32(cloud, layout, 3U, cloud.fields[1], 42.5F) ||
    !read_numeric(cloud, layout, 3U, cloud.fields[1], value) || !close(value, 42.5))
  {
    std::cerr << "row_step-aware big-endian write failed\n";
    return 5;
  }

  auto malformed = cloud;
  malformed.row_step = 31U;
  if (validate_layout(malformed, layout, &reason)) {
    std::cerr << "undersized row_step accepted\n";
    return 6;
  }
  malformed = cloud;
  malformed.data.resize(79U);
  if (validate_layout(malformed, layout, &reason)) {
    std::cerr << "truncated data accepted\n";
    return 7;
  }
  if (!validate_layout(cloud, layout, &reason)) {
    return 8;
  }
  auto invalid_field = cloud.fields[0];
  invalid_field.count = 2U;
  if (validate_scalar_field(invalid_field, layout, &reason)) {
    std::cerr << "multi-element scalar field accepted\n";
    return 9;
  }
  invalid_field = cloud.fields[0];
  invalid_field.offset = 14U;
  if (validate_scalar_field(invalid_field, layout, &reason)) {
    std::cerr << "out-of-record field accepted\n";
    return 10;
  }

  std::cout << "pointcloud2_access_probe OK\n";
  return 0;
}
