// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gaussian_lic_frontend/pointcloud_contract.hpp"

namespace
{

sensor_msgs::msg::PointField field(
  const std::string & name, const uint32_t offset, const uint8_t datatype)
{
  sensor_msgs::msg::PointField result;
  result.name = name;
  result.offset = offset;
  result.datatype = datatype;
  result.count = 1U;
  return result;
}

sensor_msgs::msg::PointCloud2 organized_cloud(const bool big_endian)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.width = 2U;
  cloud.height = 2U;
  cloud.point_step = 16U;
  cloud.row_step = 40U;  // Eight bytes of organized row padding.
  cloud.is_bigendian = big_endian;
  cloud.fields = {
    field("x", 0U, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4U, sensor_msgs::msg::PointField::FLOAT32),
    field("z", 8U, sensor_msgs::msg::PointField::FLOAT32)};
  cloud.data.assign(80U, 0xA5U);
  return cloud;
}

template<typename Fn>
bool throws_with(Fn && fn, const std::string & token)
{
  try {
    fn();
  } catch (const std::exception & ex) {
    return std::string(ex.what()).find(token) != std::string::npos;
  }
  return false;
}

int run_for_endianness(const bool big_endian)
{
  using gaussian_lic_frontend::point_offset;
  using gaussian_lic_frontend::read_float_field;
  using gaussian_lic_frontend::validate_pointcloud_xyz_layout;
  using gaussian_lic_frontend::write_float_field;

  auto cloud = organized_cloud(big_endian);
  const auto layout = validate_pointcloud_xyz_layout(cloud);
  for (size_t row = 0U; row < cloud.height; ++row) {
    for (size_t column = 0U; column < cloud.width; ++column) {
      const size_t offset = point_offset(cloud, layout, row, column);
      uint8_t * base = cloud.data.data() + offset;
      const double value = static_cast<double>(row * 10U + column + 1U);
      write_float_field(base, *layout.x, value, cloud.is_bigendian);
      write_float_field(base, *layout.y, -value, cloud.is_bigendian);
      write_float_field(base, *layout.z, value + 0.5, cloud.is_bigendian);
    }
    for (size_t index = row * cloud.row_step + 32U; index < (row + 1U) * cloud.row_step; ++index) {
      if (cloud.data[index] != 0xA5U) {
        std::cerr << "organized row padding was modified\n";
        return 1;
      }
    }
  }
  const size_t last_offset = point_offset(cloud, layout, 1U, 1U);
  if (std::abs(read_float_field(cloud.data.data() + last_offset, *layout.x, big_endian) - 12.0) > 1e-6) {
    std::cerr << "endian-aware field round trip failed\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main()
{
  using gaussian_lic_frontend::validate_pointcloud_xyz_layout;
  if (run_for_endianness(false) != 0 || run_for_endianness(true) != 0) {
    return 1;
  }

  auto malformed = organized_cloud(false);
  malformed.row_step = 31U;
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "row_step")) {
    std::cerr << "undersized row_step was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.data.pop_back();
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "data length")) {
    std::cerr << "undersized data was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.data.push_back(0U);
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "data length")) {
    std::cerr << "oversized data was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.point_step = 0U;
  malformed.row_step = 0U;
  malformed.data.clear();
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "point_step")) {
    std::cerr << "zero point_step was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.fields[0].count = 2U;
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "count")) {
    std::cerr << "invalid xyz count was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.fields[1].datatype = sensor_msgs::msg::PointField::UINT16;
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "FLOAT32")) {
    std::cerr << "invalid xyz datatype was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.fields[2].offset = 15U;
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "point_step")) {
    std::cerr << "out-of-range xyz field was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.fields.push_back(malformed.fields[0]);
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "duplicate")) {
    std::cerr << "duplicate xyz field was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.fields.push_back(field("intensity", 12U, sensor_msgs::msg::PointField::UINT8));
  malformed.fields.back().count = 0U;
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "count must be positive")) {
    std::cerr << "invalid non-xyz field count was accepted\n";
    return 1;
  }
  malformed = organized_cloud(false);
  malformed.fields.push_back(field("ring", 15U, sensor_msgs::msg::PointField::UINT16));
  if (!throws_with([&]() {validate_pointcloud_xyz_layout(malformed);}, "point_step")) {
    std::cerr << "invalid non-xyz field offset was accepted\n";
    return 1;
  }

  std::cout << "PointCloud2 contract probe passed\n";
  return 0;
}
