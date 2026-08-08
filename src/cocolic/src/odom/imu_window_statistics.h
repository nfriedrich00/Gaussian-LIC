#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cocolic
{

struct ImuWindowStatistics
{
  std::size_t count = 0;
  Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_accel = Eigen::Vector3d::Zero();
  double gyro_stddev = 0.0;
  double accel_stddev = 0.0;
};

// ROS2_PORT_NOTE [SAFETY][BEHAVIOR]: centralizes the duplicated ROS1 loops,
// makes empty/singleton windows finite and fixes a second-pass comparison that
// mixed maxTimeNURBS() seconds with sample timestamps in nanoseconds.
// Compute statistics for the half-open nanosecond interval [begin_ns, end_ns).
template <typename ImuContainer>
ImuWindowStatistics ComputeImuWindowStatistics(
    const ImuContainer &samples, int64_t begin_ns, int64_t end_ns)
{
  ImuWindowStatistics stats;
  for (const auto &sample : samples)
  {
    if (sample.timestamp < begin_ns || sample.timestamp >= end_ns)
      continue;
    ++stats.count;
    stats.mean_gyro += sample.gyro;
    stats.mean_accel += sample.accel;
  }

  if (stats.count == 0)
    return stats;

  const double count = static_cast<double>(stats.count);
  stats.mean_gyro /= count;
  stats.mean_accel /= count;
  if (stats.count == 1)
    return stats;

  double gyro_squared_error = 0.0;
  double accel_squared_error = 0.0;
  for (const auto &sample : samples)
  {
    if (sample.timestamp < begin_ns || sample.timestamp >= end_ns)
      continue;
    gyro_squared_error += (sample.gyro - stats.mean_gyro).squaredNorm();
    accel_squared_error += (sample.accel - stats.mean_accel).squaredNorm();
  }
  const double sample_denominator = static_cast<double>(stats.count - 1);
  stats.gyro_stddev = std::sqrt(gyro_squared_error / sample_denominator);
  stats.accel_stddev = std::sqrt(accel_squared_error / sample_denominator);
  return stats;
}

}  // namespace cocolic
