#include <odom/imu_window_statistics.h>
#include <utils/yaml_utils.h>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Sample
{
  int64_t timestamp;
  Eigen::Vector3d gyro;
  Eigen::Vector3d accel;
};

template <typename Callable>
void ExpectInvalidArgument(const std::string &description, Callable &&callable)
{
  try
  {
    callable();
  }
  catch (const std::invalid_argument &)
  {
    return;
  }
  throw std::runtime_error(description + " was accepted");
}

void CheckFinite(const cocolic::ImuWindowStatistics &stats)
{
  if (!stats.mean_gyro.allFinite() || !stats.mean_accel.allFinite() ||
      !std::isfinite(stats.gyro_stddev) ||
      !std::isfinite(stats.accel_stddev))
    throw std::runtime_error("IMU window statistics are non-finite");
}
}  // namespace

int main()
{
  try
  {
    const YAML::Node valid = YAML::Load("integer: 2\nreal: 0.1\n");
    if (yaml::RequirePositive<int>(valid, "integer", "test") != 2 ||
        std::abs(yaml::RequirePositive<double>(valid, "real", "test") - 0.1) >
            1e-12)
      throw std::runtime_error("positive configuration value changed");

    const YAML::Node invalid = YAML::Load(
        "zero: 0\nnegative: -1\nnan: .nan\ninf: .inf\nmalformed: text\n");
    ExpectInvalidArgument("zero positive-only configuration", [&] {
      (void)yaml::RequirePositive<int>(invalid, "zero", "test");
    });
    ExpectInvalidArgument("negative positive-only configuration", [&] {
      (void)yaml::RequirePositive<int>(invalid, "negative", "test");
    });
    ExpectInvalidArgument("NaN positive-only configuration", [&] {
      (void)yaml::RequirePositive<double>(invalid, "nan", "test");
    });
    ExpectInvalidArgument("infinite positive-only configuration", [&] {
      (void)yaml::RequirePositive<double>(invalid, "inf", "test");
    });
    ExpectInvalidArgument("malformed positive-only configuration", [&] {
      (void)yaml::RequirePositive<double>(invalid, "malformed", "test");
    });
    ExpectInvalidArgument("missing positive-only configuration", [&] {
      (void)yaml::RequirePositive<int>(invalid, "missing", "test");
    });

    const std::vector<Sample> samples{
        {100, Eigen::Vector3d(99, 99, 99), Eigen::Vector3d(99, 99, 99)},
        {1000000000LL, Eigen::Vector3d(1, 2, 3), Eigen::Vector3d(4, 5, 6)},
        {1500000000LL, Eigen::Vector3d(3, 4, 5), Eigen::Vector3d(6, 7, 8)},
        {2000000000LL, Eigen::Vector3d(77, 77, 77), Eigen::Vector3d(77, 77, 77)}};

    const auto empty =
        cocolic::ComputeImuWindowStatistics(samples, 3000000000LL, 4000000000LL);
    CheckFinite(empty);
    if (empty.count != 0 || empty.gyro_stddev != 0.0 ||
        empty.accel_stddev != 0.0)
      throw std::runtime_error("empty IMU window fallback changed");

    const auto singleton =
        cocolic::ComputeImuWindowStatistics(samples, 1000000000LL, 1100000000LL);
    CheckFinite(singleton);
    if (singleton.count != 1 || singleton.mean_gyro != samples[1].gyro ||
        singleton.gyro_stddev != 0.0 || singleton.accel_stddev != 0.0)
      throw std::runtime_error("singleton IMU window fallback changed");

    const auto pair =
        cocolic::ComputeImuWindowStatistics(samples, 1000000000LL, 2000000000LL);
    CheckFinite(pair);
    if (pair.count != 2 ||
        !pair.mean_gyro.isApprox(Eigen::Vector3d(2, 3, 4)) ||
        !pair.mean_accel.isApprox(Eigen::Vector3d(5, 6, 7)) ||
        std::abs(pair.gyro_stddev - std::sqrt(6.0)) > 1e-12 ||
        std::abs(pair.accel_stddev - std::sqrt(6.0)) > 1e-12)
      throw std::runtime_error("two-sample IMU window statistics are wrong");
  }
  catch (const std::exception &error)
  {
    std::cerr << "numeric_guard_test: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
