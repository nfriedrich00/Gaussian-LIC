#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utils/config_path.h>
#include <utils/yaml_utils.h>

int main()
{
  try
  {
    const std::filesystem::path root(COCOLIC_SOURCE_CONFIG_DIR);
    size_t profile_count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(root))
    {
      if (!entry.is_regular_file()) continue;
      const std::string name = entry.path().filename().string();
      if (name.rfind("ct_odometry_", 0) != 0 || entry.path().extension() != ".yaml")
        continue;
      ++profile_count;
      const YAML::Node main = YAML::LoadFile((root / name).string());
      for (const auto *key : {"odometry_mode", "bag_path", "bag_start", "bag_durr",
                              "imu_yaml", "lidar_yaml", "camera_yaml", "t_add",
                              "distance0", "non_uniform", "division_coarse",
                              "t_begin_add_cam", "lidar_iter", "use_lidar_scale",
                              "is_evo_viral", "if_3dgs", "lidar_skip"})
        if (!main[key]) throw std::runtime_error(name + " misses " + key);

      (void)yaml::RequirePositive<double>(main, "t_add", name);
      (void)yaml::RequirePositive<int>(main, "division_coarse", name);
      (void)yaml::RequirePositive<int>(main, "lidar_iter", name);

      for (const auto *key : {"imu_yaml", "lidar_yaml", "camera_yaml"})
      {
        const auto referenced = cocolic::ResolveConfigPath(
            root.string(), main[key].as<std::string>());
        if (!std::filesystem::is_regular_file(referenced))
          throw std::runtime_error(name + " references missing " + referenced);
        (void)YAML::LoadFile(referenced);
      }

      const YAML::Node lidar = YAML::LoadFile(cocolic::ResolveConfigPath(
          root.string(), main["lidar_yaml"].as<std::string>()));
      (void)yaml::RequirePositive<int>(lidar, "num_lidars", name + ".lidar");
      (void)yaml::RequirePositive<int>(
          lidar["current_scan_param"], "correspondence_downsample",
          name + ".lidar.current_scan_param");
    }
    if (profile_count < 8)
      throw std::runtime_error("expected at least 8 installed odometry profiles");

    const auto expected = (root / "fastlivo2/imu.yaml").lexically_normal().string();
    if (cocolic::ResolveConfigPath(root.string(), "fastlivo2/imu.yaml") != expected ||
        cocolic::ResolveConfigPath(root.string(), "/fastlivo2/imu.yaml") != expected ||
        cocolic::ResolveConfigPath(root.string(), expected) != expected)
      throw std::runtime_error("relative/legacy/absolute config path resolution diverged");
  }
  catch (const std::exception &error)
  {
    std::cerr << "profile_config_test: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
