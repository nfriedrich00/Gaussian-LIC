#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace cocolic
{
inline std::string ResolveConfigPath(const std::string &config_directory,
                                     const std::string &reference)
{
  if (reference.empty())
    throw std::invalid_argument("configuration file reference is empty");
  std::filesystem::path path(reference);
  if (path.is_absolute() && std::filesystem::is_regular_file(path))
    return path.lexically_normal().string();

  // Upstream profiles use '/dataset/file.yaml' to mean config-root relative,
  // rather than a filesystem-root absolute path. Preserve that convention when
  // no real absolute file exists, while also supporting normal relative paths.
  if (path.is_absolute()) path = path.relative_path();
  return (std::filesystem::path(config_directory) / path)
      .lexically_normal().string();
}
} // namespace cocolic
