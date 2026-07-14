// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_mapping/optimization_sampling.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void require(const bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<Eigen::Vector3d> make_positions(const size_t count, const double final_distance)
{
  std::vector<Eigen::Vector3d> positions(count, Eigen::Vector3d::Zero());
  if (count > 1U) {
    for (size_t index = 0; index < count; ++index) {
      positions[index].x() = final_distance * static_cast<double>(index) /
        static_cast<double>(count - 1U);
    }
  }
  return positions;
}

void check_decayed_case(const size_t frame_count)
{
  const auto positions = make_positions(frame_count, 121.0);
  std::mt19937 rng(42U);
  const auto selected = gaussian_lic_mapping::select_upstream_random_optimization_indices(
    positions, 100U, true, rng);
  require(selected.size() == 50U, "distance decay did not reduce max_iters 100 to 50");

  const size_t split = frame_count * 2U / 3U;
  const size_t first_count = static_cast<size_t>(std::count_if(
      selected.begin(), selected.end(), [split](const size_t index) {return index < split;}));
  require(first_count == 25U && selected.size() - first_count == 25U,
    "distance decay did not sample 25 cameras from each segment");

  auto sorted = selected;
  std::sort(sorted.begin(), sorted.end());
  require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
    "optimization sampling returned a duplicate camera");

  std::mt19937 same_rng(42U);
  const auto repeated = gaussian_lic_mapping::select_upstream_random_optimization_indices(
    positions, 100U, true, same_rng);
  require(selected == repeated, "fixed-seed optimization selection is not deterministic");
}

}  // namespace

int main()
{
  try {
    // 80 cameras exercises the upstream ordering: decay must override the
    // ordinary "train_camera_num <= max_iters" all-camera selection.
    check_decayed_case(80U);
    check_decayed_case(150U);

    std::mt19937 short_rng(7U);
    const auto short_selected = gaussian_lic_mapping::select_upstream_random_optimization_indices(
      make_positions(80U, 120.0), 100U, true, short_rng);
    require(short_selected.size() == 80U, "distance threshold must be strictly greater than 120 m");

    std::mt19937 long_rng(7U);
    const auto long_selected = gaussian_lic_mapping::select_upstream_random_optimization_indices(
      make_positions(150U, 120.0), 100U, true, long_rng);
    require(long_selected.size() == 100U, "ordinary upstream max_iters selection changed");

    std::cout << "optimization_sampling_probe: PASS\n";
    return 0;
  } catch (const std::exception & ex) {
    std::cerr << "optimization_sampling_probe: FAIL: " << ex.what() << "\n";
    return 1;
  }
}
