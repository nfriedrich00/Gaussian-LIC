// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_mapping/optimization_sampling.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>

namespace gaussian_lic_mapping
{

bool should_decay_optimization_list(
  const std::vector<Eigen::Vector3d> & camera_positions)
{
  if (camera_positions.size() < 2U) {
    return false;
  }
  return (camera_positions.back() - camera_positions.front()).norm() > 120.0;
}

std::vector<size_t> select_upstream_random_optimization_indices(
  const std::vector<Eigen::Vector3d> & camera_positions,
  const size_t max_sample_count,
  const bool iteration_decay,
  std::mt19937 & rng)
{
  if (camera_positions.empty() || max_sample_count == 0U) {
    return {};
  }

  std::vector<size_t> all_indices(camera_positions.size());
  std::iota(all_indices.begin(), all_indices.end(), 0U);

  std::vector<size_t> selected;
  selected.reserve(std::min(max_sample_count, camera_positions.size()));

  // Upstream first creates the ordinary list, then decayOptList clears and
  // replaces it.  Test the decay condition before the <= max fast path so an
  // 80-camera sequence still becomes 25+25 when max_sample_count is 100.
  if (iteration_decay && should_decay_optimization_list(camera_positions)) {
    const size_t decayed_budget = max_sample_count / 2U;
    const size_t per_segment_budget = decayed_budget / 2U;
    const size_t split = camera_positions.size() * 2U / 3U;
    const size_t first_count = std::min(per_segment_budget, split);
    const size_t second_count = std::min(
      per_segment_budget, camera_positions.size() - split);
    std::sample(
      all_indices.begin(), all_indices.begin() + static_cast<std::ptrdiff_t>(split),
      std::back_inserter(selected), first_count, rng);
    std::sample(
      all_indices.begin() + static_cast<std::ptrdiff_t>(split), all_indices.end(),
      std::back_inserter(selected), second_count, rng);
  } else if (camera_positions.size() <= max_sample_count) {
    selected = all_indices;
  } else {
    std::sample(
      all_indices.begin(), all_indices.end(), std::back_inserter(selected),
      max_sample_count, rng);
  }

  std::shuffle(selected.begin(), selected.end(), rng);
  return selected;
}

}  // namespace gaussian_lic_mapping
