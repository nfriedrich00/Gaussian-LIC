// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include <Eigen/Core>

namespace gaussian_lic_mapping
{

bool should_decay_optimization_list(
  const std::vector<Eigen::Vector3d> & camera_positions);

// Reproduces Gaussian-LIC's optimize()/decayOptList() ordering.  In
// particular, distance decay replaces the initial list even when the number
// of cameras is already below max_sample_count.
std::vector<size_t> select_upstream_random_optimization_indices(
  const std::vector<Eigen::Vector3d> & camera_positions,
  size_t max_sample_count,
  bool iteration_decay,
  std::mt19937 & rng);

}  // namespace gaussian_lic_mapping
