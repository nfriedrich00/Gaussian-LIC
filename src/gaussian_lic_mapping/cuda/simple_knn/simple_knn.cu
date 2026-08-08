// SPDX-License-Identifier: GPL-3.0-or-later

// ROS2_PORT_NOTE [PARITY][SAFETY]: upstream simple-knn CUDA math is namespaced
// and wrapped for the Torch backend.  Reduction extrema and degenerate-axis
// guards below are correctness fixes for planar/constant-coordinate clouds.

#define BOX_SIZE 1024

#include "simple_knn.hpp"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace cg = cooperative_groups;

namespace gaussian_lic_mapping::cuda_ops
{
namespace
{

struct CustomMin
{
  __device__ __forceinline__ float3 operator()(const float3 & lhs, const float3 & rhs) const
  {
    return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
  }
};

struct CustomMax
{
  __device__ __forceinline__ float3 operator()(const float3 & lhs, const float3 & rhs) const
  {
    return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
  }
};

__host__ __device__ uint32_t prep_morton(uint32_t value)
{
  value = (value | (value << 16)) & 0x030000FF;
  value = (value | (value << 8)) & 0x0300F00F;
  value = (value | (value << 4)) & 0x030C30C3;
  value = (value | (value << 2)) & 0x09249249;
  return value;
}

__host__ __device__ uint32_t coord_to_morton(float3 coord, float3 min_point, float3 max_point)
{
  // ROS2_PORT_NOTE [SAFETY]: upstream divided by the raw span; a flat axis
  // produced NaN/Inf Morton coordinates.  The epsilon only affects that
  // degenerate input class.
  const float span_x = fmaxf(max_point.x - min_point.x, 1.0e-9F);
  const float span_y = fmaxf(max_point.y - min_point.y, 1.0e-9F);
  const float span_z = fmaxf(max_point.z - min_point.z, 1.0e-9F);
  const uint32_t max_coord = (1U << 10U) - 1U;
  const auto x = prep_morton(static_cast<uint32_t>(((coord.x - min_point.x) / span_x) * max_coord));
  const auto y = prep_morton(static_cast<uint32_t>(((coord.y - min_point.y) / span_y) * max_coord));
  const auto z = prep_morton(static_cast<uint32_t>(((coord.z - min_point.z) / span_z) * max_coord));
  return x | (y << 1U) | (z << 2U);
}

__global__ void coord_to_morton_kernel(
  int point_count,
  const float3 * points,
  float3 min_point,
  float3 max_point,
  uint32_t * codes)
{
  const auto index = cg::this_grid().thread_rank();
  if (index >= point_count) {
    return;
  }
  codes[index] = coord_to_morton(points[index], min_point, max_point);
}

struct MinMax
{
  float3 min_point;
  float3 max_point;
};

__global__ void box_min_max_kernel(uint32_t point_count, float3 * points, uint32_t * indices, MinMax * boxes)
{
  const auto index = cg::this_grid().thread_rank();

  MinMax current;
  if (index < point_count) {
    current.min_point = points[indices[index]];
    current.max_point = points[indices[index]];
  } else {
    current.min_point = {FLT_MAX, FLT_MAX, FLT_MAX};
    current.max_point = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  }

  __shared__ MinMax reduced[BOX_SIZE];
  for (int offset = BOX_SIZE / 2; offset >= 1; offset /= 2) {
    if (threadIdx.x < 2 * offset) {
      reduced[threadIdx.x] = current;
    }
    __syncthreads();

    if (threadIdx.x < offset) {
      const MinMax other = reduced[threadIdx.x + offset];
      current.min_point.x = min(current.min_point.x, other.min_point.x);
      current.min_point.y = min(current.min_point.y, other.min_point.y);
      current.min_point.z = min(current.min_point.z, other.min_point.z);
      current.max_point.x = max(current.max_point.x, other.max_point.x);
      current.max_point.y = max(current.max_point.y, other.max_point.y);
      current.max_point.z = max(current.max_point.z, other.max_point.z);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    boxes[blockIdx.x] = current;
  }
}

__device__ __host__ float dist_box_point(const MinMax & box, const float3 & point)
{
  float3 diff = {0.0F, 0.0F, 0.0F};
  if (point.x < box.min_point.x || point.x > box.max_point.x) {
    diff.x = min(abs(point.x - box.min_point.x), abs(point.x - box.max_point.x));
  }
  if (point.y < box.min_point.y || point.y > box.max_point.y) {
    diff.y = min(abs(point.y - box.min_point.y), abs(point.y - box.max_point.y));
  }
  if (point.z < box.min_point.z || point.z > box.max_point.z) {
    diff.z = min(abs(point.z - box.min_point.z), abs(point.z - box.max_point.z));
  }
  return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void update_k_best(const float3 & reference, const float3 & point, float * knn)
{
  const float3 diff = {point.x - reference.x, point.y - reference.y, point.z - reference.z};
  float distance = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
  for (int index = 0; index < K; ++index) {
    if (knn[index] > distance) {
      const float previous = knn[index];
      knn[index] = distance;
      distance = previous;
    }
  }
}

__global__ void box_mean_dist_kernel(
  uint32_t point_count,
  float3 * points,
  uint32_t * indices,
  MinMax * boxes,
  float * distances)
{
  const int index = cg::this_grid().thread_rank();
  if (index >= point_count) {
    return;
  }

  const float3 point = points[indices[index]];
  float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

  for (int neighbor = max(0, index - 3); neighbor <= min(static_cast<int>(point_count) - 1, index + 3);
    ++neighbor)
  {
    if (neighbor != index) {
      update_k_best<3>(point, points[indices[neighbor]], best);
    }
  }

  const float reject = best[2];
  best[0] = FLT_MAX;
  best[1] = FLT_MAX;
  best[2] = FLT_MAX;

  const int box_count = (point_count + BOX_SIZE - 1) / BOX_SIZE;
  for (int box_index = 0; box_index < box_count; ++box_index) {
    const MinMax box = boxes[box_index];
    const float box_distance = dist_box_point(box, point);
    if (box_distance > reject || box_distance > best[2]) {
      continue;
    }

    for (int neighbor = box_index * BOX_SIZE; neighbor < min(static_cast<int>(point_count), (box_index + 1) * BOX_SIZE);
      ++neighbor)
    {
      if (neighbor != index) {
        update_k_best<3>(point, points[indices[neighbor]], best);
      }
    }
  }
  distances[indices[index]] = (best[0] + best[1] + best[2]) / 3.0F;
}

}  // namespace

void SimpleKNN::knn(int point_count, float3 * points, float * mean_distances)
{
  float3 * reduction_result = nullptr;
  cudaMalloc(&reduction_result, sizeof(float3));

  size_t temp_storage_bytes = 0;
  const float3 min_init = {FLT_MAX, FLT_MAX, FLT_MAX};
  const float3 max_init = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  float3 min_point;
  float3 max_point;

  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, reduction_result, point_count, CustomMin(), min_init);
  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  cub::DeviceReduce::Reduce(
    temp_storage.data().get(), temp_storage_bytes, points, reduction_result, point_count, CustomMin(), min_init);
  cudaMemcpy(&min_point, reduction_result, sizeof(float3), cudaMemcpyDeviceToHost);

  cub::DeviceReduce::Reduce(
    temp_storage.data().get(), temp_storage_bytes, points, reduction_result, point_count, CustomMax(), max_init);
  cudaMemcpy(&max_point, reduction_result, sizeof(float3), cudaMemcpyDeviceToHost);

  thrust::device_vector<uint32_t> morton(point_count);
  thrust::device_vector<uint32_t> morton_sorted(point_count);
  coord_to_morton_kernel<<<(point_count + 255) / 256, 256>>>(
    point_count, points, min_point, max_point, morton.data().get());

  thrust::device_vector<uint32_t> indices(point_count);
  thrust::sequence(indices.begin(), indices.end());
  thrust::device_vector<uint32_t> indices_sorted(point_count);

  cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(),
    indices_sorted.data().get(), point_count);
  temp_storage.resize(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(
    temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(),
    indices.data().get(), indices_sorted.data().get(), point_count);

  const uint32_t box_count = (point_count + BOX_SIZE - 1) / BOX_SIZE;
  thrust::device_vector<MinMax> boxes(box_count);
  box_min_max_kernel<<<box_count, BOX_SIZE>>>(point_count, points, indices_sorted.data().get(), boxes.data().get());
  box_mean_dist_kernel<<<box_count, BOX_SIZE>>>(
    point_count, points, indices_sorted.data().get(), boxes.data().get(), mean_distances);

  cudaFree(reduction_result);
}

}  // namespace gaussian_lic_mapping::cuda_ops
