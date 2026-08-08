// SPDX-License-Identifier: GPL-3.0-or-later

// ROS2_PORT_NOTE [UPSTREAM_LIC2][COMPAT]: SPNet itself was already added by
// ROS1 LIC2.  Jazzy rewrites only its TensorRT host runtime for TRT 8/10 named
// I/O, dynamic shapes, an owned CUDA stream/pinned buffers and RAII; acceptance
// and metric-depth post-processing remain the LIC2 contract.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <NvInfer.h>

namespace gaussian_lic_mapping
{

class DepthCompleter
{
public:
  DepthCompleter(const std::string & engine_path, int input_width, int input_height);
  ~DepthCompleter();

  DepthCompleter(const DepthCompleter &) = delete;
  DepthCompleter & operator=(const DepthCompleter &) = delete;
  DepthCompleter(DepthCompleter &&) = delete;
  DepthCompleter & operator=(DepthCompleter &&) = delete;

  cv::Mat complete(const cv::Mat & rgb_image, const cv::Mat & sparse_depth_m);

private:
  struct InferDeleter
  {
    template<typename T>
    void operator()(T * obj) const
    {
      delete obj;
    }
  };

  class Logger final : public nvinfer1::ILogger
  {
  public:
    void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override;
  };

  void init_engine(const std::string & engine_path);
  void release_resources() noexcept;
  std::vector<char> read_file(const std::string & filename) const;
  void allocate_buffers();
  void assign_tensor_index(const std::string & name, int index, bool is_input);
  bool run_inference();
  size_t volume(const nvinfer1::Dims & dims) const;
  void prepare_inputs(const cv::Mat & rgb_image, const cv::Mat & sparse_depth_m);
  cv::Mat process_output();

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;
  cudaStream_t stream_{nullptr};
  std::vector<void *> device_buffers_;
  std::vector<float *> host_buffers_;
  std::vector<size_t> buffer_element_counts_;
  std::vector<std::string> tensor_names_;
  std::vector<int> input_indices_;
  int rgb_index_{0};
  int depth_index_{1};
  int mask_index_{2};
  int output_index_{3};
  int input_width_{0};
  int input_height_{0};
};

}  // namespace gaussian_lic_mapping
