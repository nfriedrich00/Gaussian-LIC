// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_mapping/depth_completer.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

int main(int argc, char ** argv)
{
  if (argc != 4) {
    std::cerr << "usage: depth_completer_probe ENGINE WIDTH HEIGHT\n";
    return 2;
  }
  try {
    const std::string engine_path(argv[1]);
    const int width = std::stoi(argv[2]);
    const int height = std::stoi(argv[3]);
    cv::Mat rgb(height, width, CV_32FC3, cv::Scalar(0.25F, 0.5F, 0.75F));
    cv::Mat sparse = cv::Mat::zeros(height, width, CV_32FC1);
    for (int v = 8; v < height; v += 32) {
      for (int u = 8; u < width; u += 32) {
        sparse.at<float>(v, u) = 5.0F + static_cast<float>(v) / static_cast<float>(height);
      }
    }

    gaussian_lic_mapping::DepthCompleter completer(engine_path, width, height);
    const cv::Mat completed = completer.complete(rgb, sparse);
    if (completed.type() != CV_32FC1 || completed.rows != height || completed.cols != width) {
      throw std::runtime_error("TensorRT output shape/type does not match metric CV_32FC1 contract");
    }
    if (!cv::checkRange(completed, true, nullptr, -1.0e6, 1.0e6)) {
      throw std::runtime_error("TensorRT output contains NaN/Inf or implausible magnitude");
    }
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(completed, &minimum, &maximum);
    const cv::Mat known_mask = sparse > 0.0F;
    cv::Mat completed_known;
    completed.copyTo(completed_known, known_mask);
    const double mean_known_difference = cv::mean(completed_known - sparse, known_mask)[0];
    std::cout << "engine=" << engine_path << "\n";
    std::cout << "output=" << completed.cols << "x" << completed.rows << " CV_32FC1\n";
    std::cout << "min_m=" << minimum << "\n";
    std::cout << "max_m=" << maximum << "\n";
    std::cout << "mean_known_difference_m=" << mean_known_difference << "\n";
    return 0;
  } catch (const std::exception & ex) {
    std::cerr << "depth_completer_probe: FAIL: " << ex.what() << "\n";
    return 1;
  }
}
