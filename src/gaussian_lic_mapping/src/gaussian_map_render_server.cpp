// SPDX-License-Identifier: GPL-3.0-or-later
//
// In-loop render server: load a 3DGS map (point_cloud.ply) once onto CUDA and
// serve synchronous render requests over a Unix domain socket. Companion to
// gaussian_map_renderer (same PLY parsing / camera composition), but instead of
// a fixed TUM, each request carries a body pose IN THE MAP WORLD FRAME so the
// continuous-time tracker can render at its CURRENT ESTIMATE (paper
// Gaussian-LIC2 Camera Factor Option 2 shape). The CT package stays torch-free:
// it only speaks this tiny binary protocol.
//
// Usage:
//   gaussian_map_render_server <point_cloud.ply> <socket_path>
//                              [fx fy cx cy qx qy qz qw tx ty tz]
//
// Protocol (little-endian, per request):
//   client -> server: 7 x float64  (qx qy qz qw tx ty tz — body pose, map frame)
//   server -> client: uint32 width, uint32 height, width*height x uint8 gray
//                     (BT.601 from the rendered RGB); width=height=0 on failure.

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <gaussian_lic_mapping/backend_config.hpp>
#include <gaussian_lic_mapping/mapper_dataset.hpp>
#include <gaussian_lic_mapping/torch_backend.hpp>

namespace
{

// Defaults: CBD / fastlivo2 calibration (identical to gaussian_map_renderer).
constexpr double kFx = 646.78472;
constexpr double kFy = 646.65775;
constexpr double kCx = 313.456795;
constexpr double kCy = 261.399612;
constexpr int kWidth = 640;
constexpr int kHeight = 512;
constexpr double kCamImuQx = -0.4991948721;
constexpr double kCamImuQy = 0.5038197882;
constexpr double kCamImuQz = -0.4930665852;
constexpr double kCamImuQw = 0.5038406923;
constexpr double kCamImuTx = 0.0673699;
constexpr double kCamImuTy = 0.0412418;
constexpr double kCamImuTz = 0.0764217;

struct PlyColumns
{
  int64_t n{0};
  std::vector<float> xyz;
  std::vector<float> dc;
  std::vector<float> rest;
  std::vector<float> opacity;
  std::vector<float> scaling;
  std::vector<float> rotation;
};

// Binary-little-endian 59-col Gaussian PLY parser (same layout contract as
// gaussian_map_renderer; ASCII fallback intentionally omitted here).
bool parse_ply(const std::string & path, PlyColumns & out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "failed to open PLY " << path << "\n";
    return false;
  }
  std::string line;
  int64_t vertex_count = -1;
  bool binary_little_endian = false;
  while (std::getline(in, line)) {
    if (line == "format binary_little_endian 1.0") {
      binary_little_endian = true;
    }
    if (line.rfind("element vertex", 0) == 0) {
      vertex_count = std::stoll(line.substr(std::string("element vertex").size()));
    }
    if (line.rfind("end_header", 0) == 0) {
      break;
    }
  }
  if (vertex_count <= 0 || !binary_little_endian) {
    std::cerr << "PLY must be binary_little_endian with vertex count\n";
    return false;
  }
  out.n = vertex_count;
  out.xyz.resize(static_cast<size_t>(vertex_count) * 3U);
  out.dc.resize(static_cast<size_t>(vertex_count) * 3U);
  out.rest.resize(static_cast<size_t>(vertex_count) * 45U);
  out.opacity.resize(static_cast<size_t>(vertex_count));
  out.scaling.resize(static_cast<size_t>(vertex_count) * 3U);
  out.rotation.resize(static_cast<size_t>(vertex_count) * 4U);
  constexpr int kCols = 59;
  float row[kCols];
  for (int64_t r = 0; r < vertex_count; ++r) {
    in.read(reinterpret_cast<char *>(row), sizeof(row));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(row))) {
      std::cerr << "binary PLY truncated at row " << r << "\n";
      return false;
    }
    const size_t i = static_cast<size_t>(r);
    std::copy(row, row + 3, &out.xyz[i * 3]);
    std::copy(row + 3, row + 6, &out.dc[i * 3]);
    std::copy(row + 6, row + 51, &out.rest[i * 45]);
    out.opacity[i] = row[51];
    std::copy(row + 52, row + 55, &out.scaling[i * 3]);
    std::copy(row + 55, row + 59, &out.rotation[i * 4]);
  }
  return true;
}

gaussian_lic_mapping::TorchGaussianMap build_map(const PlyColumns & c, torch::Device device)
{
  const auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
  const int64_t n = c.n;
  auto xyz = torch::from_blob(const_cast<float *>(c.xyz.data()), {n, 3}, cpu).clone();
  auto dc = torch::from_blob(const_cast<float *>(c.dc.data()), {n, 3}, cpu)
              .clone().reshape({n, 3, 1}).transpose(1, 2).contiguous();
  auto rest = torch::from_blob(const_cast<float *>(c.rest.data()), {n, 45}, cpu)
                .clone().reshape({n, 3, 15}).transpose(1, 2).contiguous();
  auto opacity = torch::from_blob(const_cast<float *>(c.opacity.data()), {n, 1}, cpu).clone();
  auto scaling = torch::from_blob(const_cast<float *>(c.scaling.data()), {n, 3}, cpu).clone();
  auto rotation = torch::from_blob(const_cast<float *>(c.rotation.data()), {n, 4}, cpu).clone();
  gaussian_lic_mapping::TorchGaussianMap map;
  map.xyz = xyz.to(device).contiguous();
  map.features_dc = dc.to(device).contiguous();
  map.features_rest = rest.to(device).contiguous();
  map.opacity = opacity.to(device).contiguous();
  map.scaling = scaling.to(device).contiguous();
  map.rotation = rotation.to(device).contiguous();
  map.sh_degree = 3;
  map.foreground_count = static_cast<size_t>(n);
  map.skybox_count = 0;
  return map;
}

bool read_full(int fd, void * buf, size_t len)
{
  auto * p = static_cast<uint8_t *>(buf);
  while (len > 0) {
    const ssize_t r = ::read(fd, p, len);
    if (r <= 0) {
      return false;
    }
    p += r;
    len -= static_cast<size_t>(r);
  }
  return true;
}

bool write_full(int fd, const void * buf, size_t len)
{
  const auto * p = static_cast<const uint8_t *>(buf);
  while (len > 0) {
    const ssize_t w = ::write(fd, p, len);
    if (w <= 0) {
      return false;
    }
    p += w;
    len -= static_cast<size_t>(w);
  }
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 3 && argc != 14) {
    std::cerr << "usage: gaussian_map_render_server <ply> <socket_path>"
                 " [fx fy cx cy qx qy qz qw tx ty tz]\n";
    return 2;
  }
  const std::string ply_path = argv[1];
  const std::string socket_path = argv[2];
  double fx = kFx, fy = kFy, cx = kCx, cy = kCy;
  double eqx = kCamImuQx, eqy = kCamImuQy, eqz = kCamImuQz, eqw = kCamImuQw;
  double etx = kCamImuTx, ety = kCamImuTy, etz = kCamImuTz;
  if (argc == 14) {
    fx = std::atof(argv[3]);
    fy = std::atof(argv[4]);
    cx = std::atof(argv[5]);
    cy = std::atof(argv[6]);
    eqx = std::atof(argv[7]);
    eqy = std::atof(argv[8]);
    eqz = std::atof(argv[9]);
    eqw = std::atof(argv[10]);
    etx = std::atof(argv[11]);
    ety = std::atof(argv[12]);
    etz = std::atof(argv[13]);
  }
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available\n";
    return 2;
  }
  const auto device = torch::Device(torch::kCUDA);

  PlyColumns cols;
  if (!parse_ply(ply_path, cols)) {
    return 1;
  }
  auto map = build_map(cols, device);
  PlyColumns().xyz.swap(cols.xyz);
  PlyColumns().rest.swap(cols.rest);
  std::cout << "[render-server] map N=" << map.foreground_count << " on CUDA" << std::endl;

  const Eigen::Quaterniond q_cam_to_imu =
    Eigen::Quaterniond(eqw, eqx, eqy, eqz).normalized();
  const Eigen::Vector3d p_cam_to_imu(etx, ety, etz);

  gaussian_lic_mapping::GaussianBackendConfig config;
  config.white_background = false;
  const cv::Mat dummy_img(kHeight, kWidth, CV_32FC3, cv::Scalar(0, 0, 0));
  const cv::Mat dummy_depth(kHeight, kWidth, CV_32FC1, cv::Scalar(0));

  ::unlink(socket_path.c_str());
  const int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "socket() failed\n";
    return 1;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
    ::listen(listen_fd, 1) != 0)
  {
    std::cerr << "bind/listen failed on " << socket_path << "\n";
    return 1;
  }
  std::cout << "[render-server] listening on " << socket_path << std::endl;

  torch::NoGradGuard no_grad;
  uint64_t served = 0;
  while (true) {
    const int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
      continue;
    }
    std::cout << "[render-server] client connected" << std::endl;
    double req[7];
    while (read_full(fd, req, sizeof(req))) {
      const Eigen::Quaterniond q_body =
        Eigen::Quaterniond(req[3], req[0], req[1], req[2]).normalized();
      const Eigen::Vector3d t_body(req[4], req[5], req[6]);

      gaussian_lic_mapping::CameraFrameRecord frame;
      frame.frame_index = served;
      frame.is_keyframe = false;
      frame.image_name = std::to_string(served);
      frame.width = kWidth;
      frame.height = kHeight;
      frame.image_rgb_float = dummy_img;
      frame.depth_m_float = dummy_depth;
      frame.r_wc = (q_body * q_cam_to_imu).normalized().toRotationMatrix();
      frame.t_wc = t_body + q_body * p_cam_to_imu;

      uint32_t header[2] = {0, 0};
      std::vector<uint8_t> gray;
      try {
        const auto camera = gaussian_lic_mapping::make_torch_camera(
          frame, fx, fy, cx, cy, device, device);
        const auto result =
          gaussian_lic_mapping::render_gaussian_map_from_camera(map, camera, config, device);
        auto img = result.rendered_image.detach().clamp(0.0F, 1.0F);  // [3,H,W] RGB
        auto gray_t =
          (0.299F * img[0] + 0.587F * img[1] + 0.114F * img[2]) * 255.0F;
        auto u8 = gray_t.round().clamp(0.0F, 255.0F)
          .to(torch::kU8).to(torch::kCPU).contiguous();
        gray.assign(
          u8.data_ptr<uint8_t>(),
          u8.data_ptr<uint8_t>() + static_cast<size_t>(kWidth) * kHeight);
        header[0] = static_cast<uint32_t>(kWidth);
        header[1] = static_cast<uint32_t>(kHeight);
      } catch (const std::exception & ex) {
        std::cerr << "[render-server] render failed: " << ex.what() << std::endl;
      }
      if (!write_full(fd, header, sizeof(header))) {
        break;
      }
      if (!gray.empty() && !write_full(fd, gray.data(), gray.size())) {
        break;
      }
      ++served;
      if (served % 200 == 0) {
        std::cout << "[render-server] served " << served << " renders" << std::endl;
      }
    }
    ::close(fd);
    std::cout << "[render-server] client disconnected (served=" << served << ")" << std::endl;
  }
  return 0;
}
