#include <camera/rgb_map/pointcloud_rgbd.hpp>
#include <camera/r3live_slim.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

int main()
{
  static_assert(cocolic::R3LIVE::kStartProjectionService == 0,
                "slim offline replay must not start the wall-clock projection service");
  {
    Global_map slim_map(cocolic::R3LIVE::kStartProjectionService);
    if (slim_map.m_thread_service)
      throw std::runtime_error("slim projection service was unexpectedly started");
  }

  {
    Global_map map(0);
    map.m_minimum_depth_for_projection = 0.0;
    map.m_maximum_depth_for_projection = 100.0;

    for (int index = 0; index < 4; ++index)
    {
      auto point = std::make_shared<RGB_pts>();
      auto voxel = std::make_shared<RGB_Voxel>();
      voxel->add_pt(point);
      map.m_voxels_recent_visited.insert(voxel);
    }

    // Assign descending stable IDs in the set's actual iteration order. The
    // projection result must still be ascending after deterministic sorting.
    int iteration_index = 0;
    for (const auto &voxel : map.m_voxels_recent_visited)
    {
      auto &point = voxel->m_pts_in_grid.back();
      point->m_pt_index = 100 - 10 * iteration_index;
      point->set_pos(vec_3(-0.9 + 0.6 * iteration_index, 0.0, 5.0));
      ++iteration_index;
    }

    Eigen::Matrix3d camera_k = Eigen::Matrix3d::Identity();
    camera_k(0, 0) = 100.0; camera_k(1, 1) = 100.0;
    camera_k(0, 2) = 320.0; camera_k(1, 2) = 240.0;
    auto image = std::make_shared<Image_frame>(camera_k);
    image->m_img_cols = 640;
    image->m_img_rows = 480;
    image->m_fov_margin = 0.0;
    image->set_pose(eigen_q::Identity(), vec_3::Zero());

    Eigen::aligned_vector<Eigen::Vector3d> new_points;
    Eigen::aligned_vector<Eigen::Vector2d> new_pixels;
    std::vector<RGB_pt_ptr> selected;
    map.selection_points_for_projection(false, new_points, new_pixels, image,
                                        &selected, nullptr, 1.0, 1, 0);
    if (selected.size() != 4)
      throw std::runtime_error("projection candidate test lost a valid point");
    for (std::size_t index = 1; index < selected.size(); ++index)
    {
      if (selected[index - 1]->m_pt_index >= selected[index]->m_pt_index)
        throw std::runtime_error("projection candidates are not stably ordered");
    }
  }

  for (int instance = 0; instance < 2; ++instance)
  {
    Global_map map(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto point = std::make_shared<RGB_pts>();
    point->m_pos[0] = 1.0; point->m_pos[1] = 2.0; point->m_pos[2] = 3.0;
    point->m_rgb[0] = 10.0; point->m_rgb[1] = 20.0; point->m_rgb[2] = 30.0;
    point->m_N_rgb = 3;
    map.m_rgb_pts_vec.push_back(point);

    const auto output_dir = std::filesystem::temp_directory_path() /
        ("cocolic_pcd_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) +
         "_" + std::to_string(instance));
    map.save_to_pcd(output_dir.string(), "minimal", 1);
    const auto output_file = output_dir / "minimal.pcd";
    std::ifstream input(output_file, std::ios::binary);
    if (!input) throw std::runtime_error("PCD export did not create its output");
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    const std::string marker = "DATA binary\n";
    const auto data_offset = bytes.find(marker);
    if (data_offset == std::string::npos ||
        bytes.find("FIELDS x y z rgb\n") == std::string::npos ||
        bytes.find("POINTS 1\n") == std::string::npos ||
        bytes.size() != data_offset + marker.size() + 16U)
      throw std::runtime_error("PCD export header or binary payload is invalid");
    std::filesystem::remove_all(output_dir);
  }
  return 0;
}
