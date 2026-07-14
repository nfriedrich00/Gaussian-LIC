#include <camera/rgb_map/image_frame.hpp>
#include <camera/rgb_map/rgbmap_tracker.hpp>
#include <lidar/livox_feature_extraction.h>
#include <utils/cloud_tool.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
YAML::Node LivoxConfig(const int group_size = 3)
{
  return YAML::Load("Livox:\n"
                    "  n_scan: 1\n"
                    "  blind: 0.0\n"
                    "  inf_bound: 4.0\n"
                    "  group_size: " + std::to_string(group_size) + "\n"
                    "  disA: 0.01\n"
                    "  disB: 0.1\n"
                    "  limit_maxmid: 6.25\n"
                    "  limit_midmin: 6.25\n"
                    "  limit_maxmin: 3.24\n"
                    "  p2l_ratio: 225.0\n"
                    "  jump_up_limit: 170.0\n"
                    "  jump_down_limit: 8.0\n"
                    "  edgea: 2.0\n"
                    "  edgeb: 0.1\n"
                    "  smallp_intersect: 172.5\n"
                    "  smallp_ratio: 1.2\n"
                    "  point_filter_num: 1\n");
}

cocolic::CustomMsgLite::Ptr ValidLivoxMessage()
{
  auto message = std::make_shared<cocolic::CustomMsgLite>();
  message->timebase = 1000000000LL;
  for (uint32_t index = 0; index < 12; ++index)
  {
    cocolic::CustomPointLite point;
    point.x = 2.0F + 0.1F * static_cast<float>(index);
    point.y = 0.02F * static_cast<float>(index);
    point.z = 0.01F * static_cast<float>(index);
    point.reflectivity = static_cast<uint8_t>(10 + index);
    point.tag = 0x10;
    point.line = 0;
    point.offset_time = index * 1000000U;
    message->points.push_back(point);
  }
  message->point_num = static_cast<uint32_t>(message->points.size());
  return message;
}
}  // namespace

int main()
{
  try
  {
    RTPointCloud::Ptr empty(new RTPointCloud);
    if (pcl::GetCloudMaxTime(empty) != 0.0F || pcl::GetCloudMaxTimeNs(empty) != 0)
      throw std::runtime_error("empty cloud max-time contract failed");
    RTPointCloud::Ptr unsorted(new RTPointCloud);
    RTPoint first{};
    first.time = 9;
    RTPoint second{};
    second.time = 2;
    unsorted->push_back(first);
    unsorted->push_back(second);
    if (pcl::GetCloudMaxTime(unsorted) != 9.0F ||
        pcl::GetCloudMaxTimeNs(unsorted) != 9)
      throw std::runtime_error("unsorted cloud max-time lost point zero");

    RTPointCloudTmp::Ptr empty_tmp(new RTPointCloudTmp);
    RTPointCloud::Ptr converted(new RTPointCloud);
    pcl::RTPointCloudTmp2RTPointCloud(empty_tmp, converted);
    if (!converted->empty())
      throw std::runtime_error("empty rotating-LiDAR conversion produced points");
    cocolic::VoxelFilter<RTPoint> voxel_filter;
    voxel_filter.SetResolution(0.2F);
    voxel_filter.SetInputCloud(empty);
    voxel_filter.Filter(converted);
    if (!converted->empty())
      throw std::runtime_error("empty voxel-filter input produced points");

    RTPointCloud::Ptr relative(new RTPointCloud);
    RTPoint timed{};
    timed.x = 3.0F;
    timed.ring = 9;
    timed.time = 200;
    relative->push_back(timed);
    pcl::CloudToRelativeMeasureTime(relative, 1000, 900, 100);
    if (relative->front().time != -1 || relative->front().ring != 9 ||
        relative->front().x != 0.0F || relative->front().z != 0.0F)
      throw std::runtime_error("invalid-time sentinel did not preserve ring/zero coordinates");

    bool rejected = false;
    try
    {
      voxel_filter.SetResolution(0.0F);
    }
    catch (const std::invalid_argument &)
    {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("zero voxel resolution was accepted");
    rejected = false;
    try
    {
      cocolic::LivoxFeatureExtraction invalid(LivoxConfig(1));
      (void)invalid;
    }
    catch (const std::invalid_argument &)
    {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("unsafe Livox group_size was accepted");

    cocolic::LivoxFeatureExtraction livox(LivoxConfig());
    const auto message = ValidLivoxMessage();
    RTPointCloud::Ptr parsed(new RTPointCloud);
    if (!livox.ParsePointCloud(message, parsed) || parsed->size() != message->point_num ||
        parsed->front().ring != 0 || parsed->front().time != 0)
      throw std::runtime_error("Livox feature parser lost the first point or metadata");
    if (!livox.ParsePointCloudR3LIVE(message, parsed) ||
        parsed->size() != message->point_num)
      throw std::runtime_error("R3LIVE Livox parser rejected a valid nonempty scan");
    if (!livox.ParsePointCloudNoFeature(message, parsed) ||
        parsed->size() != message->point_num)
      throw std::runtime_error("no-feature Livox parser rejected a valid nonempty scan");

    Image_frame image_frame;
    pcl::PointXYZI point;
    point.x = 0.0F;
    point.y = 0.0F;
    point.z = 2.0F;
    Eigen::Matrix3d camera = Eigen::Matrix3d::Identity();
    double u = 0.0;
    double v = 0.0;
    if (image_frame.project_3d_to_2d(point, camera, u, v, 1.0))
      throw std::runtime_error("projection without pose did not fail closed");
    eigen_q pose = eigen_q::Identity();
    vec_3 translation = vec_3::Zero();
    image_frame.set_pose(pose, translation);
    if (image_frame.project_3d_to_2d(point, camera, u, v, 1.0))
      throw std::runtime_error("projection without intrinsics did not fail closed");
    rejected = false;
    try
    {
      (void)image_frame.get_grey_color(u, v, 1);
    }
    catch (const std::invalid_argument &)
    {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("unsupported grey pyramid layer was accepted");

    Image_frame initialized(camera);
    initialized.set_pose(pose, translation);
    initialized.m_img = cv::Mat(3, 3, CV_8UC3, cv::Scalar(10, 20, 30));
    initialized.init_cubic_interpolation();
    double sample_u = 1.2;
    double sample_v = 1.2;
    cv::Mat expected_gray;
    cv::cvtColor(initialized.m_img, expected_gray, cv::COLOR_BGR2GRAY);
    const double gray = initialized.get_grey_color(sample_u, sample_v, 0);
    if (std::abs(gray - expected_gray.at<uint8_t>(1, 1)) > 1e-6)
      throw std::runtime_error("grey sampling did not use the BGR-derived gray image");

    auto behind_camera = std::make_shared<RGB_pts>();
    behind_camera->set_pos(vec_3(0.0, 0.0, -2.0));
    Rgbmap_tracker tracker;
    tracker.m_map_rgb_pts_in_last_frame_pos[behind_camera.get()] = cv::Point2f(1.0F, 1.0F);
    Global_map empty_map(0);
    auto tracker_frame = std::make_shared<Image_frame>(camera);
    tracker_frame->set_pose(pose, translation);
    tracker_frame->m_img = cv::Mat(20, 20, CV_8UC3, cv::Scalar::all(0));
    tracker_frame->init_cubic_interpolation();
    tracker.update_and_append_track_pts(tracker_frame, empty_map, 10.0);
    if (!tracker.m_map_rgb_pts_in_last_frame_pos.empty())
      throw std::runtime_error("failed projection left a stale RGB-map track");
  }
  catch (const std::exception &error)
  {
    std::cerr << "safety_contract_test: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
