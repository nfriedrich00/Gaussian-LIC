cmake_minimum_required(VERSION 3.16)
project(cocolic)

# Unofficial ROS2 Jazzy port of upstream Coco-LIC. The package builds the
# continuous-time spline core, LiDAR/IMU/camera factors, rosbag2 ingestion, and
# the ROS2 odometry node.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -Wno-sign-compare -Wno-unused -Wno-comment -pthread)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(gaussian_lic_msgs REQUIRED)  # Mapper-feedback: RenderedFeedback subscription
find_package(pcl_conversions REQUIRED)  # ROS2 port: pcl::fromROSMsg for Velodyne (M2DGR)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
# ROS2 port: only common + kdtree are used (transformPointCloud, KdTreeFLANN).
# Dropped 'io' — it pulls system GDAL/HDF5 whose libcurl clashes with conda's,
# and pcl::io is used only by the (not-built) camera module.
find_package(PCL 1.10 REQUIRED COMPONENTS common kdtree)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(livox_ros_driver2 QUIET)

# glog (upstream uses <glog/logging.h>)
find_library(GLOG_LIB glog REQUIRED)
# lz4 — transitive PCL/FLANN dependency, must be explicit at link time.
find_library(LZ4_LIB NAMES lz4 liblz4 REQUIRED)
# System libcurl (CURL_OPENSSL_4): if a transitive PCL dependency still drags
# in GDAL/HDF5, a conda libcurl on the link path may lack the versioned system
# symbols. Prefer the platform multiarch system path, but keep this optional on
# systems where curl is not part of the dependency closure.
find_library(SYS_CURL_LIB NAMES curl
  PATHS
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /usr/lib
    /lib
  NO_DEFAULT_PATH)
set(COCOLIC_SYSTEM_CURL_LINK)
if(SYS_CURL_LIB)
  list(APPEND COCOLIC_SYSTEM_CURL_LINK ${SYS_CURL_LIB})
endif()

# Upstream resolves includes relative to src/ (e.g. sophus_lib/so3.hpp,
# ../utils/x from spline/). Mirror that with src/ on the include path.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include   # include/ros compat shim
  # S4 camera module include paths (R3LIVE visual front-end).
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/tools
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/rgb_map
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/optical_flow
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/loam/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/loam/include/kd_tree
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/loam/include/FOV_Checker
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Core library: the ROS-agnostic continuous-time spline + params (verbatim).
add_library(cocolic_core STATIC
  src/spline/trajectory.cpp
  src/utils/parameter_struct.cpp
  src/odom/trajectory_estimator.cpp
  src/odom/factor/analytic_diff/marginalization_factor.cpp
  src/odom/trajectory_manager.cpp
  src/lidar/livox_feature_extraction.cpp
  src/lidar/velodyne_feature_extraction.cpp
  src/lidar/lidar_handler.cpp
  src/imu/imu_initializer.cpp
  src/imu/imu_state_estimator.cpp
  src/odom/msg_manager.cpp
  src/odom/odometry_manager.cpp
)
target_link_libraries(cocolic_core
  ${CERES_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
  ${GLOG_LIB}
)
# msg_manager.cpp uses rosbag2 + rclcpp serialization + cv_bridge + sensor_msgs.
ament_target_dependencies(cocolic_core
  rclcpp rosbag2_cpp rosbag2_storage cv_bridge pcl_conversions sensor_msgs std_msgs
  geometry_msgs nav_msgs tf2_ros gaussian_lic_msgs)
if(livox_ros_driver2_FOUND)
  target_compile_definitions(cocolic_core PUBLIC COCOLIC_HAS_LIVOX_ROS_DRIVER2=1)
  ament_target_dependencies(cocolic_core livox_ros_driver2)
  message(STATUS "Coco-LIC: native livox_ros_driver2/CustomMsg support enabled")
else()
  message(STATUS "Coco-LIC: livox_ros_driver2 not found; PointCloud2 Livox input remains available")
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(cocolic_core OpenMP::OpenMP_CXX)
endif()

# Active R3LIVE visual front-end modules. They are linked into odometry_node via
# cocolic_camera; this is a runtime component, not a compile-only staging target.
add_library(cocolic_camera STATIC
  src/camera/rgb_map/image_frame.cpp
  src/camera/optical_flow/lkpyramid.cpp
  src/camera/loam/include/kd_tree/ikd_Tree.cpp
  src/camera/loam/include/FOV_Checker/FOV_Checker.cpp
  src/camera/rgb_map/pointcloud_rgbd.cpp
  src/camera/rgb_map/rgbmap_tracker.cpp
  src/camera/r3live_slim.cpp
)
target_link_libraries(cocolic_camera
  ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIB})
if(OpenMP_CXX_FOUND)
  target_link_libraries(cocolic_camera OpenMP::OpenMP_CXX)
endif()

# Custom messages (ROS2 rosidl port of upstream msg/*.msg).
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/FeatureCloud.msg
  msg/ImuArray.msg
  msg/PoseArray.msg
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)
add_dependencies(cocolic_core ${PROJECT_NAME}__rosidl_generator_cpp)
rosidl_get_typesupport_target(cocolic_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(cocolic_core ${cocolic_typesupport_target})

# Main odometry node (ROS2 port of upstream odometry_node).
add_executable(odometry_node src/odometry_node.cpp)
# cocolic_core (odometry_manager) now calls the slimmed R3LIVE in cocolic_camera.
# --start-group/--end-group resolves any cross-lib symbol order between them.
target_link_libraries(odometry_node
  -Wl,--start-group cocolic_core cocolic_camera -Wl,--end-group
  ${GLOG_LIB} ${LZ4_LIB} ${COCOLIC_SYSTEM_CURL_LINK})
ament_target_dependencies(odometry_node rclcpp ament_index_cpp rosbag2_cpp
  rosbag2_storage sensor_msgs nav_msgs tf2_ros)
install(TARGETS odometry_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})
install(FILES README_ROS2.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  add_executable(bag_ingest_test test/bag_ingest_test.cpp)
  ament_target_dependencies(bag_ingest_test rclcpp rosbag2_cpp sensor_msgs)

  add_executable(ingest_offset_time_test test/ingest_offset_time_test.cpp)
  ament_target_dependencies(ingest_offset_time_test rclcpp rosbag2_cpp
    rosbag2_storage sensor_msgs)

  add_executable(storage_autodetect_test test/storage_autodetect_test.cpp)
  ament_target_dependencies(storage_autodetect_test rclcpp rosbag2_cpp rosbag2_storage)

  add_test(NAME cocolic_bag_ingest_test COMMAND $<TARGET_FILE:bag_ingest_test>)
  add_test(NAME cocolic_ingest_offset_time_test COMMAND $<TARGET_FILE:ingest_offset_time_test>)
  add_test(NAME cocolic_storage_autodetect_test COMMAND $<TARGET_FILE:storage_autodetect_test>)
  set_tests_properties(
    cocolic_bag_ingest_test
    cocolic_ingest_offset_time_test
    cocolic_storage_autodetect_test
    PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(input_contract_test test/input_contract_test.cpp)
  target_link_libraries(input_contract_test cocolic_core ${OpenCV_LIBS}
    ${LZ4_LIB} ${COCOLIC_SYSTEM_CURL_LINK})
  ament_target_dependencies(input_contract_test rclcpp sensor_msgs cv_bridge)
  add_test(NAME cocolic_input_contract_test COMMAND $<TARGET_FILE:input_contract_test>)

  add_executable(safety_contract_test test/safety_contract_test.cpp)
  target_link_libraries(safety_contract_test
    -Wl,--start-group cocolic_core cocolic_camera -Wl,--end-group
    ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIB}
    ${LZ4_LIB} ${COCOLIC_SYSTEM_CURL_LINK})
  ament_target_dependencies(safety_contract_test rclcpp sensor_msgs cv_bridge)
  add_test(NAME cocolic_safety_contract_test COMMAND $<TARGET_FILE:safety_contract_test>)

  add_executable(global_map_lifecycle_test test/global_map_lifecycle_test.cpp)
  target_link_libraries(global_map_lifecycle_test cocolic_camera
    ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIB}
    ${COCOLIC_SYSTEM_CURL_LINK})
  add_test(NAME cocolic_global_map_lifecycle_test
    COMMAND $<TARGET_FILE:global_map_lifecycle_test>)

  add_executable(profile_config_test test/profile_config_test.cpp)
  target_link_libraries(profile_config_test yaml-cpp)
  target_compile_definitions(profile_config_test PRIVATE
    COCOLIC_SOURCE_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config")
  add_test(NAME cocolic_profile_config_test COMMAND $<TARGET_FILE:profile_config_test>)

  add_executable(numeric_guard_test test/numeric_guard_test.cpp)
  target_link_libraries(numeric_guard_test yaml-cpp)
  add_test(NAME cocolic_numeric_guard_test COMMAND $<TARGET_FILE:numeric_guard_test>)

  add_executable(odometry_viewer_contract_test test/odometry_viewer_contract_test.cpp)
  target_link_libraries(odometry_viewer_contract_test
    cocolic_core ${PCL_LIBRARIES} ${GLOG_LIB} ${COCOLIC_SYSTEM_CURL_LINK})
  ament_target_dependencies(odometry_viewer_contract_test
    rclcpp sensor_msgs nav_msgs tf2_ros pcl_conversions)
  add_test(NAME cocolic_odometry_viewer_contract_test
    COMMAND $<TARGET_FILE:odometry_viewer_contract_test>)
endif()

ament_package()
