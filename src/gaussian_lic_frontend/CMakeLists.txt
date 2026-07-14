cmake_minimum_required(VERSION 3.16)
project(gaussian_lic_frontend)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(lic2_contract_adapter
  src/lic2_contract_adapter_node.cpp
)
target_include_directories(lic2_contract_adapter PRIVATE include)
ament_target_dependencies(lic2_contract_adapter
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  tf2_ros
)

install(TARGETS lic2_contract_adapter
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(PROGRAMS
  scripts/livox_custom_to_pointcloud2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME livox_custom_to_pointcloud2
)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  enable_testing()
  add_test(
    NAME livox_custom_bridge_self_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/livox_custom_to_pointcloud2.py
      --self-test
      --ros-args
      --log-level warn
  )
  add_executable(pointcloud_contract_probe
    test/pointcloud_contract_probe.cpp
  )
  target_include_directories(pointcloud_contract_probe PRIVATE include)
  ament_target_dependencies(pointcloud_contract_probe sensor_msgs)
  add_test(NAME pointcloud_contract_probe COMMAND pointcloud_contract_probe)
  add_test(
    NAME lic2_adapter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test/lic2_adapter_contract.py
      ${CMAKE_CURRENT_SOURCE_DIR}/src/lic2_contract_adapter_node.cpp
  )
endif()

ament_package()
