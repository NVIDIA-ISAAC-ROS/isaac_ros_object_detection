# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_detectnet LANGUAGES C CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_auto_find_build_dependencies()

# NVDSINFER_DBSCAN

execute_process(COMMAND uname -m COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCHITECTURE
)
message( STATUS "Architecture: ${ARCHITECTURE}" )

add_library(nvdsinfer_dbscan SHARED IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET nvdsinfer_dbscan PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/dbscan/x86_64/libnvds_dbscan.so)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET nvdsinfer_dbscan PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/dbscan/jetpack50/libnvds_dbscan.so)
endif()


# Decoder node
ament_auto_add_library(detectnet_decoder_node SHARED src/detectnet_decoder_node.cpp)
target_compile_definitions(detectnet_decoder_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(detectnet_decoder_node nvdsinfer_dbscan)

rclcpp_components_register_nodes(detectnet_decoder_node "nvidia::isaac_ros::detectnet::DetectNetDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::detectnet::DetectNetDecoderNode;$<TARGET_FILE:detectnet_decoder_node>\n")

install(TARGETS detectnet_decoder_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_detectnet_pol_test.py TIMEOUT "300")
endif()

# Visualizer python scripts

ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/isaac_ros_detectnet_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  resources/rosbags/detectnet_rosbag
  DESTINATION share/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch)

find_package(vision_msgs REQUIRED)
