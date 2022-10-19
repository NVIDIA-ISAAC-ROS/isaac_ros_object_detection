# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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

execute_process(COMMAND uname -m COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCHITECTURE
)
message( STATUS "Architecture: ${ARCHITECTURE}" )

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Decoder node
ament_auto_add_library(detectnet_decoder_node SHARED src/detectnet_decoder_node.cpp)
target_compile_definitions(detectnet_decoder_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)

rclcpp_components_register_nodes(detectnet_decoder_node "nvidia::isaac_ros::detectnet::DetectNetDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::detectnet::DetectNetDecoderNode;$<TARGET_FILE:detectnet_decoder_node>\n")

install(TARGETS detectnet_decoder_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_detectnet_pol_test.py TIMEOUT "400")
endif()

# Visualizer python scripts

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/isaac_ros_detectnet_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  resources/rosbags/detectnet_rosbag
  DESTINATION share/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch config)

find_package(vision_msgs REQUIRED)
