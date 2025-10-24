# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_grounding_dino LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Ensure CUDA imported targets (e.g., CUDA::nvtx3) are available
find_package(CUDAToolkit REQUIRED)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Install python scripts
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/isaac_ros_grounding_dino_text_tokenizer.py
  scripts/isaac_ros_grounding_dino_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# NVTX
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    link_libraries("CUDA::nvtx3")
endif()

# GroundingDinoPreprocessorNode
ament_auto_add_library(grounding_dino_preprocessor_node SHARED src/grounding_dino_preprocessor_node.cpp)
set_target_properties(grounding_dino_preprocessor_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)
rclcpp_components_register_nodes(grounding_dino_preprocessor_node "nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode;"
  "$<TARGET_FILE:grounding_dino_preprocessor_node>\n")

# GroundingDinoDecoderNode
ament_auto_add_library(grounding_dino_decoder_node SHARED src/grounding_dino_decoder_node.cpp)
target_link_libraries(grounding_dino_decoder_node Eigen3::Eigen)
rclcpp_components_register_nodes(grounding_dino_decoder_node "nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode")
set(node_plugins
  "${node_plugins}nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode;"
  "$<TARGET_FILE:grounding_dino_decoder_node>\n")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_grounding_dino_pol_test.py TIMEOUT "600")
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
