# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_detectnet LANGUAGES C CXX)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    std
)
find_package(isaac_ros_nitros_detection2_d_array_type REQUIRED)
find_package(yaml-cpp)

# DetectNet extension
add_library(gxf_detectnet SHARED
  detectnet.cpp
  detectnet_decoder.cpp
  deepstream_utils/nvdsinferutils/dbscan/nvdsinfer_dbscan.cpp
  deepstream_utils/nvdsinferutils/dbscan/nvdsinfer_dbscan.hpp
  deepstream_utils/nvdsinferutils/dbscan/EigenDefs.hpp
)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
target_include_directories(gxf_detectnet PRIVATE
    ${isaac_ros_nitros_detection2_d_array_type_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/detectnet
    ${CMAKE_CURRENT_SOURCE_DIR}/deepstream_utils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deepstream_utils/nvdsinfer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deepstream_utils/nvdsinferutils/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deepstream_utils/nvdsinferutils/dbscan
)
target_link_libraries(gxf_detectnet
  PRIVATE
    Eigen3::Eigen
    ${isaac_ros_nitros_detection2_d_array_type_LIBRARIES}
  PUBLIC
    GXF::std
    yaml-cpp
)

