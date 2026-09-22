// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_RTDETR__RTDETR_DECODER_UTILS_HPP_
#define ISAAC_ROS_RTDETR__RTDETR_DECODER_UTILS_HPP_

#include <cstddef>

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{

constexpr std::size_t kBoundingBoxElementCount = 4;

constexpr bool AreOutputTensorSizesValid(
  const std::size_t labels_size, const std::size_t boxes_size,
  const std::size_t scores_size) noexcept
{
  return labels_size == scores_size &&
         boxes_size % kBoundingBoxElementCount == 0 &&
         boxes_size / kBoundingBoxElementCount == scores_size;
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_RTDETR__RTDETR_DECODER_UTILS_HPP_
