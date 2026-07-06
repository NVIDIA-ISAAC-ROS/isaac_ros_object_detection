// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_
#define ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{

class RtDetrPreprocessorNode : public rclcpp::Node
{
public:
  explicit RtDetrPreprocessorNode(const rclcpp::NodeOptions & options);

  ~RtDetrPreprocessorNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorList & msg);

  // QOS settings
  const int16_t input_queue_size_;
  const int16_t output_queue_size_;
  std::string input_image_tensor_name_{};
  std::string output_image_tensor_name_{};
  std::string output_size_tensor_name_{};
  int64_t image_height_{};
  int64_t image_width_{};
  bool use_max_dim_for_orig_size_{};
  int64_t memory_pool_block_size_{1920 * 1200 * 4};
  int64_t memory_pool_num_blocks_{40};

  // Subscriber and publisher for input and output NitrosTensorList messages
  rclcpp::Subscription<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr nitros_sub_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosTensorList>::SharedPtr nitros_pub_;

  // CUDA Resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;
};

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_
