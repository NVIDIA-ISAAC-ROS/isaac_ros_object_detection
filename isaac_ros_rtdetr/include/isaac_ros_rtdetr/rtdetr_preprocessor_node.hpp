// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{

class RtDetrPreprocessorNode : public rclcpp::Node
{
public:
  explicit RtDetrPreprocessorNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~RtDetrPreprocessorNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

  // Subscription to input NitrosTensorList messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  // Publisher for output NitrosTensorList messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosTensorList>> nitros_pub_;

  std::string input_image_tensor_name_{};
  std::string output_image_tensor_name_{};
  std::string output_size_tensor_name_{};
  int64_t image_height_{};
  int64_t image_width_{};
  cudaStream_t stream_;
};

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_RTDETR__RTDETR_PREPROCESSOR_NODE_HPP_
