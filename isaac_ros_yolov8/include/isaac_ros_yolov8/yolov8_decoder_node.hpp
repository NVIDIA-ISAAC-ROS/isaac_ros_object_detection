// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
#define ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"

#include "std_msgs/msg/string.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{

class YoloV8DecoderNode : public rclcpp::Node
{
public:
  explicit YoloV8DecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~YoloV8DecoderNode();

private:
  void InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg);

  // Subscription to input NitrosTensorList messages
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>> nitros_sub_;

  // Publisher for output Detection2DArray messages
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;

  // Name of tensor in NitrosTensorList
  std::string tensor_name_{};

  // YOLOv8 Decoder Parameters
  double confidence_threshold_{};
  double nms_threshold_{};
  int64_t num_classes_{};
};

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_YOLOV8__YOLOV8_DECODER_NODE_HPP_
