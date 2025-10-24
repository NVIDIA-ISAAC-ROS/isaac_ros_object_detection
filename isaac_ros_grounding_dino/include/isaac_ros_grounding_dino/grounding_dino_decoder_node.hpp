// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_GROUNDING_DINO__GROUNDING_DINO_DECODER_NODE_HPP_
#define ISAAC_ROS_GROUNDING_DINO__GROUNDING_DINO_DECODER_NODE_HPP_

#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_grounding_dino_interfaces/srv/sync_data_with_decoder.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace grounding_dino
{

namespace Nitros = nvidia::isaac_ros::nitros;

class GroundingDinoDecoderNode : public rclcpp::Node
{
public:
  explicit GroundingDinoDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~GroundingDinoDecoderNode();

private:
  void TensorCallback(const Nitros::NitrosTensorListView & tensor_msg);

  // Service callback for synchronizing data between preprocessor and decoder
  void SyncDataWithDecoderCallback(
    const std::shared_ptr<
      isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder::Request> request,
    std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder::Response>
    response);

  // QOS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Service server for SyncDataWithDecoder
  rclcpp::Service<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder>::SharedPtr
    sync_data_service_;

  // Subscription to tensor input
  std::shared_ptr<Nitros::ManagedNitrosSubscriber<Nitros::NitrosTensorListView>> tensor_sub_;

  // Publisher for output Detection2DArray messages
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;

  std::string boxes_tensor_name_;
  std::string scores_tensor_name_;
  double confidence_threshold_;
  int image_width_;
  int image_height_;
  std::optional<std::vector<std::string>> class_ids_;
  std::optional<isaac_ros_tensor_list_interfaces::msg::Tensor> pos_maps_;

  // Mutex to prevent race condition for accessing class ids and pos maps
  std::mutex mutex_;

  // CUDA stream for GPU operations
  cudaStream_t stream_;

  // Callback groups
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
};

}  // namespace grounding_dino
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GROUNDING_DINO__GROUNDING_DINO_DECODER_NODE_HPP_
