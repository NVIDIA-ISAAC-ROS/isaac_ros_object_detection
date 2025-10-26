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

#ifndef ISAAC_ROS_GROUNDING_DINO__GROUNDING_DINO_PREPROCESSOR_NODE_HPP_
#define ISAAC_ROS_GROUNDING_DINO__GROUNDING_DINO_PREPROCESSOR_NODE_HPP_

#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_grounding_dino_interfaces/srv/get_text_tokens.hpp"
#include "isaac_ros_grounding_dino_interfaces/srv/set_prompt.hpp"
#include "isaac_ros_grounding_dino_interfaces/srv/sync_data_with_decoder.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace grounding_dino
{

namespace Nitros = nvidia::isaac_ros::nitros;

class GroundingDinoPreprocessorNode : public rclcpp::Node
{
public:
  explicit GroundingDinoPreprocessorNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~GroundingDinoPreprocessorNode();

private:
  // Service callback for setting prompt
  void SetPromptCallback(
    const std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SetPrompt::Request> request,
    std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SetPrompt::Response> response);

  // Callback for single image tensor input
  void ImageCallback(const Nitros::NitrosTensorListView & msg);

  // Helper function to set the prompt
  bool SetPrompt(const std::string & prompt);

  // Helper function to call text tokenizer service
  bool GetTextTokens(const std::string & prompt);

  // Helper function to call update class ids service
  bool SyncDataWithDecoder();

  // QoS settings
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  // Service server for SetPrompt
  rclcpp::Service<isaac_ros_grounding_dino_interfaces::srv::SetPrompt>::SharedPtr
    set_prompt_service_;

  // Service client for GetTextTokens
  rclcpp::Client<isaac_ros_grounding_dino_interfaces::srv::GetTextTokens>::SharedPtr
    get_text_tokens_client_;

  // Service client for SyncDataWithDecoder
  rclcpp::Client<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder>::SharedPtr
    sync_data_client_;

  // Subscription to image tensor input
  std::shared_ptr<Nitros::ManagedNitrosSubscriber<Nitros::NitrosTensorListView>> image_nitros_sub_;

  // Publisher for the output tensor
  std::shared_ptr<Nitros::ManagedNitrosPublisher<Nitros::NitrosTensorList>> tensor_pub_;

  // Cached text tensors and pos maps from tokenizer
  std::optional<isaac_ros_tensor_list_interfaces::msg::TensorList> text_tensors_;
  std::optional<isaac_ros_tensor_list_interfaces::msg::Tensor> pos_maps_;
  std::optional<std::vector<std::string>> class_ids_;

  // Mutex to prevent race condition for accessing text tokens
  std::mutex mutex_;

  // CUDA stream for GPU operations
  cudaStream_t stream_;

  // Callback groups for different types of callbacks
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  rclcpp::CallbackGroup::SharedPtr get_text_tokens_cb_group_;
  rclcpp::CallbackGroup::SharedPtr sync_data_cb_group_;

  // Default prompt
  std::string default_prompt_;

  // Service call timeout in seconds
  int service_call_timeout_;

  // Service discovery timeout in seconds
  int service_discovery_timeout_;
};

}  // namespace grounding_dino
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_GROUNDING_DINO__GROUNDING_DINO_PREPROCESSOR_NODE_HPP_
