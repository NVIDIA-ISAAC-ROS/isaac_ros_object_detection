// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_grounding_dino/grounding_dino_preprocessor_node.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_grounding_dino_interfaces/srv/sync_data_with_decoder.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace grounding_dino
{
namespace
{

size_t TensorElementSize(const Tensor & tensor)
{
  if (tensor.dtype_bits == 0 || tensor.dtype_lanes == 0) {
    throw std::invalid_argument(
            "[GroundingDinoPreprocessorNode] Tensor dtype bits and lanes must be positive");
  }
  const size_t bit_count =
    static_cast<size_t>(tensor.dtype_bits) * static_cast<size_t>(tensor.dtype_lanes);
  return (bit_count + 7) / 8;
}

void ValidateTensorBuffer(const Tensor & tensor)
{
  if (tensor.data.empty()) {
    throw std::invalid_argument(
            "[GroundingDinoPreprocessorNode] Tensor data buffer is empty");
  }
  const size_t storage_count = isaac_ros_tensor_msgs::RequiredStorageElements(tensor);
  const size_t element_size = TensorElementSize(tensor);
  if (storage_count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::overflow_error("[GroundingDinoPreprocessorNode] Tensor byte size overflow");
  }
  const size_t byte_count = storage_count * element_size;
  if (tensor.byte_offset > tensor.data.size() ||
    byte_count > tensor.data.size() - static_cast<size_t>(tensor.byte_offset))
  {
    throw std::invalid_argument(
            "[GroundingDinoPreprocessorNode] Tensor data buffer is too small");
  }
}

Tensor CopyTensor(const Tensor & input, cudaStream_t stream)
{
  ValidateTensorBuffer(input);
  Tensor output;
  output.dtype_code = input.dtype_code;
  output.dtype_bits = input.dtype_bits;
  output.dtype_lanes = input.dtype_lanes;
  output.shape = input.shape;
  output.strides = input.strides;
  output.byte_offset = input.byte_offset;
  output.data = cuda_buffer_backend::allocate_buffer(input.data.size());

  auto input_handle = cuda_buffer_backend::from_input_buffer(input.data, stream);
  auto output_handle = cuda_buffer_backend::from_output_buffer(output.data, stream);
  cuda_buffer_backend::to_buffer(
    input_handle.get_ptr(), input.data.size(), output_handle, stream);
  return output;
}

}  // namespace

GroundingDinoPreprocessorNode::GroundingDinoPreprocessorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("grounding_dino_preprocessor_node", options),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10)),
  input_image_tensor_name_{declare_parameter<std::string>("input_image_tensor_name",
    "input_tensor")},
  default_prompt_{declare_parameter<std::string>("default_prompt", "")},
  service_call_timeout_{static_cast<int>(declare_parameter<int>("service_call_timeout", 5))},
  service_discovery_timeout_{static_cast<int>(declare_parameter<int>(
      "service_discovery_timeout", 5))}
{
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("GroundingDinoPreprocessorNode");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_tensor_sub_ = create_subscription<TensorList>(
    "image_tensor", input_qos,
    std::bind(&GroundingDinoPreprocessorNode::ImageCallback, this, std::placeholders::_1),
    sub_options);
  tensor_pub_ = create_publisher<TensorList>("tensor_pub",
    output_qos, pub_options);

  // Create callback groups
  service_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  get_text_tokens_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  sync_data_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create service server for SetPrompt
  set_prompt_service_ = create_service<isaac_ros_grounding_dino_interfaces::srv::SetPrompt>(
    "set_prompt",
    std::bind(
      &GroundingDinoPreprocessorNode::SetPromptCallback, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_cb_group_);

  // Create service client for GetTextTokens
  get_text_tokens_client_ = create_client<isaac_ros_grounding_dino_interfaces::srv::GetTextTokens>(
    "get_text_tokens", rclcpp::ServicesQoS(), get_text_tokens_cb_group_);

  // Create service client for SyncDataWithDecoder
  sync_data_client_ = create_client<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder>(
    "sync_data_with_decoder", rclcpp::ServicesQoS(), sync_data_cb_group_);
}

GroundingDinoPreprocessorNode::~GroundingDinoPreprocessorNode() {}

void GroundingDinoPreprocessorNode::SetPromptCallback(
  const std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SetPrompt::Request> request,
  std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SetPrompt::Response> response)
{
  response->success = SetPrompt(request->prompt);
}

bool GroundingDinoPreprocessorNode::SetPrompt(const std::string & prompt)
{
  RCLCPP_INFO(get_logger(), "Setting new prompt: %s", prompt.c_str());
  return GetTextTokens(prompt) && SyncDataWithDecoder();
}

bool GroundingDinoPreprocessorNode::GetTextTokens(const std::string & prompt)
{
  if (!get_text_tokens_client_->wait_for_service(std::chrono::seconds(
          service_discovery_timeout_)))
  {
    RCLCPP_ERROR(get_logger(), "GetTextTokens service not available");
    return false;
  }

  auto request =
    std::make_shared<isaac_ros_grounding_dino_interfaces::srv::GetTextTokens::Request>();
  request->prompt = prompt;

  auto future = get_text_tokens_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(service_call_timeout_)) == std::future_status::timeout) {
    RCLCPP_ERROR(get_logger(), "GetTextTokens service call timed out");
    return false;
  }

  auto response = future.get();

  if (response->text_tensors.names.size() != response->text_tensors.tensors.size()) {
    RCLCPP_ERROR(get_logger(), "Text tensor names and tensors must have the same size");
    return false;
  }
  text_tensors_ = response->text_tensors;
  pos_maps_ = response->pos_maps;
  class_ids_ = response->class_ids;

  RCLCPP_INFO(get_logger(), "Successfully generated text tokens for prompt: %s", prompt.c_str());
  return true;
}

bool GroundingDinoPreprocessorNode::SyncDataWithDecoder()
{
  if (!class_ids_.has_value()) {
    RCLCPP_ERROR(get_logger(), "The class IDs have not been generated");
    return false;
  }
  if (!pos_maps_.has_value()) {
    RCLCPP_ERROR(get_logger(), "The positive map tensor has not been generated");
    return false;
  }

  if (!sync_data_client_->wait_for_service(std::chrono::seconds(service_discovery_timeout_))) {
    RCLCPP_ERROR(get_logger(), "SyncDataWithDecoder service not available");
    return false;
  }

  auto request = std::make_shared<
    isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder::Request>();
  request->class_ids = class_ids_.value();
  request->pos_maps = pos_maps_.value();

  auto future = sync_data_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(service_call_timeout_)) == std::future_status::timeout) {
    RCLCPP_ERROR(get_logger(), "SyncDataWithDecoder service call timed out");
    return false;
  }

  auto response = future.get();
  if (response->success) {
    RCLCPP_INFO(get_logger(), "Successfully synced data between preprocessor and decoder");
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to sync data between preprocessor and decoder");
  }
  return response->success;
}

void GroundingDinoPreprocessorNode::ImageCallback(
  const TensorList::ConstSharedPtr & msg)
{
  if (msg->names.size() != msg->tensors.size()) {
    RCLCPP_ERROR(get_logger(), "Image tensor names and tensors must have the same size");
    return;
  }

  // Use default prompt if no text tensors are cached
  if (!text_tensors_.has_value() || !pos_maps_.has_value()) {
    RCLCPP_INFO(get_logger(), "Setting default prompt: %s", default_prompt_.c_str());
    SetPrompt(default_prompt_);
    return;
  }

  // Process image tensor
  const Tensor * image_tensor_ptr =
    isaac_ros_tensor_msgs::FindTensorByName(*msg, input_image_tensor_name_);
  if (image_tensor_ptr == nullptr) {
    RCLCPP_ERROR(get_logger(),
            "Image tensor '%s' is not found (set input_image_tensor_name to match upstream)",
      input_image_tensor_name_.c_str());
    return;
  }

  try {
    TensorList output;
    output.header = msg->header;
    output.names.reserve(1 + text_tensors_->tensors.size());
    output.tensors.reserve(1 + text_tensors_->tensors.size());
    output.names.push_back("images");
    output.tensors.push_back(CopyTensor(*image_tensor_ptr, *cuda_stream_));

    for (size_t i = 0; i < text_tensors_->tensors.size(); ++i) {
      output.names.push_back(text_tensors_->names[i]);
      output.tensors.push_back(CopyTensor(text_tensors_->tensors[i], *cuda_stream_));
    }
    tensor_pub_->publish(std::move(output));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "%s", error.what());
  }
}

}  // namespace grounding_dino
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode)
