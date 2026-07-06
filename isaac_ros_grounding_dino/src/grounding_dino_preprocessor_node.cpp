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

#include <unordered_map>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_grounding_dino_interfaces/srv/sync_data_with_decoder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "std_msgs/msg/header.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace grounding_dino
{

GroundingDinoPreprocessorNode::GroundingDinoPreprocessorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("grounding_dino_preprocessor_node", options),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10)),
  input_image_tensor_name_{declare_parameter<std::string>("input_image_tensor_name",
    "input_tensor")},
  default_prompt_{declare_parameter<std::string>("default_prompt", "")},
  service_call_timeout_{static_cast<int>(declare_parameter<int>("service_call_timeout", 5))},
  service_discovery_timeout_{static_cast<int>(declare_parameter<int>(
      "service_discovery_timeout", 5))},
  memory_pool_block_size_(declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)),
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40))
{
  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("GroundingDinoPreprocessorNode");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_nitros_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "image_tensor", input_qos,
    std::bind(&GroundingDinoPreprocessorNode::ImageCallback, this, std::placeholders::_1),
    sub_options);
  tensor_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>("tensor_pub",
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
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & msg)
{
  // Use default prompt if no text tensors are cached
  if (!text_tensors_.has_value() || !pos_maps_.has_value()) {
    RCLCPP_INFO(get_logger(), "Setting default prompt: %s", default_prompt_.c_str());
    SetPrompt(default_prompt_);
    return;
  }

  // Forward header from input message
  std_msgs::msg::Header header{};
  header.stamp.sec = msg->get_timestamp_sec();
  header.stamp.nanosec = msg->get_timestamp_nsec();
  header.frame_id = msg->get_frame_id();

  // Process image tensor
  auto image_tensor_ptr = msg->get_tensor_by_name(input_image_tensor_name_);
  if (image_tensor_ptr == nullptr) {
    RCLCPP_ERROR(get_logger(),
            "Image tensor '%s' is not found (set input_image_tensor_name to match upstream)",
      input_image_tensor_name_.c_str());
    throw std::runtime_error("Image tensor '" + input_image_tensor_name_ + "' is not found");
  }

  const auto & image_tensor = *image_tensor_ptr;
  if (!pool_.initialized()) {
    const int64_t actual_block_size = std::max(
      static_cast<int64_t>(image_tensor.tensor_size()), memory_pool_block_size_);
    CHECK_CUDA_ERROR(pool_.create(
      static_cast<size_t>(actual_block_size),
      static_cast<size_t>(memory_pool_num_blocks_),
      nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device),
      "[GroundingDinoPreprocessorNode] Failed to create CUDA memory pool");
  }

  // Process text tensors
  std::unordered_map<std::string, void *> gpu_buffers;
  for (const auto & tensor : text_tensors_.value().tensors) {
    void * gpu_buffer;
    size_t tensor_size = tensor.data.size();
    CHECK_CUDA_ERROR(
      cudaMallocAsync(&gpu_buffer, tensor_size, *cuda_stream_),
      "[GroundingDinoPreprocessorNode] Failed to allocate memory for tensor: %s",
      tensor.name.c_str());
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      gpu_buffer, tensor.data.data(),
      tensor_size, cudaMemcpyHostToDevice, *cuda_stream_),
      "[GroundingDinoPreprocessorNode] Failed to copy data for tensor: %s", tensor.name.c_str());
    gpu_buffers[tensor.name] = gpu_buffer;
  }

  // Create output image tensor
  auto tensor_shape = image_tensor_ptr->shape();
  nvidia::isaac_ros::nitros::NitrosTensorShape shape(tensor_shape);
  nvidia::isaac_ros::nitros::NitrosTensor output_image_tensor;
  auto image_write_handle = output_image_tensor.from_pool(
    "images",
    pool_,
    shape,
    nvidia::isaac_ros::nitros::NitrosDataType::kFloat32,
    *cuda_stream_);

  float * image_output_buffer = reinterpret_cast<float *>(image_write_handle.get_ptr());
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    image_output_buffer, image_tensor.get_read_handle(*cuda_stream_).get_ptr(),
    image_tensor.tensor_size(), cudaMemcpyDefault, *cuda_stream_),
    "[GroundingDinoPreprocessorNode] Failed to copy data for image tensor");

  // Build output tensor list and add image tensor
  auto output_tensor_builder =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder().WithHeader(header);
  output_tensor_builder.AddTensor("images", output_image_tensor);

  // Add text tensors
  for (const auto & tensor : text_tensors_.value().tensors) {
    std::vector<int32_t> dims(tensor.shape.dims.begin(), tensor.shape.dims.end());
    Nitros::NitrosTensorShape shape{dims};

    auto nitros_tensor = Nitros::NitrosTensorBuilder()
      .WithShape(shape)
      .WithDataType(static_cast<nvidia::isaac_ros::nitros::NitrosDataType>(tensor.data_type))
      .WithData(gpu_buffers[tensor.name])
      .WithReleaseCallback([gpu_buffer = gpu_buffers[tensor.name], stream = *cuda_stream_]() {
          cudaFreeAsync(gpu_buffer, stream);
      })
      .WithName(tensor.name)
      .Build();
    output_tensor_builder.AddTensor(tensor.name, nitros_tensor);
  }

  tensor_pub_->publish(output_tensor_builder.Build());
}

}  // namespace grounding_dino
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode)
