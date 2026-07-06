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

#include "isaac_ros_rtdetr/rtdetr_preprocessor_node.hpp"

#include <algorithm>
#include <stdexcept>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "std_msgs/msg/header.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{


RtDetrPreprocessorNode::RtDetrPreprocessorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("rtdetr_preprocessor_node", options),
  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10)),
  input_image_tensor_name_{declare_parameter<std::string>(
    "input_image_tensor_name",
    "input_tensor")},
  output_image_tensor_name_{declare_parameter<std::string>(
    "output_image_tensor_name",
    "images")},
  output_size_tensor_name_{declare_parameter<std::string>(
    "output_size_tensor_name",
    "orig_target_sizes")},
  image_height_{declare_parameter<int64_t>("image_height", 480)},
  image_width_{declare_parameter<int64_t>("image_width", 640)},
  use_max_dim_for_orig_size_{declare_parameter<bool>("use_max_dim_for_orig_size", true)},
  memory_pool_block_size_{declare_parameter<int64_t>("memory_pool_block_size", 1920 * 1200 * 4)},
  memory_pool_num_blocks_{declare_parameter<int64_t>("memory_pool_num_blocks", 40)}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("RtDetrPreprocessorNode");

  // Create CUDA memory pool
  cudaError_t err = pool_.create(
    static_cast<size_t>(memory_pool_block_size_),
    static_cast<size_t>(memory_pool_num_blocks_),
    nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
  CHECK_CUDA_ERROR(err, "[RtDetrPreprocessorNode] Failed to create CUDA memory pool");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  // Create subscribers for input and output tensors
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  nitros_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "encoded_tensor", input_qos,
    std::bind(&RtDetrPreprocessorNode::InputCallback, this, std::placeholders::_1),
    sub_options);
  nitros_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosTensorList>(
    "tensor_pub", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[RtDetrPreprocessorNode] Setup complete");
}

RtDetrPreprocessorNode::~RtDetrPreprocessorNode() {}

void RtDetrPreprocessorNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList & msg)
{
  RCLCPP_DEBUG(get_logger(), "[RtDetrPreprocessorNode] InputCallback called");

  // Forward header from input message
  std_msgs::msg::Header header{};
  header.stamp.sec = msg.get_timestamp_sec();
  header.stamp.nanosec = msg.get_timestamp_nsec();
  header.frame_id = msg.get_frame_id();

  // Fetch input encoded image, perform preprocessing, then forward to output
  auto input_image_tensor_ptr =
    msg.get_tensor_by_name(input_image_tensor_name_);
  if (input_image_tensor_ptr == nullptr) {
    RCLCPP_ERROR(get_logger(), "Input image tensor is not found");
    throw std::runtime_error("Input image tensor(" + input_image_tensor_name_ + ") is not found");
  }
  const auto & input_image_tensor = *input_image_tensor_ptr;

  // Output tensor list is composed of image and  size tensors
  // Create output size tensor
  const int64_t orig_width = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_width_;
  const int64_t orig_height = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_height_;

  int64_t output_size[2]{orig_width, orig_height};
  void * output_size_buffer;
  cudaMallocAsync(&output_size_buffer, sizeof(output_size), *cuda_stream_);
  cudaMemcpyAsync(output_size_buffer, output_size, sizeof(output_size), cudaMemcpyDefault,
    *cuda_stream_);

  // Create output image tensor
  auto tensor_shape = input_image_tensor.shape();
  nvidia::isaac_ros::nitros::NitrosTensorShape shape(tensor_shape);
  nvidia::isaac_ros::nitros::NitrosTensor output_image_tensor;
  auto image_write_handle = output_image_tensor.from_pool(
    output_image_tensor_name_, pool_, shape, nvidia::isaac_ros::nitros::NitrosDataType::kFloat32,
    *cuda_stream_);

  float * output_image_buffer = reinterpret_cast<float *>(image_write_handle.get_ptr());
  auto input_image_handle = input_image_tensor.get_read_handle(*cuda_stream_);
  cudaMemcpyAsync(
    output_image_buffer, input_image_handle.get_ptr(),
    input_image_tensor.tensor_size(), cudaMemcpyDefault, *cuda_stream_);

  auto size_tensor = nvidia::isaac_ros::nitros::NitrosTensorBuilder()
    .WithShape(nvidia::isaac_ros::nitros::NitrosTensorShape({1, 2}))
    .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kInt64)
    .WithData(output_size_buffer)
    .WithName(output_size_tensor_name_)
    .Build();

  // Compose new output tensor list that contains encoded image and target size
  auto output_tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(output_image_tensor)
    .AddTensor(size_tensor)
    .Build();

  nitros_pub_->publish(output_tensor_list);
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode)
