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
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{

namespace
{

Tensor CloneTensor(
  const Tensor & input, cudaStream_t stream)
{
  if (input.data.empty()) {
    throw std::invalid_argument("[RtDetrPreprocessorNode] Input image tensor data is empty");
  }

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
  const cudaError_t err = cudaMemcpyAsync(
    output_handle.get_ptr(), input_handle.get_ptr(), input.data.size(),
    cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("[RtDetrPreprocessorNode] Failed to copy image tensor: ") +
            cudaGetErrorString(err));
  }
  return output;
}

Tensor MakeSizeTensor(
  int64_t width, int64_t height, cudaStream_t stream)
{
  constexpr uint8_t kDLPackInt = 0;
  const int64_t values[2]{width, height};

  Tensor tensor;
  tensor.dtype_code = kDLPackInt;
  tensor.dtype_bits = 64;
  tensor.dtype_lanes = 1;
  tensor.shape = {1, 2};
  tensor.byte_offset = 0;
  tensor.data = cuda_buffer_backend::allocate_buffer(sizeof(values));

  auto output_handle = cuda_buffer_backend::from_output_buffer(tensor.data, stream);
  const cudaError_t err = cudaMemcpyAsync(
    output_handle.get_ptr(), values, sizeof(values), cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("[RtDetrPreprocessorNode] Failed to copy size tensor: ") +
            cudaGetErrorString(err));
  }
  return tensor;
}

}  // namespace

RtDetrPreprocessorNode::RtDetrPreprocessorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("rtdetr_preprocessor_node", options),
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
  use_max_dim_for_orig_size_{declare_parameter<bool>("use_max_dim_for_orig_size", true)}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("RtDetrPreprocessorNode");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  // Create subscribers for input and output tensors
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<TensorList>(
    "encoded_tensor", input_qos,
    std::bind(&RtDetrPreprocessorNode::InputCallback, this, std::placeholders::_1),
    sub_options);
  tensor_pub_ = create_publisher<TensorList>(
    "tensor_pub", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[RtDetrPreprocessorNode] Setup complete");
}

RtDetrPreprocessorNode::~RtDetrPreprocessorNode() {}

void RtDetrPreprocessorNode::InputCallback(
  const TensorList::ConstSharedPtr msg)
{
  RCLCPP_DEBUG(get_logger(), "[RtDetrPreprocessorNode] InputCallback called");

  const Tensor * input_image_tensor =
    isaac_ros_tensor_msgs::FindTensorByName(*msg, input_image_tensor_name_);
  if (input_image_tensor == nullptr) {
    RCLCPP_ERROR(
      get_logger(), "[RtDetrPreprocessorNode] Input image tensor '%s' is not found",
      input_image_tensor_name_.c_str());
    return;
  }

  const int64_t orig_width = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_width_;
  const int64_t orig_height = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_height_;

  try {
    Tensor output_image_tensor =
      CloneTensor(*input_image_tensor, *cuda_stream_);
    Tensor size_tensor =
      MakeSizeTensor(orig_width, orig_height, *cuda_stream_);

    TensorList output_tensor_list;
    output_tensor_list.header = msg->header;
    output_tensor_list.names = {output_image_tensor_name_, output_size_tensor_name_};
    output_tensor_list.tensors.reserve(2);
    output_tensor_list.tensors.push_back(std::move(output_image_tensor));
    output_tensor_list.tensors.push_back(std::move(size_tensor));
    tensor_pub_->publish(std::move(output_tensor_list));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "%s", error.what());
  }
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode)
