// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{


RtDetrPreprocessorNode::RtDetrPreprocessorNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_preprocessor_node", options),
  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "encoded_tensor",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&RtDetrPreprocessorNode::InputCallback, this,
      std::placeholders::_1),
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, input_qos_)},
  nitros_pub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosTensorList>>(
      this,
      "tensor_pub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      nvidia::isaac_ros::nitros::NitrosDiagnosticsConfig{}, output_qos_)},
  input_image_tensor_name_{declare_parameter<std::string>(
      "input_image_tensor_name",
      "input_tensor")},
  output_image_tensor_name_{declare_parameter<std::string>("output_image_tensor_name", "images")},
  output_size_tensor_name_{declare_parameter<std::string>(
      "output_size_tensor_name",
      "orig_target_sizes")},
  image_height_{declare_parameter<int64_t>("image_height", 480)},
  image_width_{declare_parameter<int64_t>("image_width", 640)},
  use_max_dim_for_orig_size_{declare_parameter<bool>("use_max_dim_for_orig_size", true)}
{
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      stream_, "isaac_ros_rtdetr_preprocessor_node"),
    "Error initializing CUDA stream");
}

RtDetrPreprocessorNode::~RtDetrPreprocessorNode()
{
  cudaStreamDestroy(stream_);
}

void RtDetrPreprocessorNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  // Forward header from input message
  std_msgs::msg::Header header{};
  header.stamp.sec = msg.GetTimestampSeconds();
  header.stamp.nanosec = msg.GetTimestampNanoseconds();
  header.frame_id = msg.GetFrameId();

  // Fetch input encoded image, perform preprocessing, then forward to output
  auto input_image_tensor = msg.GetNamedTensor(input_image_tensor_name_);
  float * output_image_buffer;
  cudaMallocAsync(&output_image_buffer, input_image_tensor.GetTensorSize(), stream_);
  cudaMemcpyAsync(
    output_image_buffer, input_image_tensor.GetBuffer(),
    input_image_tensor.GetTensorSize(), cudaMemcpyDefault, stream_);

  const int64_t orig_width = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_width_;
  const int64_t orig_height = use_max_dim_for_orig_size_ ?
    std::max(image_height_, image_width_) : image_height_;

  int64_t output_size[2]{orig_width, orig_height};
  void * output_size_buffer;
  cudaMallocAsync(&output_size_buffer, sizeof(output_size), stream_);
  cudaMemcpyAsync(output_size_buffer, output_size, sizeof(output_size), cudaMemcpyDefault, stream_);
  cudaStreamSynchronize(stream_);

  // Compose new output tensor list that contains encoded image and target size
  auto output_tensor_list =
    nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(header)
    .AddTensor(
    output_image_tensor_name_,
    (
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
      .WithShape(input_image_tensor.GetShape())
      .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
      .WithData(output_image_buffer)
      .Build()
    )
    )
    .AddTensor(
    output_size_tensor_name_,
    (
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
      .WithShape({1, 2})
      .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kInt64)
      .WithData(output_size_buffer)
      .Build()
    )
    )
    .Build();

  nitros_pub_->publish(output_tensor_list);
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode)
