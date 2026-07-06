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

#include "isaac_ros_rtdetr/rtdetr_decoder_node.hpp"

#include <stdexcept>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace rtdetr
{
namespace
{

template<typename T>
std::vector<T> TensorToVector(
  const nvidia::isaac_ros::nitros::NitrosTensorList & tensor_list,
  const std::string & tensor_name, cudaStream_t stream)
{
  auto tensor_ptr = tensor_list.get_tensor_by_name(tensor_name);
  if (tensor_ptr == nullptr) {
    RCLCPP_ERROR(rclcpp::get_logger("RtDetrDecoderNode"), "Tensor is not found");
    throw std::runtime_error("Tensor(" + tensor_name + ") is not found");
  }
  std::vector<T> vector(tensor_ptr->element_count());
  cudaMemcpyAsync(
    vector.data(), tensor_ptr->get_read_handle(stream).get_ptr(),
    tensor_ptr->tensor_size(), cudaMemcpyDefault, stream);
  return vector;
}

}  // namespace

RtDetrDecoderNode::RtDetrDecoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("rtdetr_decoder_node", options),
  // This function sets the QoS parameter for publishers and subscribers setup by this NITROS node
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10)),
  labels_tensor_name_{declare_parameter<std::string>(
      "labels_tensor_name",
      "labels")},
  boxes_tensor_name_{declare_parameter<std::string>(
      "boxes_tensor_name",
      "boxes")},
  scores_tensor_name_{declare_parameter<std::string>(
      "scores_tensor_name",
      "scores")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.9)}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("RtDetrDecoderNode");

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
    "tensor_sub", input_qos,
    std::bind(&RtDetrDecoderNode::InputCallback, this, std::placeholders::_1),
    sub_options);
  detections_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "detections_output", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[RtDetrDecoderNode] Setup complete");
}

RtDetrDecoderNode::~RtDetrDecoderNode() {}

void RtDetrDecoderNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList & msg)
{
  RCLCPP_DEBUG(get_logger(), "[RtDetrDecoderNode] InputCallback called");

  // Bring labels, boxes, and scores back to CPU
  auto labels = TensorToVector<int64_t>(msg, labels_tensor_name_, *cuda_stream_);
  auto boxes = TensorToVector<float>(msg, boxes_tensor_name_, *cuda_stream_);
  auto scores = TensorToVector<float>(msg, scores_tensor_name_, *cuda_stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_),
    "[RtDetrDecoderNode] Failed to synchronize CUDA stream");

  std_msgs::msg::Header header{};
  header.stamp.sec = msg.get_timestamp_sec();
  header.stamp.nanosec = msg.get_timestamp_nsec();
  header.frame_id = msg.get_frame_id();

  vision_msgs::msg::Detection2DArray detections;
  detections.header = header;

  for (size_t i = 0; i < scores.size(); ++i) {
    // Filter out low-confidence detections
    if (scores.at(i) <= confidence_threshold_) {
      continue;
    }

    vision_msgs::msg::Detection2D detection;
    detection.header = header;

    // Save score and label
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(labels.at(i));
    hyp.hypothesis.score = scores.at(i);
    detection.results.push_back(hyp);

    // Convert (x1, y1, x2, y2) format into (cx, cy, w, h)
    // Each bounding box is stored as 4 contiguous numbers
    constexpr size_t BOX_SIZE = 4;
    float x1 = boxes.at(BOX_SIZE * i);
    float y1 = boxes.at(BOX_SIZE * i + 1);
    float x2 = boxes.at(BOX_SIZE * i + 2);
    float y2 = boxes.at(BOX_SIZE * i + 3);

    detection.bbox.center.position.x = (x1 + x2) / 2;
    detection.bbox.center.position.y = (y1 + y2) / 2;
    detection.bbox.size_x = (x2 - x1);
    detection.bbox.size_y = (y2 - y1);

    // Add detection to output array
    detections.detections.push_back(detection);
  }

  detections_pub_->publish(detections);
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RtDetrDecoderNode)
