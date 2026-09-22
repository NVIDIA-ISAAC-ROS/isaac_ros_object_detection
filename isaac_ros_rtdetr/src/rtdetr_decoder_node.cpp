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
#include "isaac_ros_rtdetr/rtdetr_decoder_utils.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
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

size_t ElementCount(const Tensor & tensor)
{
  size_t count = 1;
  for (const int64_t dim : tensor.shape) {
    if (dim < 0) {
      throw std::invalid_argument("[RtDetrDecoderNode] Negative tensor dimension");
    }
    const size_t extent = static_cast<size_t>(dim);
    if (extent != 0 && count > std::numeric_limits<size_t>::max() / extent) {
      throw std::overflow_error("[RtDetrDecoderNode] Tensor element count overflow");
    }
    count *= extent;
  }
  return count;
}

template<typename T>
std::vector<T> TensorToVector(
  const TensorList & tensor_list, const std::string & tensor_name,
  uint8_t expected_dtype_code, uint8_t expected_dtype_bits, cudaStream_t stream)
{
  const Tensor * tensor_ptr =
    isaac_ros_tensor_msgs::FindTensorByName(tensor_list, tensor_name);
  if (tensor_ptr == nullptr) {
    throw std::runtime_error("[RtDetrDecoderNode] Tensor '" + tensor_name + "' is not found");
  }
  if (tensor_ptr->dtype_code != expected_dtype_code ||
    tensor_ptr->dtype_bits != expected_dtype_bits || tensor_ptr->dtype_lanes != 1)
  {
    throw std::invalid_argument(
            "[RtDetrDecoderNode] Tensor '" + tensor_name + "' has an unexpected data type");
  }

  const size_t element_count = ElementCount(*tensor_ptr);
  if (element_count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error("[RtDetrDecoderNode] Tensor byte size overflow");
  }
  const size_t byte_count = element_count * sizeof(T);
  if (tensor_ptr->byte_offset > tensor_ptr->data.size() ||
    byte_count > tensor_ptr->data.size() - static_cast<size_t>(tensor_ptr->byte_offset))
  {
    throw std::invalid_argument(
            "[RtDetrDecoderNode] Tensor '" + tensor_name + "' data buffer is too small");
  }

  std::vector<T> vector(element_count);
  if (byte_count == 0) {
    return vector;
  }

  auto input_handle = cuda_buffer_backend::from_input_buffer(tensor_ptr->data, stream);
  const cudaError_t err = cudaMemcpyAsync(
    vector.data(), input_handle.get_ptr() + tensor_ptr->byte_offset,
    byte_count, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("[RtDetrDecoderNode] Failed to copy tensor '") + tensor_name +
            "': " + cudaGetErrorString(err));
  }
  return vector;
}

}  // namespace

RtDetrDecoderNode::RtDetrDecoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("rtdetr_decoder_node", options),
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
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<TensorList>(
    "tensor_sub", input_qos,
    std::bind(&RtDetrDecoderNode::InputCallback, this, std::placeholders::_1),
    sub_options);
  detections_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "detections_output", output_qos, pub_options);

  RCLCPP_DEBUG(get_logger(), "[RtDetrDecoderNode] Setup complete");
}

RtDetrDecoderNode::~RtDetrDecoderNode() {}

void RtDetrDecoderNode::InputCallback(
  const TensorList::ConstSharedPtr msg)
{
  RCLCPP_DEBUG(get_logger(), "[RtDetrDecoderNode] InputCallback called");

  constexpr uint8_t kDLPackInt = 0;
  constexpr uint8_t kDLPackFloat = 2;
  std::vector<int64_t> labels;
  std::vector<float> boxes;
  std::vector<float> scores;
  try {
    labels = TensorToVector<int64_t>(
      *msg, labels_tensor_name_, kDLPackInt, 64, *cuda_stream_);
    boxes = TensorToVector<float>(
      *msg, boxes_tensor_name_, kDLPackFloat, 32, *cuda_stream_);
    scores = TensorToVector<float>(
      *msg, scores_tensor_name_, kDLPackFloat, 32, *cuda_stream_);
    const cudaError_t err = cudaStreamSynchronize(*cuda_stream_);
    if (err != cudaSuccess) {
      throw std::runtime_error(
              std::string("[RtDetrDecoderNode] Failed to synchronize CUDA stream: ") +
              cudaGetErrorString(err));
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "%s", error.what());
    return;
  }

  if (!AreOutputTensorSizesValid(labels.size(), boxes.size(), scores.size())) {
    RCLCPP_ERROR(
      get_logger(),
      "Output tensor size mismatch: labels=%zu, boxes=%zu, scores=%zu; expected one label "
      "and %zu box elements per score",
      labels.size(), boxes.size(), scores.size(), kBoundingBoxElementCount);
    return;
  }

  vision_msgs::msg::Detection2DArray detections;
  detections.header = msg->header;

  for (size_t i = 0; i < scores.size(); ++i) {
    // Filter out low-confidence detections
    if (scores.at(i) <= confidence_threshold_) {
      continue;
    }

    vision_msgs::msg::Detection2D detection;
    detection.header = msg->header;

    // Save score and label
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(labels.at(i));
    hyp.hypothesis.score = scores.at(i);
    detection.results.push_back(hyp);

    // Convert (x1, y1, x2, y2) format into (cx, cy, w, h)
    // Each bounding box is stored as 4 contiguous numbers
    float x1 = boxes.at(kBoundingBoxElementCount * i);
    float y1 = boxes.at(kBoundingBoxElementCount * i + 1);
    float x2 = boxes.at(kBoundingBoxElementCount * i + 2);
    float y2 = boxes.at(kBoundingBoxElementCount * i + 3);

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
