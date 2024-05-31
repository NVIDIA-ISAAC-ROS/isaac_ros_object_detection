// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  const nvidia::isaac_ros::nitros::NitrosTensorListView & tensor_list,
  const std::string & tensor_name, cudaStream_t stream)
{
  auto tensor = tensor_list.GetNamedTensor(tensor_name);
  std::vector<T> vector(tensor.GetElementCount());
  cudaMemcpyAsync(
    vector.data(), tensor.GetBuffer(),
    tensor.GetTensorSize(), cudaMemcpyDefault, stream);
  return vector;
}

}  // namespace

RtDetrDecoderNode::RtDetrDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&RtDetrDecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 10)},
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
  cudaStreamCreate(&stream_);
}

RtDetrDecoderNode::~RtDetrDecoderNode()
{
  cudaStreamDestroy(stream_);
}

void RtDetrDecoderNode::InputCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  // Bring labels, boxes, and scores back to CPU
  auto labels = TensorToVector<int32_t>(msg, labels_tensor_name_, stream_);
  auto boxes = TensorToVector<float>(msg, boxes_tensor_name_, stream_);
  auto scores = TensorToVector<float>(msg, scores_tensor_name_, stream_);
  cudaStreamSynchronize(stream_);

  std_msgs::msg::Header header{};
  header.stamp.sec = msg.GetTimestampSeconds();
  header.stamp.nanosec = msg.GetTimestampNanoseconds();
  header.frame_id = msg.GetFrameId();

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

  pub_->publish(detections);
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RtDetrDecoderNode)
