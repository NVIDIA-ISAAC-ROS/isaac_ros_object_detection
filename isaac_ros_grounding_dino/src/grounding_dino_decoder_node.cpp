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

#include "isaac_ros_grounding_dino/grounding_dino_decoder_node.hpp"

#include <Eigen/Dense>
#include <cmath>

#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace grounding_dino
{
namespace
{

static constexpr int kNumQueries = 900;           // Number of detection queries
static constexpr int kNumTokens = 256;            // Number of text tokens
constexpr const char kDefaultQoS[] = "DEFAULT";   // Default QoS profile

template<typename T>
std::vector<T> TensorToVector(
  const Nitros::NitrosTensorListView & tensor_list,
  const std::string & tensor_name, cudaStream_t stream)
{
  auto tensor = tensor_list.GetNamedTensor(tensor_name);
  std::vector<T> vector(tensor.GetElementCount());
  cudaMemcpyAsync(
    vector.data(), tensor.GetBuffer(),
    tensor.GetTensorSize(), cudaMemcpyDefault, stream);
  return vector;
}

float sigmoid(float x)
{
  return 1.0f / (1.0f + std::exp(-x));
}

Eigen::MatrixXf GetScores(
  const Eigen::MatrixXf & pred_logits_mat,
  const Eigen::MatrixXf & pos_maps_mat)
{
  // Apply sigmoid to each element of the matrix
  Eigen::MatrixXf prob_to_token = pred_logits_mat.unaryExpr([](float x) {return sigmoid(x);});

  // Normalize pos_maps such that the mask for each label sums to 1
  Eigen::MatrixXf pos_maps_normalized = pos_maps_mat.array().colwise() /
    pos_maps_mat.rowwise().sum().array();

  // Calculate scores for each query-label combination
  return prob_to_token * pos_maps_normalized.transpose();
}

}  // namespace

GroundingDinoDecoderNode::GroundingDinoDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("grounding_dino_decoder_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, kDefaultQoS, "output_qos")},
  tensor_sub_{std::make_shared<Nitros::ManagedNitrosSubscriber<Nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      Nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&GroundingDinoDecoderNode::TensorCallback, this, std::placeholders::_1),
      Nitros::NitrosDiagnosticsConfig{}, input_qos_)},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", output_qos_)},
  boxes_tensor_name_{declare_parameter<std::string>("boxes_tensor_name", "boxes")},
  scores_tensor_name_{declare_parameter<std::string>("scores_tensor_name", "scores")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.5)},
  image_width_{static_cast<int>(declare_parameter<int64_t>("image_width", 640))},
  image_height_{static_cast<int>(declare_parameter<int64_t>("image_height", 480))}
{
  CHECK_CUDA_ERROR(
    ::nvidia::isaac_ros::common::initNamedCudaStream(
      stream_, "isaac_ros_grounding_dino_decoder_node"),
    "Error initializing CUDA stream");

  // Create callback groups
  service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create service server for SyncDataWithDecoder
  sync_data_service_ =
    create_service<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder>(
    "sync_data_with_decoder",
    std::bind(
      &GroundingDinoDecoderNode::SyncDataWithDecoderCallback, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    service_callback_group_);
}

GroundingDinoDecoderNode::~GroundingDinoDecoderNode()
{
  cudaStreamDestroy(stream_);
}

void GroundingDinoDecoderNode::SyncDataWithDecoderCallback(
  const std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder::Request>
  request,
  std::shared_ptr<isaac_ros_grounding_dino_interfaces::srv::SyncDataWithDecoder::Response> response)
{
  std::string class_ids_str = "";
  for (size_t i = 0; i < request->class_ids.size(); i++) {
    class_ids_str += request->class_ids[i];
    if (i < request->class_ids.size() - 1) {
      class_ids_str += ", ";
    }
  }
  RCLCPP_INFO(get_logger(), "Updating class IDs: [%s]", class_ids_str.c_str());

  std::lock_guard<std::mutex> lock(mutex_);
  class_ids_ = request->class_ids;
  pos_maps_ = request->pos_maps;

  response->success = true;
}

void GroundingDinoDecoderNode::TensorCallback(
  const Nitros::NitrosTensorListView & tensor_msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!class_ids_.has_value()) {
    RCLCPP_INFO(get_logger(), "Class IDs not set");
    return;
  }

  if (!pos_maps_.has_value()) {
    RCLCPP_INFO(get_logger(), "Positive maps not set");
    return;
  }

  // Bring pred_logits and pred_boxes back to CPU
  auto pred_logits = TensorToVector<float>(tensor_msg, scores_tensor_name_, stream_);
  auto pred_boxes = TensorToVector<float>(tensor_msg, boxes_tensor_name_, stream_);
  cudaStreamSynchronize(stream_);

  // Extract number of labels from pos_maps
  int num_labels = pos_maps_.value().shape.dims[0];

  // Convert pos_maps tensor data to vector
  std::vector<uint8_t> pos_maps_data;
  pos_maps_data.assign(pos_maps_.value().data.begin(), pos_maps_.value().data.end());

  // Ensure input tensors have the expected shape
  if (pred_logits.size() != kNumQueries * kNumTokens) {
    RCLCPP_ERROR(get_logger(),
      "Pred logits tensor size (%ld) does not match expected size (%d * %d = %d)",
      pred_logits.size(), kNumQueries, kNumTokens, kNumQueries * kNumTokens);
    throw std::runtime_error("Invalid pred_logits tensor size");
  }
  if (pos_maps_data.size() != static_cast<size_t>(num_labels) * kNumTokens) {
    RCLCPP_ERROR(get_logger(),
      "Positive map tensor size (%ld) does not match expected size (%d * %d = %d)",
      pos_maps_data.size(), num_labels, kNumTokens, num_labels * kNumTokens);
    throw std::runtime_error("Invalid positive map tensor size");
  }
  if (pred_boxes.size() != kNumQueries * 4) {
    RCLCPP_ERROR(get_logger(),
      "Pred boxes tensor size (%ld) does not match expected size (%d * 4 = %d)",
      pred_boxes.size(), kNumQueries, kNumQueries * 4);
    throw std::runtime_error("Invalid pred_boxes tensor size");
  }

  // Convert flat logits and positive maps to matrix form
  Eigen::MatrixXf pred_logits_mat = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
      Eigen::Dynamic, Eigen::RowMajor>>(
    pred_logits.data(), kNumQueries, kNumTokens);
  Eigen::MatrixXf pos_maps_mat = Eigen::Map<const Eigen::Matrix<uint8_t, Eigen::Dynamic,
      Eigen::Dynamic, Eigen::RowMajor>>(
    pos_maps_data.data(), num_labels, kNumTokens).cast<float>();

  // Analyze logits and positive maps to get scores for each query-label combination
  Eigen::MatrixXf scores = GetScores(pred_logits_mat, pos_maps_mat);

  // Create output message
  vision_msgs::msg::Detection2DArray detections;
  detections.header.stamp.sec = tensor_msg.GetTimestampSeconds();
  detections.header.stamp.nanosec = tensor_msg.GetTimestampNanoseconds();
  detections.header.frame_id = tensor_msg.GetFrameId();

  // Iterate through all query-label combinations
  for (int query_idx = 0; query_idx < kNumQueries; ++query_idx) {
    for (int label_idx = 0; label_idx < num_labels; ++label_idx) {
      float score = scores(query_idx, label_idx);

      // Filter out low-confidence detections
      if (score <= confidence_threshold_) {
        continue;
      }

      vision_msgs::msg::Detection2D detection;
      detection.header = detections.header;

      if (static_cast<size_t>(label_idx) >= class_ids_.value().size()) {
        RCLCPP_ERROR(get_logger(), "Class ID out of range: %d >= %ld",
          label_idx, class_ids_.value().size());
        throw std::runtime_error("Class ID out of range");
      }

      // Save score and class ID
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.score = score;
      hyp.hypothesis.class_id = class_ids_.value()[label_idx];
      detection.results.push_back(hyp);

      // Rescale and save bounding boxes in (cx, cy, w, h) format
      constexpr size_t BOX_SIZE = 4;
      detection.bbox.center.position.x = pred_boxes[query_idx * BOX_SIZE + 0] * image_width_;
      detection.bbox.center.position.y = pred_boxes[query_idx * BOX_SIZE + 1] * image_height_;
      detection.bbox.size_x = pred_boxes[query_idx * BOX_SIZE + 2] * image_width_;
      detection.bbox.size_y = pred_boxes[query_idx * BOX_SIZE + 3] * image_height_;

      detections.detections.push_back(detection);
    }
  }

  pub_->publish(detections);
}

}  // namespace grounding_dino
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode)
