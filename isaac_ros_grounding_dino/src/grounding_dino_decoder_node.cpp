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

#include "isaac_ros_grounding_dino/grounding_dino_decoder_node.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"

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

size_t ElementCount(const Tensor & tensor)
{
  if (tensor.shape.empty()) {
    throw std::invalid_argument("[GroundingDinoDecoderNode] Tensor shape is empty");
  }
  size_t count = 1;
  for (const int64_t dimension : tensor.shape) {
    if (dimension <= 0) {
      throw std::invalid_argument("[GroundingDinoDecoderNode] Tensor dimensions must be positive");
    }
    const size_t extent = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / extent) {
      throw std::overflow_error("[GroundingDinoDecoderNode] Tensor element count overflow");
    }
    count *= extent;
  }
  return count;
}

std::vector<float> TensorToVector(
  const TensorList & tensor_list,
  const std::string & tensor_name, cudaStream_t stream)
{
  const Tensor * tensor = isaac_ros_tensor_msgs::FindTensorByName(tensor_list, tensor_name);
  if (tensor == nullptr) {
    throw std::runtime_error(
            "[GroundingDinoDecoderNode] Tensor '" + tensor_name + "' is not found");
  }
  constexpr uint8_t kDLPackFloat = 2;
  if (tensor->dtype_code != kDLPackFloat || tensor->dtype_bits != 32 ||
    tensor->dtype_lanes != 1)
  {
    throw std::invalid_argument(
            "[GroundingDinoDecoderNode] Tensor '" + tensor_name + "' must be float32");
  }

  const size_t element_count = ElementCount(*tensor);
  const size_t storage_count = isaac_ros_tensor_msgs::RequiredStorageElements(*tensor);
  if (storage_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::overflow_error("[GroundingDinoDecoderNode] Tensor byte size overflow");
  }
  const size_t byte_count = storage_count * sizeof(float);
  if (tensor->byte_offset > tensor->data.size() ||
    byte_count > tensor->data.size() - static_cast<size_t>(tensor->byte_offset))
  {
    throw std::invalid_argument(
            "[GroundingDinoDecoderNode] Tensor '" + tensor_name + "' buffer is too small");
  }

  std::vector<float> storage(storage_count);
  auto input_handle = cuda_buffer_backend::from_input_buffer(tensor->data, stream);
  const cudaError_t copy_result = cudaMemcpyAsync(
    storage.data(), input_handle.get_ptr() + tensor->byte_offset, byte_count,
    cudaMemcpyDeviceToHost, stream);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(
            std::string("[GroundingDinoDecoderNode] Device-to-host copy failed: ") +
            cudaGetErrorString(copy_result));
  }
  const cudaError_t sync_result = cudaStreamSynchronize(stream);
  if (sync_result != cudaSuccess) {
    throw std::runtime_error(
            std::string("[GroundingDinoDecoderNode] CUDA stream synchronization failed: ") +
            cudaGetErrorString(sync_result));
  }

  if (tensor->strides.empty()) {
    return storage;
  }

  std::vector<float> dense(element_count);
  for (size_t linear_index = 0; linear_index < element_count; ++linear_index) {
    size_t remainder = linear_index;
    size_t storage_index = 0;
    for (size_t dimension = tensor->shape.size(); dimension-- > 0; ) {
      const size_t extent = static_cast<size_t>(tensor->shape[dimension]);
      storage_index +=
        (remainder % extent) * isaac_ros_tensor_msgs::StrideInElements(*tensor, dimension);
      remainder /= extent;
    }
    dense[linear_index] = storage[storage_index];
  }
  return dense;
}

std::vector<uint8_t> TensorToUint8Vector(const Tensor & tensor, cudaStream_t stream)
{
  constexpr uint8_t kDLPackUInt = 1;
  if (tensor.dtype_code != kDLPackUInt || tensor.dtype_bits != 8 || tensor.dtype_lanes != 1) {
    throw std::invalid_argument(
            "[GroundingDinoDecoderNode] Positive map tensor must be uint8");
  }

  const size_t element_count = ElementCount(tensor);
  const size_t storage_count = isaac_ros_tensor_msgs::RequiredStorageElements(tensor);
  if (tensor.byte_offset > tensor.data.size() ||
    storage_count > tensor.data.size() - static_cast<size_t>(tensor.byte_offset))
  {
    throw std::invalid_argument(
            "[GroundingDinoDecoderNode] Positive map tensor buffer is too small");
  }

  std::vector<uint8_t> storage(storage_count);
  auto input_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream);
  const cudaError_t copy_result = cudaMemcpyAsync(
    storage.data(), input_handle.get_ptr() + tensor.byte_offset, storage_count,
    cudaMemcpyDeviceToHost, stream);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(
            std::string("[GroundingDinoDecoderNode] Device-to-host copy failed: ") +
            cudaGetErrorString(copy_result));
  }
  const cudaError_t sync_result = cudaStreamSynchronize(stream);
  if (sync_result != cudaSuccess) {
    throw std::runtime_error(
            std::string("[GroundingDinoDecoderNode] CUDA stream synchronization failed: ") +
            cudaGetErrorString(sync_result));
  }

  if (tensor.strides.empty()) {
    return storage;
  }

  std::vector<uint8_t> dense(element_count);
  for (size_t linear_index = 0; linear_index < element_count; ++linear_index) {
    size_t remainder = linear_index;
    size_t storage_index = 0;
    for (size_t dimension = tensor.shape.size(); dimension-- > 0; ) {
      const size_t extent = static_cast<size_t>(tensor.shape[dimension]);
      storage_index +=
        (remainder % extent) * isaac_ros_tensor_msgs::StrideInElements(tensor, dimension);
      remainder /= extent;
    }
    dense[linear_index] = storage[storage_index];
  }
  return dense;
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

GroundingDinoDecoderNode::GroundingDinoDecoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("grounding_dino_decoder_node", options),
  input_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos").keep_last(10)),
  output_qos_(::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos").keep_last(10)),
  boxes_tensor_name_{declare_parameter<std::string>("boxes_tensor_name", "boxes")},
  scores_tensor_name_{declare_parameter<std::string>("scores_tensor_name", "scores")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.5)},
  image_width_{static_cast<int>(declare_parameter<int64_t>("image_width", 640))},
  image_height_{static_cast<int>(declare_parameter<int64_t>("image_height", 480))}
{
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("GroundingDinoDecoderNode");

  // Create callback groups
  service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<TensorList>(
    "tensor_sub", input_qos_,
    std::bind(&GroundingDinoDecoderNode::TensorCallback, this, std::placeholders::_1),
    sub_options);
  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("detections_output", output_qos_,
    pub_options);

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

GroundingDinoDecoderNode::~GroundingDinoDecoderNode() {}

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
  const TensorList::ConstSharedPtr & tensor_msg)
{
  if (tensor_msg->names.size() != tensor_msg->tensors.size()) {
    RCLCPP_ERROR(get_logger(), "Tensor names and tensors must have the same size");
    return;
  }

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
  std::vector<float> pred_logits;
  std::vector<float> pred_boxes;
  std::vector<uint8_t> pos_maps_data;
  int num_labels;
  try {
    pred_logits = TensorToVector(*tensor_msg, scores_tensor_name_, *cuda_stream_);
    pred_boxes = TensorToVector(*tensor_msg, boxes_tensor_name_, *cuda_stream_);
    const Tensor & pos_maps = pos_maps_.value();
    if (class_ids_->size() > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
              "[GroundingDinoDecoderNode] Class ID count exceeds the supported range");
    }
    num_labels = static_cast<int>(class_ids_->size());
    if (num_labels > 0) {
      pos_maps_data = TensorToUint8Vector(pos_maps, *cuda_stream_);
    }
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "%s", error.what());
    return;
  }

  // Ensure input tensors have the expected shape
  if (pred_logits.size() != kNumQueries * kNumTokens) {
    RCLCPP_ERROR(get_logger(),
      "Pred logits tensor size (%ld) does not match expected size (%d * %d = %d)",
      pred_logits.size(), kNumQueries, kNumTokens, kNumQueries * kNumTokens);
    return;
  }
  if (pos_maps_data.size() != static_cast<size_t>(num_labels) * kNumTokens) {
    RCLCPP_ERROR(get_logger(),
      "Positive map tensor size (%ld) does not match expected size (%d * %d = %d)",
      pos_maps_data.size(), num_labels, kNumTokens, num_labels * kNumTokens);
    return;
  }
  if (pred_boxes.size() != kNumQueries * 4) {
    RCLCPP_ERROR(get_logger(),
      "Pred boxes tensor size (%ld) does not match expected size (%d * 4 = %d)",
      pred_boxes.size(), kNumQueries, kNumQueries * 4);
    return;
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
  detections.header = tensor_msg->header;

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
