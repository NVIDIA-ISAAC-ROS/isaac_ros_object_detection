// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace yolov8
{
namespace
{
constexpr uint8_t kDLPackFloat = 2;
constexpr size_t kBatchDimension = 0;
constexpr size_t kFeatureDimension = 1;
constexpr size_t kDetectionDimension = 2;
constexpr int64_t kBoxParameters = 4;

void ValidateTensor(const Tensor & tensor, int64_t num_classes)
{
  if (tensor.dtype_code != kDLPackFloat || tensor.dtype_bits != 32 ||
    tensor.dtype_lanes != 1)
  {
    throw std::invalid_argument("[YoloV8DecoderNode] Input tensor must be float32");
  }
  if (tensor.shape.size() != 3) {
    throw std::invalid_argument(
            "[YoloV8DecoderNode] Input tensor must have shape [batch, features, detections]");
  }
  if (num_classes <= 0 ||
    tensor.shape[kBatchDimension] != 1 ||
    tensor.shape[kFeatureDimension] != kBoxParameters + num_classes ||
    tensor.shape[kDetectionDimension] <= 0)
  {
    throw std::invalid_argument("[YoloV8DecoderNode] Input tensor has an unexpected shape");
  }
  if (tensor.shape[kDetectionDimension] > std::numeric_limits<int>::max()) {
    throw std::overflow_error("[YoloV8DecoderNode] Detection dimension is too large");
  }
}

std::vector<float> CopyTensorToHost(
  const Tensor & tensor, cudaStream_t stream)
{
  const size_t element_count = isaac_ros_tensor_msgs::RequiredStorageElements(tensor);
  if (element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::overflow_error("[YoloV8DecoderNode] Tensor byte size overflow");
  }
  const size_t byte_count = element_count * sizeof(float);
  if (tensor.byte_offset > tensor.data.size() ||
    byte_count > tensor.data.size() - static_cast<size_t>(tensor.byte_offset))
  {
    throw std::invalid_argument("[YoloV8DecoderNode] Tensor data buffer is too small");
  }

  std::vector<float> host_data(element_count);
  auto input_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream);
  const cudaError_t result = cudaMemcpyAsync(
    host_data.data(), input_handle.get_ptr() + tensor.byte_offset, byte_count,
    cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) {
    throw std::runtime_error(
            std::string("[YoloV8DecoderNode] Device-to-host copy failed: ") +
            cudaGetErrorString(result));
  }
  return host_data;
}

}  // namespace

YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10)),
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  num_classes_{declare_parameter<int64_t>("num_classes", 80)}
{
  RCLCPP_DEBUG(get_logger(), "[YoloV8DecoderNode] In YoloV8DecoderNode's constructor");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(input_queue_size_);
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(output_queue_size_);

  // Create CUDA resources
  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("YoloV8DecoderNode");

  // Create subscribers for input and output tensors
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<TensorList>(
    "tensor_sub", input_qos,
    std::bind(&YoloV8DecoderNode::InputCallback, this, std::placeholders::_1),
    sub_options);
  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "detections_output", output_qos,
    pub_options);

  RCLCPP_DEBUG(get_logger(), "[YoloV8DecoderNode] Setup complete");
}

YoloV8DecoderNode::~YoloV8DecoderNode() = default;

void YoloV8DecoderNode::InputCallback(
  const TensorList::ConstSharedPtr msg
)
{
  RCLCPP_DEBUG(get_logger(), "[YoloV8DecoderNode] Received input tensor list");
  const Tensor * input_tensor = isaac_ros_tensor_msgs::FindTensorByName(*msg, tensor_name_);
  if (input_tensor == nullptr) {
    RCLCPP_ERROR(get_logger(), "[YoloV8DecoderNode] Input tensor %s not found",
      tensor_name_.c_str());
    return;
  }

  std::vector<float> results_vector;
  size_t feature_stride;
  size_t detection_stride;
  try {
    ValidateTensor(*input_tensor, num_classes_);
    feature_stride =
      isaac_ros_tensor_msgs::StrideInElements(*input_tensor, kFeatureDimension);
    detection_stride =
      isaac_ros_tensor_msgs::StrideInElements(*input_tensor, kDetectionDimension);
    results_vector = CopyTensorToHost(*input_tensor, *cuda_stream_);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "%s", error.what());
    return;
  }

  const cudaError_t cuda_result = cudaStreamSynchronize(*cuda_stream_);
  if (cuda_result != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[YoloV8DecoderNode] Failed to synchronize CUDA stream: %s",
      cudaGetErrorString(cuda_result));
    return;
  }
  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  const int out_dim = static_cast<int>(input_tensor->shape[kDetectionDimension]);

  for (int i = 0; i < out_dim; i++) {
    const size_t detection_offset = static_cast<size_t>(i) * detection_stride;
    const float x = results_vector.at(detection_offset);
    const float y = results_vector.at(feature_stride + detection_offset);
    const float w = results_vector.at(2 * feature_stride + detection_offset);
    const float h = results_vector.at(3 * feature_stride + detection_offset);

    float x1 = (x - (0.5 * w));
    float y1 = (y - (0.5 * h));
    float width = w;
    float height = h;

    std::vector<float> conf;
    for (int j = 0; j < num_classes_; j++) {
      conf.push_back(
        results_vector.at(
          static_cast<size_t>(kBoxParameters + j) * feature_stride + detection_offset));
    }

    std::vector<float>::iterator ind_max_conf;
    ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
    int max_index = distance(std::begin(conf), ind_max_conf);
    float val_max_conf = *max_element(std::begin(conf), std::end(conf));

    bboxes.push_back(cv::Rect(x1, y1, width, height));
    indices.push_back(i);
    scores.push_back(val_max_conf);
    classes.push_back(max_index);
  }

  RCLCPP_DEBUG(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);

  vision_msgs::msg::Detection2DArray final_detections_arr;

  for (size_t i = 0; i < indices.size(); i++) {
    int ind = indices[i];
    vision_msgs::msg::Detection2D detection;

    geometry_msgs::msg::Pose center;
    geometry_msgs::msg::Point position;
    geometry_msgs::msg::Quaternion orientation;

    // 2D object Bbox
    vision_msgs::msg::BoundingBox2D bbox;
    float w = bboxes[ind].width;
    float h = bboxes[ind].height;
    float x_center = bboxes[ind].x + (0.5 * w);
    float y_center = bboxes[ind].y + (0.5 * h);
    detection.bbox.center.position.x = x_center;
    detection.bbox.center.position.y = y_center;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;

    // Class probabilities
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(classes.at(ind));
    hyp.hypothesis.score = scores.at(ind);
    detection.results.push_back(hyp);

    detection.header = msg->header;

    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header = msg->header;
  pub_->publish(final_detections_arr);
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)
