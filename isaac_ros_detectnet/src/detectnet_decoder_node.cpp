// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_detectnet/detectnet_decoder_node.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_tensor_msgs/tensor_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

#include "deepstream_utils/nvdsinferutils/include/nvdsinfer_dbscan.h"

namespace nvidia
{
namespace isaac_ros
{
namespace detectnet
{
namespace
{
constexpr int kTensorHeightIdx = 2;
constexpr int kTensorWidthIdx = 3;
constexpr int kTensorClassIdx = 1;
constexpr int kBoundingBoxParams = 4;
constexpr int kStride = 16;
constexpr uint32_t kDbscanCluster = 1;
constexpr uint32_t kDbscanClusterHybrid = 2;

NvDsInferObjectDetectionInfo CreateDetectionInfo(
  unsigned int class_id,
  float left,
  float top,
  float width,
  float height,
  float detection_confidence)
{
  NvDsInferObjectDetectionInfo detection_info{};
  detection_info.classId = class_id;
  detection_info.left = left;
  detection_info.top = top;
  detection_info.width = width;
  detection_info.height = height;
  detection_info.detectionConfidence = detection_confidence;
  return detection_info;
}

void ValidateFloatTensor(const Tensor & tensor, const std::string & name)
{
  constexpr uint8_t kDLPackFloat = 2;
  if (tensor.dtype_code != kDLPackFloat || tensor.dtype_bits != 32 ||
    tensor.dtype_lanes != 1)
  {
    throw std::invalid_argument(
            "[DetectNetDecoderNode] Tensor '" + name + "' must be float32");
  }
  if (tensor.shape.size() != 4) {
    throw std::invalid_argument(
            "[DetectNetDecoderNode] Tensor '" + name + "' must have rank 4");
  }
  for (const int64_t dim : tensor.shape) {
    if (dim <= 0) {
      throw std::invalid_argument(
              "[DetectNetDecoderNode] Tensor '" + name + "' dimensions must be positive");
    }
  }
}

std::vector<float> CopyTensorToHost(
  const Tensor & tensor, cudaStream_t stream)
{
  const size_t element_count = isaac_ros_tensor_msgs::RequiredStorageElements(tensor);
  if (element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::overflow_error("[DetectNetDecoderNode] Tensor byte size overflow");
  }
  const size_t byte_count = element_count * sizeof(float);
  if (tensor.byte_offset > tensor.data.size() ||
    byte_count > tensor.data.size() - static_cast<size_t>(tensor.byte_offset))
  {
    throw std::invalid_argument("[DetectNetDecoderNode] Tensor data buffer is too small");
  }

  std::vector<float> host_data(element_count);
  auto input_handle = cuda_buffer_backend::from_input_buffer(tensor.data, stream);
  const cudaError_t err = cudaMemcpyAsync(
    host_data.data(), input_handle.get_ptr() + tensor.byte_offset,
    byte_count, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("[DetectNetDecoderNode] Device-to-host copy failed: ") +
            cudaGetErrorString(err));
  }
  return host_data;
}

}  // namespace

DetectNetDecoderNode::DetectNetDecoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("detectnet_decoder_node", options),
  label_list_(declare_parameter<std::vector<std::string>>("label_list", {"person", "bag", "face"})),
  enable_confidence_threshold_(declare_parameter<bool>("enable_confidence_threshold", true)),
  enable_bbox_area_threshold_(declare_parameter<bool>("enable_bbox_area_threshold", true)),
  enable_dbscan_clustering_(declare_parameter<bool>("enable_dbscan_clustering", true)),
  confidence_threshold_(declare_parameter<double>("confidence_threshold", 0.6)),
  min_bbox_area_(declare_parameter<double>("min_bbox_area", 100.0)),
  dbscan_confidence_threshold_(declare_parameter<double>("dbscan_confidence_threshold", 0.6)),
  dbscan_eps_(declare_parameter<double>("dbscan_eps", 1.0)),
  dbscan_min_boxes_(declare_parameter<int>("dbscan_min_boxes", 1)),
  dbscan_enable_athr_filter_(declare_parameter<int>("dbscan_enable_athr_filter", 0)),
  dbscan_threshold_athr_(declare_parameter<double>("dbscan_threshold_athr", 0.0)),
  dbscan_clustering_algorithm_(declare_parameter<int>("dbscan_clustering_algorithm", 1)),
  bounding_box_scale_(declare_parameter<double>("bounding_box_scale", 35.0)),
  bounding_box_offset_(declare_parameter<double>("bounding_box_offset", 0.0)),
  cov_tensor_name_(declare_parameter<std::string>("cov_tensor_name", "output_cov")),
  bbox_tensor_name_(declare_parameter<std::string>("bbox_tensor_name", "output_bbox")),
  input_queue_size_(declare_parameter<int16_t>("input_queue_size", 10)),
  output_queue_size_(declare_parameter<int16_t>("output_queue_size", 10))
{
  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] Constructor");

  const rclcpp::QoS input_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "input_qos").keep_last(static_cast<size_t>(input_queue_size_));
  const rclcpp::QoS output_qos = ::isaac_ros::common::AddQosParameter(
    *this, "DEFAULT", "output_qos").keep_last(static_cast<size_t>(output_queue_size_));

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream("DetectNetDecoderNode");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_sub_ = create_subscription<TensorList>(
      "tensor_sub", input_qos,
      std::bind(&DetectNetDecoderNode::InputCallback, this, std::placeholders::_1),
      sub_options);
  pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "detectnet/detections", output_qos,
    pub_options);

  // initialize dbscan parameters
  params_.eps = dbscan_eps_;
  params_.minBoxes = dbscan_min_boxes_ > 0 ? dbscan_min_boxes_ : 0;
  params_.enableATHRFilter = dbscan_enable_athr_filter_;
  params_.thresholdATHR = dbscan_threshold_athr_;
  params_.minScore = dbscan_confidence_threshold_;

  if (dbscan_clustering_algorithm_ != kDbscanCluster &&
    dbscan_clustering_algorithm_ != kDbscanClusterHybrid)
  {
    RCLCPP_ERROR(get_logger(),
      "[DetectNetDecoderNode] Invalid value for dbscan_clustering_algorithm: %i",
      dbscan_clustering_algorithm_);
    throw std::invalid_argument(
      "[DetectNetDecoderNode] Invalid value for dbscan_clustering_algorithm");
  }

  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] Setup complete");
}

DetectNetDecoderNode::~DetectNetDecoderNode() {}

void DetectNetDecoderNode::InputCallback(
  const TensorList::ConstSharedPtr msg)
{
  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] Received input tensor list");

  const Tensor * cov_tensor = isaac_ros_tensor_msgs::FindTensorByName(*msg, cov_tensor_name_);
  const Tensor * bbox_tensor = isaac_ros_tensor_msgs::FindTensorByName(*msg, bbox_tensor_name_);
  if (cov_tensor == nullptr || bbox_tensor == nullptr) {
    RCLCPP_ERROR(
      get_logger(),
      "[DetectNetDecoderNode] Missing tensor(s): cov='%s' bbox='%s'",
      cov_tensor_name_.c_str(), bbox_tensor_name_.c_str());
    return;
  }

  try {
    ValidateFloatTensor(*cov_tensor, cov_tensor_name_);
    ValidateFloatTensor(*bbox_tensor, bbox_tensor_name_);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_logger(), "%s", error.what());
    return;
  }

  const auto & cov_dims = cov_tensor->shape;
  const auto & bbox_dims = bbox_tensor->shape;
  const int64_t num_classes = cov_dims[static_cast<size_t>(kTensorClassIdx)];
  const int64_t grid_size_rows = bbox_dims[static_cast<size_t>(kTensorHeightIdx)];
  const int64_t grid_size_cols = bbox_dims[static_cast<size_t>(kTensorWidthIdx)];
  if (cov_dims[static_cast<size_t>(kTensorHeightIdx)] != grid_size_rows ||
    cov_dims[static_cast<size_t>(kTensorWidthIdx)] != grid_size_cols)
  {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] Coverage and bbox grid sizes differ");
    return;
  }
  if (num_classes <= 0 ||
    bbox_dims[static_cast<size_t>(kTensorClassIdx)] % num_classes != 0)
  {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] Invalid class dimensions");
    return;
  }
  const int64_t num_box_parameters =
    bbox_dims[static_cast<size_t>(kTensorClassIdx)] / num_classes;
  if (num_box_parameters != kBoundingBoxParams) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] Wrong number of box parameters");
    return;
  }

  size_t bbox_tensor_height_stride;
  size_t bbox_tensor_width_stride;
  size_t bbox_tensor_class_stride;
  size_t cov_tensor_height_stride;
  size_t cov_tensor_width_stride;
  size_t cov_tensor_class_stride;
  std::vector<float> cov_tensor_arr;
  std::vector<float> bbox_tensor_arr;
  try {
    bbox_tensor_height_stride =
      isaac_ros_tensor_msgs::StrideInElements(*bbox_tensor, kTensorHeightIdx);
    bbox_tensor_width_stride =
      isaac_ros_tensor_msgs::StrideInElements(*bbox_tensor, kTensorWidthIdx);
    bbox_tensor_class_stride =
      isaac_ros_tensor_msgs::StrideInElements(*bbox_tensor, kTensorClassIdx);
    cov_tensor_height_stride =
      isaac_ros_tensor_msgs::StrideInElements(*cov_tensor, kTensorHeightIdx);
    cov_tensor_width_stride =
      isaac_ros_tensor_msgs::StrideInElements(*cov_tensor, kTensorWidthIdx);
    cov_tensor_class_stride =
      isaac_ros_tensor_msgs::StrideInElements(*cov_tensor, kTensorClassIdx);
    cov_tensor_arr = CopyTensorToHost(*cov_tensor, *cuda_stream_);
    bbox_tensor_arr = CopyTensorToHost(*bbox_tensor, *cuda_stream_);
  } catch (const std::exception & error) {
    const cudaError_t sync_result = cudaStreamSynchronize(*cuda_stream_);
    if (sync_result != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] stream sync failed: %s",
        cudaGetErrorString(sync_result));
    }
    RCLCPP_ERROR(get_logger(), "%s", error.what());
    return;
  }

  const cudaError_t err_sync = cudaStreamSynchronize(*cuda_stream_);
  if (err_sync != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] stream sync failed: %s",
      cudaGetErrorString(err_sync));
    return;
  }

  std::vector<NvDsInferObjectDetectionInfo> detection_info_vector;
  for (int64_t row = 0; row < grid_size_rows; ++row) {
    for (int64_t col = 0; col < grid_size_cols; ++col) {
      for (int64_t object_class = 0; object_class < num_classes; ++object_class) {
        const size_t cov_pos =
          static_cast<size_t>(row) * cov_tensor_height_stride +
          static_cast<size_t>(col) * cov_tensor_width_stride +
          static_cast<size_t>(object_class) * cov_tensor_class_stride;
        const float coverage = cov_tensor_arr.at(cov_pos);

        const float grid_center_y = (row + static_cast<float>(bounding_box_offset_)) * kStride;
        const float grid_center_x = (col + static_cast<float>(bounding_box_offset_)) * kStride;

        float bbox[kBoundingBoxParams];
        const size_t grid_offset =
          static_cast<size_t>(row) * bbox_tensor_height_stride +
          static_cast<size_t>(col) * bbox_tensor_width_stride;
        for (int64_t bbox_element = 0; bbox_element < num_box_parameters; ++bbox_element) {
          const size_t pos = grid_offset +
            static_cast<size_t>(object_class * num_box_parameters + bbox_element) *
            bbox_tensor_class_stride;
          bbox[static_cast<size_t>(bbox_element)] = bbox_tensor_arr.at(pos) *
            static_cast<float>(bounding_box_scale_);
        }

        const float size_x = bbox[0] + bbox[2];
        const float size_y = bbox[1] + bbox[3];
        const double bbox_area = static_cast<double>(size_x) * static_cast<double>(size_y);
        if (enable_bbox_area_threshold_ && bbox_area < min_bbox_area_) {
          continue;
        }
        if (enable_confidence_threshold_ && coverage < static_cast<float>(confidence_threshold_)) {
          continue;
        }
        if (static_cast<size_t>(object_class) >= label_list_.size()) {
          RCLCPP_ERROR(
            get_logger(),
            "[DetectNetDecoderNode] object_class %i out of range for label_list size %zu",
            static_cast<int>(object_class), label_list_.size());
          return;
        }

        detection_info_vector.push_back(CreateDetectionInfo(
            static_cast<unsigned int>(object_class),
            grid_center_x - bbox[0],
            grid_center_y - bbox[1],
            size_x,
            size_y,
            coverage));
      }
    }
  }

  size_t num_detections = detection_info_vector.size();
  if (enable_dbscan_clustering_ && num_detections > 0U) {
    NvDsInferObjectDetectionInfo * const detection_info_pointer = detection_info_vector.data();
    NvDsInferDBScanHandle const dbscan_hdl = NvDsInferDBScanCreate();
    if (dbscan_clustering_algorithm_ == static_cast<int>(kDbscanCluster)) {
      NvDsInferDBScanCluster(
        dbscan_hdl, &params_, detection_info_pointer,
        &num_detections);
    } else {
      NvDsInferDBScanClusterHybrid(
        dbscan_hdl, &params_, detection_info_pointer,
        &num_detections);
    }
    NvDsInferDBScanDestroy(dbscan_hdl);
  }

  vision_msgs::msg::Detection2DArray out;
  out.header = msg->header;

  for (size_t i = 0; i < num_detections; ++i) {
    const NvDsInferObjectDetectionInfo & d = detection_info_vector[i];
    vision_msgs::msg::Detection2D det;
    det.header = msg->header;
    det.bbox.center.position.x = d.left + d.width / 2.F;
    det.bbox.center.position.y = d.top + d.height / 2.F;
    det.bbox.size_x = d.width;
    det.bbox.size_y = d.height;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = label_list_.at(d.classId);
    hyp.hypothesis.score = d.detectionConfidence;
    det.results.push_back(hyp);
    out.detections.push_back(det);
  }

  pub_->publish(out);
}

}  // namespace detectnet
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::detectnet::DetectNetDecoderNode)
