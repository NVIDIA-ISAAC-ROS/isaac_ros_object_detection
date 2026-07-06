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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
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

uint32_t StrideInFloats(const nvidia::isaac_ros::nitros::NitrosTensor & tensor, int dim_idx)
{
  const auto & strides = tensor.strides();
  if (dim_idx < 0 || static_cast<size_t>(dim_idx) >= strides.size()) {
    throw std::runtime_error("Stride index out of range");
  }
  return static_cast<uint32_t>(strides[static_cast<size_t>(dim_idx)] / sizeof(float));
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
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  nitros_sub_ = create_subscription<nvidia::isaac_ros::nitros::NitrosTensorList>(
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
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr msg)
{
  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] Received input tensor list");

  std::shared_ptr<nvidia::isaac_ros::nitros::NitrosTensor> cov_tensor =
    msg->get_tensor_by_name(cov_tensor_name_);
  std::shared_ptr<nvidia::isaac_ros::nitros::NitrosTensor> bbox_tensor =
    msg->get_tensor_by_name(bbox_tensor_name_);
  if (cov_tensor == nullptr || bbox_tensor == nullptr) {
    RCLCPP_ERROR(
      get_logger(),
      "[DetectNetDecoderNode] Missing tensor(s): cov='%s' bbox='%s'",
      cov_tensor_name_.c_str(), bbox_tensor_name_.c_str());
    return;
  }

  if (cov_tensor->data_type() != nvidia::isaac_ros::nitros::NitrosDataType::kFloat32 ||
    bbox_tensor->data_type() != nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
  {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] Tensors must be float32");
    return;
  }

  const auto cov_shape = cov_tensor->shape();
  const auto bbox_shape = bbox_tensor->shape();
  if (cov_shape.rank() != 4U || bbox_shape.rank() != 4U) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] Expected rank-4 tensors");
    return;
  }

  const std::vector<int32_t> cov_dims = cov_shape.dims();
  const std::vector<int32_t> bbox_dims = bbox_shape.dims();
  const int num_classes = cov_dims[static_cast<size_t>(kTensorClassIdx)];
  const int grid_size_rows = bbox_dims[static_cast<size_t>(kTensorHeightIdx)];
  const int grid_size_cols = bbox_dims[static_cast<size_t>(kTensorWidthIdx)];
  const int num_box_parameters = bbox_dims[static_cast<size_t>(kTensorClassIdx)] / num_classes;
  if (num_box_parameters != kBoundingBoxParams) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] Wrong number of box parameters");
    return;
  }

  const uint32_t bbox_tensor_height_stride = StrideInFloats(*bbox_tensor, kTensorHeightIdx);
  const uint32_t bbox_tensor_width_stride = StrideInFloats(*bbox_tensor, kTensorWidthIdx);
  const uint32_t bbox_tensor_class_stride = StrideInFloats(*bbox_tensor, kTensorClassIdx);
  const uint32_t cov_tensor_height_stride = StrideInFloats(*cov_tensor, kTensorHeightIdx);
  const uint32_t cov_tensor_width_stride = StrideInFloats(*cov_tensor, kTensorWidthIdx);
  const uint32_t cov_tensor_class_stride = StrideInFloats(*cov_tensor, kTensorClassIdx);

  std::vector<float> cov_tensor_arr(cov_tensor->element_count());
  std::vector<float> bbox_tensor_arr(bbox_tensor->element_count());

  // Copy data to CPU for further processing
  const cudaError_t err_cov = cudaMemcpyAsync(
    cov_tensor_arr.data(), cov_tensor->get_read_handle(*cuda_stream_).get_ptr(),
    cov_tensor->tensor_size(), cudaMemcpyDeviceToHost, *cuda_stream_);
  if (err_cov != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] cov memcpy failed: %s",
        cudaGetErrorString(err_cov));
    return;
  }
  const cudaError_t err_bbox = cudaMemcpyAsync(
    bbox_tensor_arr.data(), bbox_tensor->get_read_handle(*cuda_stream_).get_ptr(),
    bbox_tensor->tensor_size(), cudaMemcpyDeviceToHost, *cuda_stream_);
  if (err_bbox != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] bbox memcpy failed: %s",
      cudaGetErrorString(err_bbox));
    return;
  }
  const cudaError_t err_sync = cudaStreamSynchronize(*cuda_stream_);
  if (err_sync != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "[DetectNetDecoderNode] stream sync failed: %s",
      cudaGetErrorString(err_sync));
    return;
  }

  std::vector<NvDsInferObjectDetectionInfo> detection_info_vector;
  for (int row = 0; row < grid_size_rows; ++row) {
    for (int col = 0; col < grid_size_cols; ++col) {
      for (int object_class = 0; object_class < num_classes; ++object_class) {
        const int cov_pos = (row * static_cast<int>(cov_tensor_height_stride)) +
          (col * static_cast<int>(cov_tensor_width_stride)) +
          (object_class * static_cast<int>(cov_tensor_class_stride));
        const float coverage = cov_tensor_arr[static_cast<size_t>(cov_pos)];

        const float grid_center_y = (row + static_cast<float>(bounding_box_offset_)) * kStride;
        const float grid_center_x = (col + static_cast<float>(bounding_box_offset_)) * kStride;

        float bbox[kBoundingBoxParams];
        const int grid_offset =
          (row * static_cast<int>(bbox_tensor_height_stride)) +
          (col * static_cast<int>(bbox_tensor_width_stride));
        for (int bbox_element = 0; bbox_element < num_box_parameters; ++bbox_element) {
          const int pos = grid_offset +
            ((object_class * num_box_parameters + bbox_element) *
            static_cast<int>(bbox_tensor_class_stride));
          bbox[bbox_element] = bbox_tensor_arr[static_cast<size_t>(pos)] *
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
            object_class, label_list_.size());
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

  const std_msgs::msg::Header header = msg->get_header();
  vision_msgs::msg::Detection2DArray out;
  out.header = header;

  for (size_t i = 0; i < num_detections; ++i) {
    const NvDsInferObjectDetectionInfo & d = detection_info_vector[i];
    vision_msgs::msg::Detection2D det;
    det.header = header;
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
