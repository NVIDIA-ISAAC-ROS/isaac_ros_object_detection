// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "detectnet_decoder.hpp"
#include "detection2_d_array_message.hpp"

#include <string>
#include <climits>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/timestamp.hpp"
#include "cuda.h"
#include "cuda_runtime.h"


namespace nvidia
{
namespace isaac_ros
{
namespace
{
// constant parameters for DetectNetv2
constexpr int kTensorHeightIdx = 2;
constexpr int kTensorWidthIdx = 3;
constexpr int kTensorClassIdx = 1;
constexpr int kBoundingBoxParams = 4;
// Grid box length in pixels. DetectNetv2 currently only supports square grid boxes of size 16.
constexpr int kStride = 16;
// dbscan varients
constexpr uint32_t kDbscanCluster = 1;
constexpr uint32_t kDbscanClusterHybrid = 2;

NvDsInferObjectDetectionInfo GetNewDetectionInfo(
  unsigned int classId,
  float left,
  float top,
  float width,
  float height,
  float detectionConfidence)
{
  NvDsInferObjectDetectionInfo detection_info;

  detection_info.classId = classId;
  detection_info.left = left;
  detection_info.top = top;
  detection_info.width = width;
  detection_info.height = height;
  detection_info.detectionConfidence = detectionConfidence;

  return detection_info;
}

void FillMessage(
  Detection2DParts &message_parts,
  const std::vector<NvDsInferObjectDetectionInfo> &detection_info_vector,
  gxf::Handle<nvidia::gxf::Timestamp> tensorlist_timestamp,
  size_t num_detections, const std::vector<std::string> &label_list)
{
  for (uint32_t i = 0; i < num_detections; i++) {
    NvDsInferObjectDetectionInfo detection_info = detection_info_vector[i];
    Detection2D temp_detection;
    temp_detection.center_x = static_cast<int>(detection_info.left + (detection_info.width / 2));
    temp_detection.center_y = static_cast<int>(detection_info.top + (detection_info.height / 2));
    temp_detection.size_x = static_cast<int>(detection_info.width);
    temp_detection.size_y = static_cast<int>(detection_info.height);
    Hypothesis temp_hypothesis;
    // populate string class_id from label_list parameter
    // out of range check is done at line 330
    temp_hypothesis.class_id = label_list[detection_info.classId];
    temp_hypothesis.score = detection_info.detectionConfidence;
    temp_detection.results.push_back(temp_hypothesis);
    (message_parts.detection2_d_array)->push_back(temp_detection);
  }
  *(message_parts.timestamp) = *tensorlist_timestamp;
}
} // anonymous namespace


gxf_result_t DetectnetDecoder::registerInterface(gxf::Registrar * registrar) noexcept
{
  gxf::Expected<void> result;

  result &= registrar->parameter(
    tensorlist_receiver_, "tensorlist_receiver", "Tensorlist Input",
    "The detections as a tensorlist");

  result &= registrar->parameter(
    detections_transmitter_, "detections_transmitter", "Detections output",
    "The filtered detections output");

  result &= registrar->parameter(
    label_list_, "label_list", "List of network labels",
    "List of labels corresponding to the int labels received from the tensors", {"person", "bag",
      "face"});

  result &= registrar->parameter(
    enable_confidence_threshold_, "enable_confidence_threshold", "Enable Confidence Threshold",
    "Flag to enable minimum confidence thresholding", true);

  result &= registrar->parameter(
    enable_bbox_area_threshold_, "enable_bbox_area_threshold", "Enable Bbox Area Threshold",
    "Flag to enable minimum bounding box area thresholding", true);

  result &= registrar->parameter(
    enable_dbscan_clustering_, "enable_dbscan_clustering", "Enable Dbscan Clustering",
    "Flag to enable Dbscan clustering", true);

  result &= registrar->parameter(
    confidence_threshold_, "confidence_threshold", "Confidence Threshold",
    "The min value of confidence used to threshold detections before clustering", 0.6);

  result &= registrar->parameter(
    min_bbox_area_, "min_bbox_area", "Min Bbox Area",
    "The min value of bouding box area used to threshold detections before clustering", 100.0);

  result &= registrar->parameter(
    dbscan_confidence_threshold_, "dbscan_confidence_threshold", "Dbscan Confidence Threshold",
    "Minimum score in a cluster for the cluster to be considered an object \
     during grouping. Different clustering may cause the algorithm \
     to use different scores.",
    0.6);

  result &= registrar->parameter(
    dbscan_eps_, "dbscan_eps", "Dbscan Epsilon",
    "Holds the epsilon to control merging of overlapping boxes. \
    Refer to OpenCV groupRectangles and DBSCAN documentation for more information on epsilon. ",
    0.01);

  result &= registrar->parameter(
    dbscan_min_boxes_, "dbscan_min_boxes", "Dbscan Minimum Boxes",
    "Holds the minimum number of boxes in a cluster to be considered \
     an object during grouping using DBSCAN",
    1);

  result &= registrar->parameter(
    dbscan_enable_athr_filter_, "dbscan_enable_athr_filter", "Dbscan Enable Athr Filter",
    "true enables the area-to-hit ratio (ATHR) filter. \
     The ATHR is calculated as: ATHR = sqrt(clusterArea) / nObjectsInCluster.",
    0);

  result &= registrar->parameter(
    dbscan_threshold_athr_, "dbscan_threshold_athr", "Dbscan Threshold Athr",
    "Holds the area-to-hit ratio threshold", 0.0);

  result &= registrar->parameter(
    dbscan_clustering_algorithm_, "dbscan_clustering_algorithm", "Dbscan Clustering Algorithm",
    "The clustering algorithm used. 1 => NvDsInferDBScanCluster 2 => NvDsInferDBScanClusterHybrid",
    1);

  result &= registrar->parameter(
    bounding_box_scale_, "bounding_box_scale", "Bounding Box Scale",
    "The scale parameter, which should match the training configuration", 35.0);

  result &= registrar->parameter(
    bounding_box_offset_, "bounding_box_offset", "Bounding Box Offset",
    "Bounding box offset for both X and Y dimensions", 0.5);

  return gxf::ToResultCode(result);
}

gxf_result_t DetectnetDecoder::start() noexcept
{
  params_.eps = dbscan_eps_;
  params_.minBoxes = dbscan_min_boxes_ > 0 ? dbscan_min_boxes_ : 0;
  params_.enableATHRFilter = dbscan_enable_athr_filter_;
  params_.thresholdATHR = dbscan_threshold_athr_;
  params_.minScore = dbscan_confidence_threshold_;

  if (dbscan_clustering_algorithm_ != kDbscanCluster &&
    dbscan_clustering_algorithm_ != kDbscanClusterHybrid)
  {
    GXF_LOG_ERROR(
      "Invalid value for dbscan_clustering_algorithm: %i",
      dbscan_clustering_algorithm_.get());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t DetectnetDecoder::tick() noexcept
{

  gxf::Expected<void> result;

  // Receive disparity image and left/right camera info
  auto maybe_tensorlist_message = tensorlist_receiver_->receive();
  if (!maybe_tensorlist_message) {
    GXF_LOG_ERROR("Failed to receive tensorlist message");
    return gxf::ToResultCode(maybe_tensorlist_message);
  }

  // Get timestamp from message
  auto maybe_tensorlist_timestamp = maybe_tensorlist_message->get<gxf::Timestamp>(
    "timestamp");
  if (!maybe_tensorlist_timestamp) {
    GXF_LOG_ERROR("Failed to get a timestamp from tensorlist message");
    return gxf::ToResultCode(maybe_tensorlist_timestamp);
  }
  gxf::Handle<nvidia::gxf::Timestamp> tensorlist_timestamp = maybe_tensorlist_timestamp.value();

  // Extract all tensors from message
  auto maybe_gxf_tensors = maybe_tensorlist_message->findAll<nvidia::gxf::Tensor>();
  if (!maybe_gxf_tensors) {
    GXF_LOG_ERROR("Failed find all GXF tensors: %s", GxfResultStr(maybe_gxf_tensors.error()));
    return gxf::ToResultCode(maybe_gxf_tensors);
  }
  auto gxf_tensors = maybe_gxf_tensors.value();
  if (gxf_tensors.size() != 2) {
    GXF_LOG_ERROR("Received tensorlists does not contain exactly 2 tensors");
    return GXF_FAILURE;
  }

  auto cov_tensor = gxf_tensors[0]->get();
  auto bbox_tensor = gxf_tensors[1]->get();


  // ensure that data is stored on GPU
  if (cov_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("[DetectNet Decoder] Tensor MemoryStorageType should to be kDevice");
    return GXF_FAILURE;
  }
  if (bbox_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("[DetectNet Decoder] Tensor MemoryStorageType should to be kDevice");
    return GXF_FAILURE;
  }

  // ensure that type of data in tensor is kFloat32
  nvidia::gxf::PrimitiveType cov_tensor_data_type =
    static_cast<nvidia::gxf::PrimitiveType>(cov_tensor->element_type());
  nvidia::gxf::PrimitiveType bbox_tensor_data_type =
    static_cast<nvidia::gxf::PrimitiveType>(bbox_tensor->element_type());
  if (cov_tensor_data_type != nvidia::gxf::PrimitiveType::kFloat32) {
    GXF_LOG_ERROR("[DetectNet Decoder] Tensor PrimitiveType should to be kFloat32");
    return GXF_FAILURE;
  }
  if (bbox_tensor_data_type != nvidia::gxf::PrimitiveType::kFloat32) {
    GXF_LOG_ERROR("[DetectNet Decoder] Tensor PrimitiveType should to be kFloat32");
    return GXF_FAILURE;
  }

  // ensure that the tensors have a rank of 4
  if (cov_tensor->shape().rank() != 4 || bbox_tensor->shape().rank() != 4) {
    GXF_LOG_ERROR("[DetectNet Decoder] Tensor Rank should to be 4");
    return GXF_FAILURE;
  }

  // TODO(ashwinvk): Do not copy data to host and perform decoding using cuda
  // copy memory to host
  std::unique_ptr<float[]> cov_tensor_arr(new float[cov_tensor->element_count()]);
  const cudaError_t cuda_error_cov_tensor = cudaMemcpy(
    cov_tensor_arr.get(), cov_tensor->pointer(),
    cov_tensor->size(), cudaMemcpyDeviceToHost);
  if (cuda_error_cov_tensor != cudaSuccess) {
    GXF_LOG_ERROR("Error while copying kernel: %s", cudaGetErrorString(cuda_error_cov_tensor));
    return GXF_FAILURE;
  }

  float bbox_tensor_arr[bbox_tensor->size() / sizeof(float)]; // since data in tensor is kFloat32
  const cudaError_t cuda_error_bbox_tensor = cudaMemcpy(
    &bbox_tensor_arr, bbox_tensor->pointer(),
    bbox_tensor->size(), cudaMemcpyDeviceToHost);
  if (cuda_error_bbox_tensor != cudaSuccess) {
    GXF_LOG_ERROR("Error while copying kernel: %s", cudaGetErrorString(cuda_error_bbox_tensor));
    return GXF_FAILURE;
  }

  const uint32_t bbox_tensor_height_stride = bbox_tensor->stride(kTensorHeightIdx) /
    sizeof(float);
  const uint32_t bbox_tensor_width_stride = bbox_tensor->stride(kTensorWidthIdx) / sizeof(float);
  const uint32_t bbox_tensor_class_stride = bbox_tensor->stride(kTensorClassIdx) / sizeof(float);

  // Reinterpret the strides and data as above, this time for the coverage tensor
  const uint32_t cov_tensor_height_stride = cov_tensor->stride(kTensorHeightIdx) / sizeof(float);
  const uint32_t cov_tensor_width_stride = cov_tensor->stride(kTensorWidthIdx) / sizeof(float);
  const uint32_t cov_tensor_class_stride = cov_tensor->stride(kTensorClassIdx) / sizeof(float);

  // Get grid size based on the size of bbox_tensor
  const int num_classes = cov_tensor->shape().dimension(kTensorClassIdx);
  const int grid_size_rows = bbox_tensor->shape().dimension(kTensorHeightIdx);
  const int grid_size_cols = bbox_tensor->shape().dimension(kTensorWidthIdx);

  // Validate number of bounding box parameters
  const int num_box_parameters = bbox_tensor->shape().dimension(kTensorClassIdx) / num_classes;
  if (num_box_parameters != kBoundingBoxParams) {
    GXF_LOG_ERROR("[DetectNet Decoder] Received wrong number of box parameters");
    return GXF_FAILURE;
  }

  // detections data structure for dbscan library
  std::vector<NvDsInferObjectDetectionInfo> detection_info_vector;
  // Go through every grid box and extract all bboxes for the image
  for (int row = 0; row < grid_size_rows; row++) {
    for (int col = 0; col < grid_size_cols; col++) {
      for (int object_class = 0; object_class < num_classes; object_class++) {
        // position of the current bounding box coverage value in the reinterpreted cov tensor
        int cov_pos = (row * cov_tensor_height_stride) + (col * cov_tensor_width_stride) +
          (object_class * cov_tensor_class_stride);
        float coverage = cov_tensor_arr[cov_pos];

        // Center of the grid in pixels
        float grid_center_y = (row + bounding_box_offset_ ) * kStride;
        float grid_center_x = (col + bounding_box_offset_ ) * kStride;

        // Get each element of the bounding box
        float bbox[kBoundingBoxParams];
        int grid_offset = (row * bbox_tensor_height_stride) + (col * bbox_tensor_width_stride);
        for (int bbox_element = 0; bbox_element < num_box_parameters; bbox_element++) {
          int pos = grid_offset + ((object_class + bbox_element) * bbox_tensor_class_stride);
          bbox[bbox_element] = bbox_tensor_arr[pos] * bounding_box_scale_;
        }

        float size_x = bbox[0] + bbox[2];
        float size_y = bbox[1] + bbox[3];

        // Filter by box area.
        double bbox_area = size_x * size_y;
        if (enable_bbox_area_threshold_ && bbox_area < min_bbox_area_) {
          continue;
        }

        if (enable_confidence_threshold_ && coverage < confidence_threshold_) {
          continue;
        }

        // check if object_class is out of range for label_list_
        if (static_cast<size_t>(object_class) >= label_list_.get().size()) {
          GXF_LOG_ERROR(
            "[DetectNet Decoder] object_class %i is out of range for provided label_list_ of size %lu", object_class,
            label_list_.get().size());
          return GXF_FAILURE;
        }

        // Bounding box is in the form of (x0, y0, x1, y1) in grid coordinates
        // Center relative to is grid found averaging the averaging the dimensions and offset
        // by grid center position to convert to image coordinates
        NvDsInferObjectDetectionInfo detection_info = GetNewDetectionInfo(
          static_cast<unsigned int>(object_class),
          grid_center_x - bbox[0], grid_center_y - bbox[1], size_x, size_y, coverage);
        detection_info_vector.push_back(detection_info);
      }
    }
  }

  size_t num_detections = detection_info_vector.size();
  if (enable_dbscan_clustering_) {
    NvDsInferObjectDetectionInfo * detection_info_pointer = &detection_info_vector[0];
    NvDsInferDBScanHandle dbscan_hdl = NvDsInferDBScanCreate();
    if (dbscan_clustering_algorithm_ == kDbscanCluster) {
      NvDsInferDBScanCluster(dbscan_hdl, &params_, detection_info_pointer, &num_detections);
    } else if (dbscan_clustering_algorithm_ == kDbscanClusterHybrid) {
      NvDsInferDBScanClusterHybrid(dbscan_hdl, &params_, detection_info_pointer, &num_detections);
    } else {
      GXF_LOG_ERROR(
        "Invalid value for dbscan_clustering_algorithm: %i",
        dbscan_clustering_algorithm_.get());
      return GXF_FAILURE;
    }
    NvDsInferDBScanDestroy(dbscan_hdl);
  }

  return gxf::ToResultCode(
    // We first create the default-initialized message parts struct.
    CreateDetection2DList(context())
    // We now use the map method to fill and publish the parts struct.
    .map(
      [&](Detection2DParts message_parts) {
        FillMessage(
          message_parts, detection_info_vector, tensorlist_timestamp,
          num_detections, label_list_);
        return detections_transmitter_->publish(message_parts.message);
      }));

}
}  // namespace isaac_ros
}  // namespace nvidia
