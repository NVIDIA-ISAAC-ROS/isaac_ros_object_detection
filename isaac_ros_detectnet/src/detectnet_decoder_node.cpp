/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "isaac_ros_detectnet/detectnet_decoder_node.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "nvdsinferutils/dbscan/nvdsinfer_dbscan.hpp"
#pragma GCC diagnostic pop
#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"

namespace
{
const int32_t kFloat32 = 9;
const int32_t kMinBoxArea = 100;
const int kBoundingBoxParams = 4;

// Grid box length in pixels. DetectNetv2 currently only supports square grid boxes of size 16.
const int kStride = 16;
}   // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace detectnet
{
struct DetectNetDecoderNode::DetectNetDecoderImpl
{
  const int32_t kTensorHeightIdx = 2;
  const int32_t kTensorWidthIdx = 3;
  const int32_t kTensorClassIdx = 1;
  std::vector<std::string> label_names_;
  float coverage_threshold_;
  float bounding_box_scale_;
  float bounding_box_offset_;
  NvDsInferDBScanClusteringParams params_;
  int clustering_algorithm_;

  DetectNetDecoderImpl(
    const std::vector<std::string> & label_names,
    const float & coverage_threshold,
    const float & bounding_box_scale,
    const float & bounding_box_offset,
    const float & eps,
    const int & min_boxes,
    const int & enable_athr_filter,
    const float & threshold_athr,
    const int & clustering_algorithm)
  {
    label_names_ = label_names;
    coverage_threshold_ = coverage_threshold;
    bounding_box_scale_ = bounding_box_scale;
    bounding_box_offset_ = bounding_box_offset;

    params_.eps = eps;
    params_.minBoxes = min_boxes > 0 ? min_boxes : 0;
    params_.enableATHRFilter = enable_athr_filter;
    params_.thresholdATHR = threshold_athr;
    params_.minScore = coverage_threshold;

    clustering_algorithm_ = clustering_algorithm;
  }

  void OnCallback(
    vision_msgs::msg::Detection2DArray & detections_msg,
    const isaac_ros_tensor_list_interfaces::msg::Tensor & bbox_tensor,
    const isaac_ros_tensor_list_interfaces::msg::Tensor & cov_tensor,
    const std_msgs::msg::Header & tensor_header,
    const rclcpp::Logger & logger)
  {
    ConvertTensorToDetectionsArray(
      detections_msg,
      bbox_tensor,
      cov_tensor,
      tensor_header,
      logger);
  }

  void ConvertTensorToDetectionsArray(
    vision_msgs::msg::Detection2DArray & detections_msg,
    const isaac_ros_tensor_list_interfaces::msg::Tensor & bbox_tensor,
    const isaac_ros_tensor_list_interfaces::msg::Tensor & cov_tensor,
    const std_msgs::msg::Header & tensor_header,
    const rclcpp::Logger & logger)
  {
    if (bbox_tensor.data_type == kFloat32) {
      DecodeDetections(detections_msg, bbox_tensor, cov_tensor, tensor_header, logger);
    } else {
      throw std::runtime_error("Received invalid Tensor data! Expected float32!");
    }
  }

  void DecodeDetections(
    vision_msgs::msg::Detection2DArray & detections_msg,
    const isaac_ros_tensor_list_interfaces::msg::Tensor & bbox_tensor,
    const isaac_ros_tensor_list_interfaces::msg::Tensor & cov_tensor,
    const std_msgs::msg::Header & tensor_header,
    const rclcpp::Logger & logger)
  {
    // Reinterpret the strides (which are in bytes) + data as the relevant data type
    const float * bbox_tensor_data = reinterpret_cast<const float *>(bbox_tensor.data.data());
    const uint32_t bbox_tensor_height_stride = bbox_tensor.strides[kTensorHeightIdx] /
      sizeof(float);
    const uint32_t bbox_tensor_width_stride = bbox_tensor.strides[kTensorWidthIdx] / sizeof(float);
    const uint32_t bbox_tensor_class_stride = bbox_tensor.strides[kTensorClassIdx] / sizeof(float);

    // Reinterpret the strides and data as above, this time for the coverage tensor
    const float * cov_tensor_data = reinterpret_cast<const float *>(cov_tensor.data.data());
    const uint32_t cov_tensor_height_stride = cov_tensor.strides[kTensorHeightIdx] / sizeof(float);
    const uint32_t cov_tensor_width_stride = cov_tensor.strides[kTensorWidthIdx] / sizeof(float);
    const uint32_t cov_tensor_class_stride = cov_tensor.strides[kTensorClassIdx] / sizeof(float);

    // Get grid size based on the size of bbox_tensor
    const int num_classes = cov_tensor.shape.dims[kTensorClassIdx];
    const int grid_size_rows = bbox_tensor.shape.dims[kTensorHeightIdx];
    const int grid_size_cols = bbox_tensor.shape.dims[kTensorWidthIdx];

    // Validate number of bounding box parameters
    const int num_box_parameters = bbox_tensor.shape.dims[kTensorClassIdx] / num_classes;
    if (num_box_parameters != kBoundingBoxParams) {
      RCLCPP_WARN(logger, "Received wrong number of box parameters");
      return;
    }

    std::vector<NvDsInferObjectDetectionInfo> detection_info_list;
    // Go through every grid box and extract all bboxes for the image
    for (int row = 0; row < grid_size_rows; row++) {
      for (int col = 0; col < grid_size_cols; col++) {
        for (int object_class = 0; object_class < num_classes; object_class++) {
          // position of the current bounding box coverage value in the reinterpreted cov tensor
          int cov_pos = (row * cov_tensor_height_stride) + (col * cov_tensor_width_stride) +
            (object_class * cov_tensor_class_stride);
          float coverage = cov_tensor_data[cov_pos];

          // Center of the grid in pixels
          float grid_center_y = (row + bounding_box_offset_ ) * kStride;
          float grid_center_x = (col + bounding_box_offset_ ) * kStride;

          // Get each element of the bounding box
          float bbox[kBoundingBoxParams];
          int grid_offset = (row * bbox_tensor_height_stride) + (col * bbox_tensor_width_stride);
          for (int bbox_element = 0; bbox_element < num_box_parameters; bbox_element++) {
            int pos = grid_offset + ((object_class + bbox_element) * bbox_tensor_class_stride);
            bbox[bbox_element] = bbox_tensor_data[pos] * bounding_box_scale_;
          }

          float size_x = bbox[0] + bbox[2];
          float size_y = bbox[1] + bbox[3];

          // Filter by box area.
          double bbox_area = size_x * size_y;
          if (bbox_area < kMinBoxArea) {
            continue;
          }

          // Bounding box is in the form of (x0, y0, x1, y1) in grid coordinates
          // Center relative to is grid found averaging the averaging the dimensions and offset
          // by grid center position to convert to image coordinates
          NvDsInferObjectDetectionInfo detection_info = GetNewDetectionInfo(
            static_cast<unsigned int>(object_class),
            grid_center_x - bbox[0], grid_center_y - bbox[1], size_x, size_y, coverage);
          detection_info_list.push_back(detection_info);
        }
      }
    }

    NvDsInferObjectDetectionInfo * detetction_info_pointer = &detection_info_list[0];
    size_t num_objs = detection_info_list.size();
    NvDsInferDBScanHandle dbscan_hdl = NvDsInferDBScanCreate();
    if (clustering_algorithm_ == 1) {
      NvDsInferDBScanCluster(dbscan_hdl, &params_, detetction_info_pointer, &num_objs);
    } else if (clustering_algorithm_ == 2) {
      NvDsInferDBScanClusterHybrid(dbscan_hdl, &params_, detetction_info_pointer, &num_objs);
    }
    NvDsInferDBScanDestroy(dbscan_hdl);

    std::vector<vision_msgs::msg::Detection2D> detections_list;
    for (size_t i = 0; i < num_objs; i++) {
      if (detetction_info_pointer[i].detectionConfidence < coverage_threshold_) {
        continue;
      }
      vision_msgs::msg::Detection2D bbox_detection = toDetection2DMsg(
        detetction_info_pointer[i], tensor_header);
      detections_list.push_back(bbox_detection);
    }
    detections_msg.header = tensor_header;
    detections_msg.detections = detections_list;
  }

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

  vision_msgs::msg::Detection2D toDetection2DMsg(
    NvDsInferObjectDetectionInfo detection_info,
    const std_msgs::msg::Header & tensor_header)
  {
    int center_x = static_cast<int>(detection_info.left + (detection_info.width / 2));
    int center_y = static_cast<int>(detection_info.top + (detection_info.height / 2));
    int size_x = static_cast<int>(detection_info.width);
    int size_y = static_cast<int>(detection_info.height);

    vision_msgs::msg::Detection2D bbox_detection = GetNewDetection2DMsg(
      center_x, center_y, size_x, size_y,
      detection_info.classId, detection_info.detectionConfidence,
      tensor_header);

    return bbox_detection;
  }

  vision_msgs::msg::Detection2D GetNewDetection2DMsg(
    const int center_x,
    const int center_y,
    const int size_x,
    const int size_y,
    const int class_id,
    const float detection_score,
    const std_msgs::msg::Header & tensor_header)
  {
    // Create an empty message with the correct dimensions using the tensor
    vision_msgs::msg::Detection2D detection_msg;
    std::vector<vision_msgs::msg::ObjectHypothesisWithPose> hypothesis_list;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;

    hypothesis.hypothesis.class_id = class_id;
    hypothesis.hypothesis.score = static_cast<_Float64>(detection_score);
    hypothesis_list.push_back(hypothesis);

    detection_msg.header = tensor_header;

    detection_msg.bbox.center.position.x = center_x;
    detection_msg.bbox.center.position.y = center_y;
    detection_msg.bbox.center.theta = 0;
    detection_msg.bbox.size_x = size_x;
    detection_msg.bbox.size_y = size_y;
    detection_msg.results = hypothesis_list;

    return detection_msg;
  }
};

DetectNetDecoderNode::DetectNetDecoderNode(const rclcpp::NodeOptions options)
: Node("detectnet_decoder_node", options),
  // Parameters
  queue_size_(declare_parameter<int>("queue_size", rmw_qos_profile_default.depth)),
  label_names_(declare_parameter<std::vector<std::string>>(
      "label_names",
      std::vector<std::string>{})),
  coverage_threshold_(declare_parameter<float>("coverage_threshold", 0.6)),
  bounding_box_scale_(declare_parameter<float>("bounding_box_scale", 35.0)),
  bounding_box_offset_(declare_parameter<float>("bounding_box_offset", 0.5)),

  // Parameters for DBScan
  eps_(declare_parameter<float>("eps", 0.01)),
  min_boxes_(declare_parameter<int>("min_boxes", 1)),
  enable_athr_filter_(declare_parameter<int>("enable_athr_filter", 0)),
  threshold_athr_(declare_parameter<float>("threshold_athr", 0)),
  clustering_algorithm_(declare_parameter<int>("clustering_algorithm", 1)),

  // Subscribers
  tensor_list_sub_(create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
      "tensor_sub", queue_size_,
      std::bind(&DetectNetDecoderNode::DetectNetDecoderCallback, this, std::placeholders::_1))),
  // Publishers
  detections_pub_(create_publisher<vision_msgs::msg::Detection2DArray>(
      "detectnet/detections",
      1)),
  // Impl initialization
  impl_(std::make_unique<DetectNetDecoderImpl>(
      label_names_, coverage_threshold_, bounding_box_scale_,
      bounding_box_offset_, eps_, min_boxes_, enable_athr_filter_, threshold_athr_,
      clustering_algorithm_))
{
}

void DetectNetDecoderNode::DetectNetDecoderCallback(
  const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr tensor_list_msg)
{
  if (tensor_list_msg->tensors.size() != 2) {
    RCLCPP_ERROR(
      get_logger(), "Received invalid tensor count! Expected two tensors. Not processing.");
  }

  auto cov_tensor = tensor_list_msg->tensors[0];
  auto bbox_tensor = tensor_list_msg->tensors[1];

  vision_msgs::msg::Detection2DArray detections_msg;

  try {
    impl_->OnCallback(
      detections_msg,
      bbox_tensor,
      cov_tensor,
      tensor_list_msg->header,
      get_logger());
    detections_pub_->publish(detections_msg);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(get_logger(), e.what());
    return;
  }
}

DetectNetDecoderNode::~DetectNetDecoderNode() = default;

}  // namespace detectnet
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::detectnet::DetectNetDecoderNode)
