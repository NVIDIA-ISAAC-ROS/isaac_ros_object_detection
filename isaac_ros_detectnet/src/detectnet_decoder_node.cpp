/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_detectnet/detectnet_decoder_node.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wpedantic"

#include "gxf/std/timestamp.hpp"

#pragma GCC diagnostic pop


namespace nvidia
{
namespace isaac_ros
{
namespace detectnet
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

#define INPUT_COMPONENT_KEY_TENSORLIST          "detectnet_decoder/tensorlist_in"
#define INPUT_DEFAULT_TENSOR_FORMAT_TENSORLIST  "nitros_tensor_list_nchw_rgb_f32"
#define INPUT_TOPIC_NAME_TENSORLIST             "tensor_sub"

#define OUTPUT_COMPONENT_KEY_DETECTIONS         "vault/vault"
#define OUTPUT_DEFAULT_TENSOR_FORMAT_DETECTIONS "nitros_detection2_d_array"
#define OUTPUT_TOPIC_NAME_TAG_DETECTIONS        "detectnet/detections"

constexpr char APP_YAML_FILENAME[] = "config/detectnet_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_detectnet";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_nitros", "gxf/std/libgxf_std.so"},
  {"isaac_ros_nitros", "gxf/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_nitros", "gxf/serialization/libgxf_serialization.so"},
  {"isaac_ros_nitros", "gxf/cuda/libgxf_cuda.so"},
  {"isaac_ros_nitros_detection2_d_array_type", "gxf/libgxf_detectnet.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_detectnet",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY_TENSORLIST,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = INPUT_DEFAULT_TENSOR_FORMAT_TENSORLIST,
      .topic_name = INPUT_TOPIC_NAME_TENSORLIST,
    }
  },
  {OUTPUT_COMPONENT_KEY_DETECTIONS,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(1),
      .compatible_data_format = OUTPUT_DEFAULT_TENSOR_FORMAT_DETECTIONS,
      .topic_name = OUTPUT_TOPIC_NAME_TAG_DETECTIONS,
      .frame_id_source_key = INPUT_COMPONENT_KEY_TENSORLIST,
    }
  }
};
#pragma GCC diagnostic pop

DetectNetDecoderNode::DetectNetDecoderNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
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
  bounding_box_offset_(declare_parameter<double>("bounding_box_offset", 0.0))
{
  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] Constructor");

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection2DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

void DetectNetDecoderNode::preLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] pretLoadGraphCallback().");
}

void DetectNetDecoderNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[DetectNetDecoderNode] postLoadGraphCallback().");
  getNitrosContext().setParameter1DStrVector(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "label_list",
    label_list_);
  getNitrosContext().setParameterBool(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "enable_confidence_threshold",
    enable_confidence_threshold_);
  getNitrosContext().setParameterBool(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "enable_bbox_area_threshold",
    enable_bbox_area_threshold_);
  getNitrosContext().setParameterBool(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "enable_dbscan_clustering",
    enable_dbscan_clustering_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "confidence_threshold",
    confidence_threshold_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "min_bbox_area", min_bbox_area_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "dbscan_confidence_threshold",
    dbscan_confidence_threshold_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "dbscan_eps",
    dbscan_eps_);
  getNitrosContext().setParameterInt32(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "dbscan_min_boxes",
    dbscan_min_boxes_);
  getNitrosContext().setParameterInt32(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "dbscan_enable_athr_filter",
    dbscan_enable_athr_filter_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "dbscan_threshold_athr",
    dbscan_threshold_athr_);
  getNitrosContext().setParameterInt32(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "dbscan_clustering_algorithm",
    dbscan_clustering_algorithm_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "bounding_box_scale",
    bounding_box_scale_);
  getNitrosContext().setParameterFloat64(
    "detectnet_decoder", "nvidia::isaac_ros::DetectnetDecoder", "bounding_box_offset",
    bounding_box_offset_);
}

DetectNetDecoderNode::~DetectNetDecoderNode() = default;

}  // namespace detectnet
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::detectnet::DetectNetDecoderNode)
