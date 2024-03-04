// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_yolonas/yolonas_decoder_node.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"


#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"


namespace nvidia
{
namespace isaac_ros
{
namespace yolonas
{
YoloNasDecoderNode::YoloNasDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolonas_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloNasDecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_conf_name_{declare_parameter<std::string>("tensor_conf_name", "conf")},
  tensor_bboxes_name_{declare_parameter<std::string>("tensor_bboxes_name", "bboxes")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  num_classes_{declare_parameter<int>("num_classes", 1)}
{}

YoloNasDecoderNode::~YoloNasDecoderNode() = default;

void YoloNasDecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto bbox_tensor = msg.GetNamedTensor(tensor_bboxes_name_);
  auto conf_tensor = msg.GetNamedTensor(tensor_conf_name_);
  size_t buffer_size{bbox_tensor.GetTensorSize()};
  size_t buffer_size_conf{conf_tensor.GetTensorSize()};
  std::vector<float> results_vector{};
  std::vector<float> conf_results_vector{};
  results_vector.resize(buffer_size);
  conf_results_vector.resize(buffer_size_conf);

  cudaMemcpy(results_vector.data(), bbox_tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);
  cudaMemcpy(conf_results_vector.data(), conf_tensor.GetBuffer(), buffer_size_conf, cudaMemcpyDefault);

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;


  int out_dim = conf_tensor.GetElementCount();
  float * results_data = reinterpret_cast<float *>(results_vector.data());
  float * conf_results_data = reinterpret_cast<float *>(conf_results_vector.data());
  RCLCPP_INFO(this->get_logger(), "Output tensor size: %d", out_dim);
  RCLCPP_INFO(this->get_logger(), "Confidence tensor size: %d", conf_tensor.GetTensorSize());
  int num_objs = 0;
  for (int i = 0; i < out_dim; i++) {
    float x0 = *(results_data + i*4);
    float y0 = *(results_data + i*4 + 1);
    float x1 = *(results_data + i*4 + 2);
    float y1 = *(results_data + i*4 + 3);

    float width = x1 - x0;
    float height = y1 - y0;


    std::vector<float> conf;
    for (int j = 0; j < num_classes_; j++) {
      conf.push_back(*(conf_results_data + (i * num_classes_) + j));
      if(conf[j]>=confidence_threshold_) num_objs++;
    }

    std::vector<float>::iterator ind_max_conf;
    ind_max_conf = std::max_element(std::begin(conf), std::end(conf));
    int max_index = distance(std::begin(conf), ind_max_conf);
    float val_max_conf = *max_element(std::begin(conf), std::end(conf));

    bboxes.push_back(cv::Rect(x0, y0, width, height));
    indices.push_back(i);
    scores.push_back(val_max_conf);
    classes.push_back(max_index);
  }

  RCLCPP_INFO(this->get_logger(), "Count of bboxes: %lu", bboxes.size());
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold_, nms_threshold_, indices, 5);
  RCLCPP_INFO(this->get_logger(), "# boxes after NMS: %lu", indices.size());
  RCLCPP_INFO(this->get_logger(), "# objects after NMS: %d", num_objs);
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
    float x_center = bboxes[ind].x + 0.5*w;
    float y_center = bboxes[ind].y + 0.5*h;
    detection.bbox.center.position.x = x_center;
    detection.bbox.center.position.y = y_center;
    detection.bbox.size_x = w;
    detection.bbox.size_y = h;


    // Class probabilities
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(classes.at(ind));
    hyp.hypothesis.score = scores.at(ind);
    detection.results.push_back(hyp);

    detection.header.stamp.sec = msg.GetTimestampSeconds();
    detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();

    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  pub_->publish(final_detections_arr);
}

}  // namespace yolonas
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolonas::YoloNasDecoderNode)
