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

#include "isaac_ros_rtdetr/rtdetr_decoder_node.hpp"

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
namespace rtdetr
{
RTDETRDecoderNode::RTDETRDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("rtdetr_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&RTDETRDecoderNode::InputCallback, this,
      std::placeholders::_1))},
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_conf_name_{declare_parameter<std::string>("tensor_conf_name", "scores")},
  tensor_bboxes_name_{declare_parameter<std::string>("tensor_bboxes_name", "boxes")},
  tensor_labels_name_{declare_parameter<std::string>("tensor_conf_name", "labels")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)}
{}

RTDETRDecoderNode::~RTDETRDecoderNode() = default;

void RTDETRDecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  auto bbox_tensor = msg.GetNamedTensor(tensor_bboxes_name_);
  auto conf_tensor = msg.GetNamedTensor(tensor_conf_name_);
  auto labels_tensor = msg.GetNamedTensor(tensor_labels_name_);
  size_t buffer_size{bbox_tensor.GetTensorSize()};
  size_t buffer_size_conf{conf_tensor.GetTensorSize()};
  size_t buffer_size_labels{labels_tensor.GetTensorSize()};
  std::vector<float> results_vector{};
  std::vector<long long> labels_vec{};
  std::vector<float> conf_results_vector{};
  results_vector.resize(buffer_size);
  conf_results_vector.resize(buffer_size_conf);
  labels_vec.resize(buffer_size_labels);

  cudaMemcpy(results_vector.data(), bbox_tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);
  cudaMemcpy(conf_results_vector.data(), conf_tensor.GetBuffer(), buffer_size_conf, cudaMemcpyDefault);
  cudaMemcpy(labels_vec.data(), labels_tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);
  



  float * results_data = reinterpret_cast<float *>(results_vector.data());
  float * conf_results_data = reinterpret_cast<float *>(conf_results_vector.data());
  long long* labels_data = reinterpret_cast<long long *>(labels_vec.data());

  vision_msgs::msg::Detection2DArray final_detections_arr;

  for (int i = 0; i < labels_vec.size(); i++) {
    if(conf_results_vector[i]>confidence_threshold_){
      float x_min = results_vector[4*i];
      float y_min = results_vector[4*i+1];
      float x_max = results_vector[4*i+2];
      float y_max = results_vector[4*i+3];
      
      vision_msgs::msg::Detection2D detection;
      vision_msgs::msg::BoundingBox2D bbox;
      float w = x_max - x_min;
      float h = y_max - y_min;
      float x_center = (x_max+x_min)/2;
      float y_center = (y_max + y_min)/2;
      detection.bbox.center.position.x = x_center;
      detection.bbox.center.position.y = y_center;
      detection.bbox.size_x = w;
      detection.bbox.size_y = h;


      // Class probabilities
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(labels_data[i]);
      hyp.hypothesis.score = conf_results_vector[i];
      detection.results.push_back(hyp);

      detection.header.stamp.sec = msg.GetTimestampSeconds();
      detection.header.stamp.nanosec = msg.GetTimestampNanoseconds();



    }
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  pub_->publish(final_detections_arr);
}

}  // namespace rtdetr
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::rtdetr::RTDETRDecoderNode)
