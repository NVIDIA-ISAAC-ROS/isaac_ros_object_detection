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

#include "isaac_ros_yolov8/yolov8_decoder_node.hpp"

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
namespace yolov8
{
YoloV8DecoderNode::YoloV8DecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("yolov8_decoder_node", options),
  nitros_sub_{std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
        nvidia::isaac_ros::nitros::NitrosTensorListView>>(
      this,
      "tensor_sub",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::supported_type_name,
      std::bind(&YoloV8DecoderNode::InputCallback, this,
      std::placeholders::_1))},
  
  // Publisher for output Detection2DArray messages
  pub_{create_publisher<vision_msgs::msg::Detection2DArray>(
      "detections_output", 50)},
  tensor_name_{declare_parameter<std::string>("tensor_name", "output_tensor")},
  confidence_threshold_{declare_parameter<double>("confidence_threshold", 0.25)},
  nms_threshold_{declare_parameter<double>("nms_threshold", 0.45)},
  target_width_{declare_parameter<int>("target_width", 1280)},
  target_height_{declare_parameter<int>("target_height", 720)},
  num_classes_{declare_parameter<int>("num_classes", 3)}
  
{}


YoloV8DecoderNode::~YoloV8DecoderNode() = default;

void YoloV8DecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView& msg)
{
  
  long int img_width = target_width_; // Specify target image width
  long int img_height = target_height_; // Specify target image height
  long int num_classes = num_classes_; // Specify number of classes

  auto tensor = msg.GetNamedTensor(tensor_name_);
  size_t buffer_size{tensor.GetTensorSize()};
  std::vector<float> results_vector{};
  results_vector.resize(buffer_size);
  cudaMemcpy(results_vector.data(), tensor.GetBuffer(), buffer_size, cudaMemcpyDefault);

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> indices;
  std::vector<int> classes;

  int out_dim = 8400;
  float* results_data = reinterpret_cast<float*>(results_vector.data());

  for (int i = 0; i < out_dim; i++) {
    float x = *(results_data + i);
    float y = *(results_data + (out_dim * 1) + i);
    float w = *(results_data + (out_dim * 2) + i);
    float h = *(results_data + (out_dim * 3) + i);

    // Convert coordinates from model output to target image dimensions
    // float x1 = ((x - (0.5 * w)));
    // float y1 = ((y - (0.5* h)));
    // debug
    float x1 = (x);
    float y1 = (y);
    float width = w;
    float height = h;

    std::vector<float> conf;
    for (int j = 0; j < num_classes; j++) {
      conf.push_back(*(results_data + (out_dim * (4 + j)) + i));
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
  RCLCPP_DEBUG(this->get_logger(), "# boxes after NMS: %lu", indices.size());

  vision_msgs::msg::Detection2DArray final_detections_arr;

  for (size_t i = 0; i < indices.size(); i++) {
    int ind = indices[i];
    vision_msgs::msg::Detection2D detection;

    geometry_msgs::msg::Pose center;
    geometry_msgs::msg::Point position;
    geometry_msgs::msg::Quaternion orientation;

    // 2D object Bbox
    vision_msgs::msg::BoundingBox2D bbox;
    // float x_center = bboxes[ind].x + (0.5 * w)+(img_width-640.0)/2;
    // float y_center = bboxes[ind].y + (0.5 * h)+(img_height-640.0)/2;


    float aspect_ratio = target_width_ / target_height_;

    if(aspect_ratio > 1.0){
      float width = 640.0;
      float height = 640.0 / aspect_ratio;

      float width_scale = target_width_ / width;
      float height_scale = target_height_ / height;

      float x1_scaled = bboxes[ind].x * width_scale;
      float y1_scaled = bboxes[ind].y * height_scale;

      float w = bboxes[ind].width * width_scale;
      float h = bboxes[ind].height * height_scale;

      float y_center = y1_scaled;
      float x_center = x1_scaled;

      float y_offset = 640 - height;
      y_center -= y_offset;

    }
    else{
      float width = 640/aspect_ratio;
      float height = 640;

      float width_scale = target_width_ / width;
      float height_scale = target_height_ / height;

      float x1_scaled = bboxes[ind].x * width_scale;
      float y1_scaled = bboxes[ind].y * height_scale;

      float w = bboxes[ind].width * width_scale;
      float h = bboxes[ind].height * height_scale;

      float y_center = y1_scaled;
      float x_center = x1_scaled;

      float x_offset = 640 - width;
      x_center -= x_offset;

    }

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
    detection.header.frame_id = msg.GetFrameId();
    final_detections_arr.detections.push_back(detection);
  }

  final_detections_arr.header.stamp.sec = msg.GetTimestampSeconds();
  final_detections_arr.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  final_detections_arr.header.frame_id = msg.GetFrameId();
  pub_->publish(final_detections_arr);
}

}  // namespace yolov8
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::yolov8::YoloV8DecoderNode)
