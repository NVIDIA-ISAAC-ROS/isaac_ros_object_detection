// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_DETECTNET__DETECTNET_DECODER_NODE_HPP_
#define ISAAC_ROS_DETECTNET__DETECTNET_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_nitros/nitros_node.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace detectnet
{

class DetectNetDecoderNode : public nitros::NitrosNode
{
public:
  explicit DetectNetDecoderNode(const rclcpp::NodeOptions &);

  ~DetectNetDecoderNode();

  DetectNetDecoderNode(const DetectNetDecoderNode &) = delete;

  DetectNetDecoderNode & operator=(const DetectNetDecoderNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  // List of string labels for the specific network
  const std::vector<std::string> label_list_;
  // Flag to enable minimum confidence thresholding
  const bool enable_confidence_threshold_;
  // Flag to enable minimum bounding box area thresholding
  const bool enable_bbox_area_threshold_;
  // Flag to enable Dbscan clustering
  const bool enable_dbscan_clustering_;
  // The min value of confidence used to threshold detections before clustering
  const double confidence_threshold_;
  // The min value of bouding box area used to threshold detections before clustering
  const double min_bbox_area_;
  // Minimum score in a cluster for the cluster to be considered an object
  // during grouping. Different clustering may cause the algorithm
  // to use different scores
  const double dbscan_confidence_threshold_;
  // Holds the epsilon to control merging of overlapping boxes.
  // Refer to OpenCV groupRectangles and DBSCAN documentation for more information on epsilon.
  const double dbscan_eps_;
  // Holds the minimum number of boxes in a cluster to be considered
  // an object during grouping using DBSCAN
  const int dbscan_min_boxes_;
  // true enables the area-to-hit ratio (ATHR) filter.
  // The ATHR is calculated as: ATHR = sqrt(clusterArea) / nObjectsInCluster.
  const int dbscan_enable_athr_filter_;
  // Holds the area-to-hit ratio threshold
  const double dbscan_threshold_athr_;
  // The clustering algorithm used. 1 => NvDsInferDBScanCluster 2 => NvDsInferDBScanClusterHybrid
  const int dbscan_clustering_algorithm_;
  // The scale parameter, which should match the training configuration
  const double bounding_box_scale_;
  // Bounding box offset for both X and Y dimensions
  const double bounding_box_offset_;
};

}  // namespace detectnet
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DETECTNET__DETECTNET_DECODER_NODE_HPP_
