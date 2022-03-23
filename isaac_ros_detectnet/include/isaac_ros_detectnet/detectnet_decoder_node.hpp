/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISAAC_ROS_DETECTNET__DETECTNET_DECODER_NODE_HPP_
#define ISAAC_ROS_DETECTNET__DETECTNET_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_nvengine_interfaces/msg/tensor_list.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "rclcpp/rclcpp.hpp"

namespace isaac_ros
{
namespace detectnet
{

class DetectNetDecoderNode : public rclcpp::Node
{
public:
  explicit DetectNetDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~DetectNetDecoderNode();

private:
/**
 * @brief Callback to decode a tensor list output by a DetectNet architecture
 *        and then publish a detection list
 *
 * @param tensor_list_msg The TensorList msg representing the detection list output by DetectNet
 */
  void DetectNetDecoderCallback(
    const isaac_ros_nvengine_interfaces::msg::TensorList::ConstSharedPtr tensor_list_msg);

  // Queue size of subscriber
  int queue_size_;

  // Frame id that the message should be in
  std::string header_frame_id_;
  // A list of class labels in the order they are used in the model
  std::vector<std::string> label_names_;
  // coverage threshold to discard detections.
  // Detections with lower coverage than the threshold will be discarded
  float coverage_threshold_;
  // Bounding box normalization for both X and Y dimensions. This value is set in the DetectNetv2
  // training specification.
  float bounding_box_scale_;
  // Bounding box offset for both X and Y dimensions. This value is set in the DetectNetv2
  // training specification.
  float bounding_box_offset_;

  // Parameters for DBscan.
  float eps_;
  int min_boxes_;
  int enable_athr_filter_;
  float threshold_athr_;
  int clustering_algorithm_;

  // Subscribes to a Tensor that will be converted to a detection list
  rclcpp::Subscription<isaac_ros_nvengine_interfaces::msg::TensorList>::SharedPtr tensor_list_sub_;

  // Publishes the processed Tensor as an array of detections (Detection2DArray)
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;
  struct DetectNetDecoderImpl;
  std::unique_ptr<DetectNetDecoderImpl> impl_;  // Pointer to implementation
};

}  // namespace detectnet
}  // namespace isaac_ros

#endif  // ISAAC_ROS_DETECTNET__DETECTNET_DECODER_NODE_HPP_
