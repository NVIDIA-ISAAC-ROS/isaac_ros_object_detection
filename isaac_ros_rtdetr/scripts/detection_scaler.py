#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray


class ScaleType(Enum):
    SCALE_UP = 'scale_up'
    SCALE_DOWN = 'scale_down'


class SubPubMessageType(Enum):
    SINGLE = 'single'
    MULTI = 'multi'


class DetectionScaler(Node):
    """
    This node scales the detections to the RGB image size.

    The Detection2D messages output by RT-DETR are scaled to match the network's input dimension,
    typically 640x640. This node rescales those detection messages to match the original image
    dimensions.
    """

    def __init__(self):
        super().__init__('detection_scaler')

        self.declare_parameter('network_image_width', 640)
        self.declare_parameter('network_image_height', 640)
        self.declare_parameter('rgb_image_width', 1920)
        self.declare_parameter('rgb_image_height', 1200)
        self.declare_parameter('scale_type', ScaleType.SCALE_UP.value)
        self.declare_parameter('sub_pub_message_type', SubPubMessageType.SINGLE.value)

        self.network_image_width = self.get_parameter(
            'network_image_width').get_parameter_value().integer_value
        self.network_image_height = self.get_parameter(
            'network_image_height').get_parameter_value().integer_value
        self.rgb_image_width = self.get_parameter(
            'rgb_image_width').get_parameter_value().integer_value
        self.rgb_image_height = self.get_parameter(
            'rgb_image_height').get_parameter_value().integer_value
        self.scale_type = self.get_parameter(
            'scale_type').get_parameter_value().string_value
        self.sub_pub_message_type = self.get_parameter(
            'sub_pub_message_type').get_parameter_value().string_value

        width_scale = float(self.rgb_image_width / self.network_image_width)
        height_scale = float(self.rgb_image_height / self.network_image_height)
        scale = max(width_scale, height_scale)

        image_width_scale_interim = int(self.rgb_image_width / scale)
        image_height_scale_interim = int(self.rgb_image_height / scale)
        self.scale_width = self.rgb_image_width / image_width_scale_interim
        self.scale_height = self.rgb_image_height / image_height_scale_interim

        # Determine the message type based on the sub_pub_message_type parameter.
        if self.sub_pub_message_type == SubPubMessageType.MULTI.value:
            message_type = Detection2DArray
        else:
            message_type = Detection2D

        if self.scale_type == ScaleType.SCALE_UP.value:
            if self.sub_pub_message_type != SubPubMessageType.MULTI.value:
                raise ValueError(
                    'Scale up is only supported for multi message type')
            self.create_subscription(message_type, 'input', self.detection_array_callback, 10)
        else:
            if self.sub_pub_message_type != SubPubMessageType.SINGLE.value:
                raise ValueError(
                    'Scale down is only supported for single message type')
            self.create_subscription(message_type, 'input', self.detection_callback, 10)
        self.output_publisher = self.create_publisher(message_type, 'output', 10)

    def detection_array_callback(self, detections_msg: Detection2DArray):

        new_detection_msg = Detection2DArray()
        new_detection_msg.header = detections_msg.header
        for detection in detections_msg.detections:
            detection.bbox.center.position.x *= self.scale_width
            detection.bbox.center.position.y *= self.scale_height
            detection.bbox.size_x *= self.scale_width
            detection.bbox.size_y *= self.scale_height
            new_detection_msg.detections.append(detection)

        self.output_publisher.publish(new_detection_msg)

    def detection_callback(self, detections_msg: Detection2D):
        detections_msg.bbox.center.position.x /= self.scale_width
        detections_msg.bbox.center.position.y /= self.scale_height
        detections_msg.bbox.size_x /= self.scale_width
        detections_msg.bbox.size_y /= self.scale_height

        self.output_publisher.publish(detections_msg)


def main():
    rclpy.init()
    rclpy.spin(DetectionScaler())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
