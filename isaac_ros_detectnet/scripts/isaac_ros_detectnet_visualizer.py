#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script listens for images and object detections on the image,
# then renders the output boxes on top of the image and publishes
# the result as an image message

import math

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

DETECTNET_DEFAULT_WIDTH = 960
DETECTNET_DEFAULT_HEIGHT = 544


def scale_bbox_to_image(center_x, center_y, width, height, image_width, image_height,
                        network_image_width, network_image_height):
    """Scale bounding box coordinates from DetectNet network space to image space."""
    if network_image_width <= 0 or network_image_height <= 0:
        raise ValueError('network image dimensions must be positive')

    scale_x = float(image_width) / float(network_image_width)
    scale_y = float(image_height) / float(network_image_height)

    return (
        center_x * scale_x,
        center_y * scale_y,
        width * scale_x,
        height * scale_y,
    )


class DetectNetVisualizer(Node):
    QUEUE_SIZE = 10
    color = (0, 255, 0)
    bbox_thickness = 1

    def __init__(self):
        super().__init__('detectnet_visualizer')
        self.declare_parameter('network_image_width', DETECTNET_DEFAULT_WIDTH)
        self.declare_parameter('network_image_height', DETECTNET_DEFAULT_HEIGHT)
        self.network_image_width = self.get_parameter(
            'network_image_width').get_parameter_value().integer_value
        self.network_image_height = self.get_parameter(
            'network_image_height').get_parameter_value().integer_value
        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            Image, 'detectnet_processed_image',  self.QUEUE_SIZE)

        self._detections_subscription = message_filters.Subscriber(
            self,
            Detection2DArray,
            'detectnet/detections')
        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            'image')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self._detections_subscription, self._image_subscription],
            self.QUEUE_SIZE)

        self.time_synchronizer.registerCallback(self.detections_callback)

    def detections_callback(self, detections_msg, img_msg):
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg)
        for detection in detections_msg.detections:
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            width = detection.bbox.size_x
            height = detection.bbox.size_y
            try:
                center_x, center_y, width, height = scale_bbox_to_image(
                    center_x, center_y, width, height, img_msg.width, img_msg.height,
                    self.network_image_width, self.network_image_height)
            except ValueError as error:
                self.get_logger().error(str(error), once=True)
                return

            min_x = float(center_x - (width / 2.0))
            min_y = float(center_y - (height / 2.0))
            max_x = float(center_x + (width / 2.0))
            max_y = float(center_y + (height / 2.0))

            if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
                continue

            min_pt = (int(round(min_x)), int(round(min_y)))
            max_pt = (int(round(max_x)), int(round(max_y)))

            cv2.rectangle(cv2_img, min_pt, max_pt,
                          self.color, self.bbox_thickness)

        processed_img = self._bridge.cv2_to_imgmsg(
            cv2_img, encoding=img_msg.encoding)
        self._processed_image_pub.publish(processed_img)


def main():
    rclpy.init()
    rclpy.spin(DetectNetVisualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
