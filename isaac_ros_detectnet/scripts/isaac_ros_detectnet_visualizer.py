#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DetectNetVisualizer(Node):
    QUEUE_SIZE = 10
    color = (0, 255, 0)
    bbox_thickness = 1

    def __init__(self):
        super().__init__('detectnet_visualizer')
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

            min_pt = (round(center_x - (width / 2.0)),
                      round(center_y - (height / 2.0)))
            max_pt = (round(center_x + (width / 2.0)),
                      round(center_y + (height / 2.0)))

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
