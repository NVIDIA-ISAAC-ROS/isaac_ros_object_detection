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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class CameraInfoScaler(Node):
    """
    This node scales the camera info by a factor.

    This is used so that we can use RTDETR with manipulator servers.
    """

    def __init__(self):
        super().__init__('camera_info_scaler')

        self.declare_parameter('scale_factor', 1.0)

        self.scale_factor = self.get_parameter('scale_factor').get_parameter_value().double_value

        self.create_subscription(
            CameraInfo, 'input', self.camera_info_callback, 10)
        self.output_publisher = self.create_publisher(CameraInfo, 'output', 10)

    def camera_info_callback(self, camera_info_msg: CameraInfo):

        # Scale it down.
        camera_info_msg.width = int(camera_info_msg.width * self.scale_factor)
        camera_info_msg.height = int(camera_info_msg.height * self.scale_factor)
        camera_info_msg.k[0] = camera_info_msg.k[0] * self.scale_factor
        camera_info_msg.k[2] = camera_info_msg.k[2] * self.scale_factor
        camera_info_msg.k[4] = camera_info_msg.k[4] * self.scale_factor
        camera_info_msg.k[5] = camera_info_msg.k[5] * self.scale_factor

        camera_info_msg.p[0] = camera_info_msg.p[0] * self.scale_factor
        camera_info_msg.p[2] = camera_info_msg.p[2] * self.scale_factor
        camera_info_msg.p[5] = camera_info_msg.p[5] * self.scale_factor
        camera_info_msg.p[6] = camera_info_msg.p[6] * self.scale_factor

        camera_info_msg.roi.height = camera_info_msg.height
        camera_info_msg.roi.width = camera_info_msg.width
        camera_info_msg.binning_x = 1
        camera_info_msg.binning_y = 1

        self.output_publisher.publish(camera_info_msg)


def main():
    rclpy.init()
    rclpy.spin(CameraInfoScaler())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
