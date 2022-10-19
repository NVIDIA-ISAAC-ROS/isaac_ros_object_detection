# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from ament_index_python.packages import get_package_share_directory
from launch import actions, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    my_package_dir = get_package_share_directory('isaac_ros_detectnet')
    return LaunchDescription([
        actions.ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '-l',
                 os.path.join(my_package_dir, 'detectnet_rosbag')]
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                my_package_dir, 'launch'),
                '/isaac_ros_detectnet.launch.py'])
        ),
        Node(
            package='isaac_ros_detectnet',
            executable='isaac_ros_detectnet_visualizer.py',
            name='detectnet_visualizer'
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='image_view',
            arguments=['/detectnet_processed_image']
        )
    ])
