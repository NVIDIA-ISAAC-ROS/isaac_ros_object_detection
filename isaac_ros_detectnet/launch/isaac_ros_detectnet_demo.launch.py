# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
