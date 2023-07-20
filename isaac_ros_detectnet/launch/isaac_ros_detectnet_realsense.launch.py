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

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    config = launch_dir_path + '/../config/params_realsense.yaml'
    model_dir_path = '/workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet/models'

    # Read labels from text file
    labels_file_path = f'{model_dir_path}/detectnet/1/labels.txt'
    with open(labels_file_path, 'r') as fd:
        label_list = fd.read().strip().splitlines()

    realsense_camera_node = Node(
        name='camera',
        namespace='camera',
        package='realsense2_camera',
        executable='realsense2_camera_node',
        parameters=[{
                'enable_infra1': True,
                'enable_infra2': True,
                'enable_color': True,
                'enable_depth': True,
                'depth_module.emitter_enabled': 0,
                'depth_module.profile': '640x360x90',
                'rgb_camera.profile': '1280x720x30',
                'enable_gyro': True,
                'enable_accel': True,
                'gyro_fps': 200,
                'accel_fps': 200,
                'unite_imu_method': 2
        }]
    )

    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': 1280,
            'network_image_height': 720
        }],
        remappings=[('encoded_tensor', 'tensor_pub'),
                    ('image', 'camera/color/image_raw')]
    )

    triton_node = ComposableNode(
        name='triton_node',
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'detectnet',
            'model_repository_paths': [model_dir_path],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['output_cov', 'output_bbox'],
            'output_binding_names': ['output_cov/Sigmoid', 'output_bbox/BiasAdd'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'log_level': 0
        }]
    )
    
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[('tensor_pub', 'tensor_pub'),
                    ('tensor_sub', 'tensor_sub')],
        parameters=[{
            'model_file_path': model_dir_path,
            'engine_file_path': "/usr/src/tensorrt/bin/trtexec",
            'output_binding_names': ['output_cov/Sigmoid', 'output_bbox/BiasAdd'],
            'output_tensor_names': ['output_cov', 'output_bbox'],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    detectnet_decoder_node = ComposableNode(
        name='detectnet_decoder_node',
        package='isaac_ros_detectnet',
        plugin='nvidia::isaac_ros::detectnet::DetectNetDecoderNode',
        parameters=[config,
                    {
                        'label_list': label_list
                    }]
    )

    detectnet_container = ComposableNodeContainer(
        name='detectnet_container',
        namespace='detectnet_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            encoder_node, triton_node, detectnet_decoder_node],
        output='screen'
    )

    detectnet_visualizer_node = Node(
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet_visualizer.py',
        name='detectnet_visualizer',
        remappings=[('image', 'camera/color/image_raw')]

    )

    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_view',
        arguments=['/detectnet_processed_image'],
        parameters=[
                {'my_str': 'rgb8'},
        ]
    )

    return LaunchDescription([realsense_camera_node, detectnet_container, detectnet_visualizer_node, rqt_image_view_node
                              ])
