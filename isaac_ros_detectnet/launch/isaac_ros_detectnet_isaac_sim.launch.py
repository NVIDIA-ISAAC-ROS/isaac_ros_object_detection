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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    config = launch_dir_path + '/../config/params_isaac_sim.yaml'
    isaac_ros_ws_path = os.environ.get('ISAAC_ROS_WS', '')
    model_dir_path = os.path.join(isaac_ros_ws_path,
                                  'isaac_ros_assets/models')

    # Read labels from text file
    labels_file_path = f'{model_dir_path}/peoplenet/1/labels.txt'
    with open(labels_file_path, 'r') as fd:
        label_list = fd.read().strip().splitlines()

    image_resize_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize_node_left',
        parameters=[{
                'output_width': 1280,
                'output_height': 720,
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('camera_info', 'front_stereo_camera/left_rgb/camerainfo'),
            ('image', 'front_stereo_camera/left_rgb/image_raw'),
            ('resize/camera_info', 'front_stereo_camera/left_rgb/camerainfo_resize'),
            ('resize/image', 'front_stereo_camera/left_rgb/image_resize')]
    )

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    detectnet_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch',
                          'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': str(1280),
            'input_image_height': str(720),
            'network_image_width': str(1280),
            'network_image_height': str(720),
            'image_mean': str([0.0, 0.0, 0.0]),
            'image_stddev': str([1.0, 1.0, 1.0]),
            'enable_padding': 'False',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'detectnet_container/detectnet_container',
            'dnn_image_encoder_namespace': 'detectnet_encoder',
            'image_input_topic': '/front_stereo_camera/left_rgb/image_resize',
            'camera_info_input_topic': '/front_stereo_camera/left_rgb/camerainfo_resize',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )

    triton_node = ComposableNode(
        name='triton_node',
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'peoplenet',
            'model_repository_paths': [model_dir_path],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['output_cov', 'output_bbox'],
            'output_binding_names': ['output_cov/Sigmoid', 'output_bbox/BiasAdd'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'log_level': 0
        }])

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
            image_resize_node_left, triton_node, detectnet_decoder_node],
        output='screen'
    )

    detectnet_visualizer_node = Node(
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet_visualizer.py',
        name='detectnet_visualizer',
        remappings=[('image', 'front_stereo_camera/left_rgb/image_resize')]

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

    return LaunchDescription([
        detectnet_container, detectnet_encoder_launch,
        detectnet_visualizer_node, rqt_image_view_node
    ])
