# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


RT_DETR_MODEL_INPUT_SIZE = 640  # RT-DETR models expect 640x640 encoded image size
RT_DETR_MODEL_NUM_CHANNELS = 3  # RT-DETR models expect 3 image channels

ESS_MODEL_IMAGE_WIDTH = 480
ESS_MODEL_IMAGE_HEIGHT = 288

SIM_IMAGE_WIDTH = 1920
SIM_IMAGE_HEIGHT = 1200

SIM_TO_RT_DETR_RATIO = SIM_IMAGE_WIDTH / RT_DETR_MODEL_INPUT_SIZE

ISAAC_ROS_ASSETS_PATH = os.path.join(os.environ['ISAAC_ROS_WS'], 'isaac_ros_assets')
ISAAC_ROS_MODELS_PATH = os.path.join(ISAAC_ROS_ASSETS_PATH, 'models')
SYNTHETICA_DETR_MODELS_PATH = os.path.join(ISAAC_ROS_MODELS_PATH, 'synthetica_detr')
RTDETR_ENGINE_PATH = os.path.join(SYNTHETICA_DETR_MODELS_PATH, 'sdetr_grasp.plan')


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_args = [
        DeclareLaunchArgument(
            'rt_detr_engine_file_path',
            default_value=RTDETR_ENGINE_PATH,
            description='The absolute file path to the RT-DETR TensorRT engine file'),

        DeclareLaunchArgument(
            'ess_depth_threshold',
            default_value='0.35',
            description='Threshold value ranges between 0.0 and 1.0 '
                        'for filtering disparity with confidence.'),

        DeclareLaunchArgument(
            'container_name',
            default_value='foundationpose_container',
            description='Name for ComposableNodeContainer'),
    ]

    rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path')
    container_name = LaunchConfiguration('container_name')

    # Resize and pad Isaac Sim images to RT-DETR model input image size
    resize_left_rt_detr_node = ComposableNode(
        name='resize_left_rt_detr_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': SIM_IMAGE_WIDTH,
            'input_height': SIM_IMAGE_HEIGHT,
            'output_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_height': RT_DETR_MODEL_INPUT_SIZE,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'front_stereo_camera/left_rgb/image_raw'),
            ('camera_info', 'front_stereo_camera/left_rgb/camerainfo')
        ]
    )
    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_image_height': RT_DETR_MODEL_INPUT_SIZE,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[(
            'image', 'resize/image'
        )]
    )

    # Convert image to tensor and reshape
    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'normalized_tensor'),
        ]
    )
    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [RT_DETR_MODEL_INPUT_SIZE,
                                   RT_DETR_MODEL_INPUT_SIZE,
                                   RT_DETR_MODEL_NUM_CHANNELS]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )
    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [RT_DETR_MODEL_NUM_CHANNELS,
                                   RT_DETR_MODEL_INPUT_SIZE,
                                   RT_DETR_MODEL_INPUT_SIZE],
            'output_tensor_shape': [1, RT_DETR_MODEL_NUM_CHANNELS,
                                    RT_DETR_MODEL_INPUT_SIZE,
                                    RT_DETR_MODEL_INPUT_SIZE]
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )
    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        remappings=[
            ('encoded_tensor', 'reshaped_tensor')
        ]
    )

    # RT-DETR objection detection pipeline
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': rt_detr_engine_file_path,
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'force_engine_update': False
        }]
    )
    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
    )

    isaac_ros_rtdetr_visualizer_node = Node(
        package='isaac_ros_rtdetr',
        executable='isaac_ros_rtdetr_visualizer.py',
        name='isaac_ros_rtdetr_visualizer',
        remappings=[
            ('image_rect', 'resize/image')
        ])

    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/rtdetr_processed_image'])

    rtdetr_container = ComposableNodeContainer(
        name=container_name,
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize_left_rt_detr_node,
            pad_node,
            image_to_tensor_node,
            interleave_to_planar_node,
            reshape_node,
            rtdetr_preprocessor_node,
            tensor_rt_node,
            rtdetr_decoder_node,
        ],
        output='screen'
    )

    return launch.LaunchDescription(launch_args + [rtdetr_container,
                                                   isaac_ros_rtdetr_visualizer_node,
                                                   rqt_image_view_node])
