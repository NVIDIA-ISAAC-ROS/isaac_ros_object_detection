# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    config = launch_dir_path + '/../config/params.yaml'

    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': 640,
            'network_image_height': 368,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative',
            'tensor_name': 'input_tensor'
        }],
        remappings=[('encoded_tensor', 'tensor_pub')]
    )

    triton_node = ComposableNode(
        name='triton_node',
        package='isaac_ros_triton',
        plugin='isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'detectnet',
            'model_repository_paths': ['/tmp/models'],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'output_tensor_names': ['output_cov', 'output_bbox'],
            'output_binding_names': ['output_cov/Sigmoid', 'output_bbox/BiasAdd'],
            'log_level': 0
        }])

    detectnet_decoder_node = ComposableNode(
        name='detectnet_decoder_node',
        package='isaac_ros_detectnet',
        plugin='isaac_ros::detectnet::DetectNetDecoderNode',
        parameters=[config]
    )

    container = ComposableNodeContainer(
        name='detectnet_container',
        namespace='detectnet_container',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[encoder_node, triton_node, detectnet_decoder_node],
        output='screen'
    )

    return launch.LaunchDescription([container])
