# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory
from isaac_ros_examples import IsaacROSLaunchFragment
import launch

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


class IsaacROSYolov8LaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

        # TensorRT parameters
        model_file_path = LaunchConfiguration('model_file_path')
        engine_file_path = LaunchConfiguration('engine_file_path')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        verbose = LaunchConfiguration('verbose')
        force_engine_update = LaunchConfiguration('force_engine_update')

        # YOLOv8 Decoder parameters
        confidence_threshold = LaunchConfiguration('confidence_threshold')
        nms_threshold = LaunchConfiguration('nms_threshold')

        return {
            'tensor_rt_node': ComposableNode(
                name='tensor_rt',
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                parameters=[{
                    'model_file_path': model_file_path,
                    'engine_file_path': engine_file_path,
                    'output_binding_names': output_binding_names,
                    'output_tensor_names': output_tensor_names,
                    'input_tensor_names': input_tensor_names,
                    'input_binding_names': input_binding_names,
                    'verbose': verbose,
                    'force_engine_update': force_engine_update
                }]
            ),
            'yolov8_decoder_node': ComposableNode(
                name='yolov8_decoder_node',
                package='isaac_ros_yolov8',
                plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
                parameters=[{
                    'confidence_threshold': confidence_threshold,
                    'nms_threshold': nms_threshold,
                }]
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:

        network_image_width = LaunchConfiguration('network_image_width')
        network_image_height = LaunchConfiguration('network_image_height')
        image_mean = LaunchConfiguration('image_mean')
        image_stddev = LaunchConfiguration('image_stddev')

        encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')

        return {
            'network_image_width': DeclareLaunchArgument(
                'network_image_width',
                default_value='640',
                description='The input image width that the network expects'
            ),
            'network_image_height': DeclareLaunchArgument(
                'network_image_height',
                default_value='640',
                description='The input image height that the network expects'
            ),
            'image_mean': DeclareLaunchArgument(
                'image_mean',
                default_value='[0.0, 0.0, 0.0]',
                description='The mean for image normalization'
            ),
            'image_stddev': DeclareLaunchArgument(
                'image_stddev',
                default_value='[1.0, 1.0, 1.0]',
                description='The standard deviation for image normalization'
            ),
            'model_file_path': DeclareLaunchArgument(
                'model_file_path',
                default_value='',
                description='The absolute file path to the ONNX file'
            ),
            'engine_file_path': DeclareLaunchArgument(
                'engine_file_path',
                default_value='',
                description='The absolute file path to the TensorRT engine file'
            ),
            'input_tensor_names': DeclareLaunchArgument(
                'input_tensor_names',
                default_value='["input_tensor"]',
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value='["images"]',
                description='A list of input tensor binding names (specified by model)'
            ),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["output_tensor"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["output0"]',
                description='A list of output tensor binding names (specified by model)'
            ),
            'verbose': DeclareLaunchArgument(
                'verbose',
                default_value='False',
                description='Whether TensorRT should verbosely log or not'
            ),
            'force_engine_update': DeclareLaunchArgument(
                'force_engine_update',
                default_value='False',
                description='Whether TensorRT should update the TensorRT engine file or not'
            ),
            'confidence_threshold': DeclareLaunchArgument(
                'confidence_threshold',
                default_value='0.25',
                description='Confidence threshold to filter candidate detections during NMS'
            ),
            'nms_threshold': DeclareLaunchArgument(
                'nms_threshold',
                default_value='0.45',
                description='NMS IOU threshold'
            ),
            'yolov8_encoder_launch': IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
                ),
                launch_arguments={
                    'input_image_width': str(interface_specs['camera_resolution']['width']),
                    'input_image_height': str(interface_specs['camera_resolution']['height']),
                    'network_image_width': network_image_width,
                    'network_image_height': network_image_height,
                    'image_mean': image_mean,
                    'image_stddev': image_stddev,
                    'attach_to_shared_component_container': 'True',
                    'component_container_name': '/isaac_ros_examples/container',
                    'dnn_image_encoder_namespace': 'yolov8_encoder',
                    'image_input_topic': '/image_rect',
                    'camera_info_input_topic': '/camera_info_rect',
                    'tensor_output_topic': '/tensor_pub',
                }.items(),
            ),
        }


def generate_launch_description():
    yolov8_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='yolov8_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSYolov8LaunchFragment
        .get_composable_nodes().values(),
        arguments=['--ros-args', '--log-level', 'INFO'],
        output='screen'
    )

    return launch.LaunchDescription(
        [yolov8_container] + IsaacROSYolov8LaunchFragment.get_launch_actions().values())
