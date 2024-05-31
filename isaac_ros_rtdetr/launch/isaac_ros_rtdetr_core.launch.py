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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

MODEL_INPUT_SIZE = 640  # RT-DETR models expect 640x640 encoded image size
MODEL_NUM_CHANNELS = 3  # RT-DETR models expect 3 image channels


class IsaacROSRtDetrLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

        # RT-DETR parameters
        confidence_threshold = LaunchConfiguration('confidence_threshold')

        # TensorRT parameters
        model_file_path = LaunchConfiguration('model_file_path')
        engine_file_path = LaunchConfiguration('engine_file_path')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        verbose = LaunchConfiguration('verbose')
        force_engine_update = LaunchConfiguration('force_engine_update')

        return {
            'resize_node': ComposableNode(
                name='resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'input_width': interface_specs['camera_resolution']['width'],
                    'input_height': interface_specs['camera_resolution']['height'],
                    'output_width': MODEL_INPUT_SIZE,
                    'output_height': MODEL_INPUT_SIZE,
                    'keep_aspect_ratio': True,
                    'encoding_desired': 'rgb8',
                    'disable_padding': True
                }],
                remappings=[
                    ('image', 'image_rect'),
                    ('camera_info', 'camera_info_rect')
                ]
            ),
            'pad_node': ComposableNode(
                name='pad_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::PadNode',
                parameters=[{
                    'output_image_width': MODEL_INPUT_SIZE,
                    'output_image_height': MODEL_INPUT_SIZE,
                    'padding_type': 'BOTTOM_RIGHT'
                }],
                remappings=[(
                    'image', 'resize/image'
                )]
            ),
            'image_format_node': ComposableNode(
                name='image_format_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                parameters=[{
                        'encoding_desired': 'rgb8',
                        'image_width': MODEL_INPUT_SIZE,
                        'image_height': MODEL_INPUT_SIZE
                }],
                remappings=[
                    ('image_raw', 'padded_image'),
                    ('image', 'image_rgb')]
            ),
            'image_to_tensor_node': ComposableNode(
                name='image_to_tensor_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
                parameters=[{
                    'scale': False,
                    'tensor_name': 'image',
                }],
                remappings=[
                    ('image', 'image_rgb'),
                    ('tensor', 'normalized_tensor'),
                ]
            ),
            'interleave_to_planar_node': ComposableNode(
                name='interleaved_to_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{
                    'input_tensor_shape': [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_NUM_CHANNELS]
                }],
                remappings=[
                    ('interleaved_tensor', 'normalized_tensor')
                ]
            ),
            'reshape_node': ComposableNode(
                name='reshape_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
                parameters=[{
                    'output_tensor_name': 'input_tensor',
                    'input_tensor_shape': [MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                    'output_tensor_shape': [
                        1, MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]
                }],
                remappings=[
                    ('tensor', 'planar_tensor')
                ],
            ),
            'rtdetr_preprocessor_node': ComposableNode(
                name='rtdetr_preprocessor',
                package='isaac_ros_rtdetr',
                plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
                parameters=[{
                    'image_width': interface_specs['camera_resolution']['width'],
                    'image_height': interface_specs['camera_resolution']['height'],
                }],
                remappings=[
                    ('encoded_tensor', 'reshaped_tensor')
                ]
            ),
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
            'rtdetr_decoder_node': ComposableNode(
                name='rtdetr_decoder',
                package='isaac_ros_rtdetr',
                plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
                parameters=[{
                    'confidence_threshold': confidence_threshold
                }]
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        return {
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
                default_value='["images", "orig_target_sizes"]',
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value='["images", "orig_target_sizes"]',
                description='A list of input tensor binding names (specified by model)'
            ),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["labels", "boxes", "scores"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["labels", "boxes", "scores"]',
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
                default_value='0.9',
                description='The minimum confidence threshold for detections'
            ),
        }


def generate_launch_description():
    rtdetr_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='rtdetr_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSRtDetrLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription(
        [rtdetr_container] + IsaacROSRtDetrLaunchFragment.get_launch_actions().values())
