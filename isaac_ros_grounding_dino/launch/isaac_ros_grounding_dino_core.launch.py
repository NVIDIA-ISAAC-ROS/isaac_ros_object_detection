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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

MODEL_NUM_CHANNELS = 3  # Grounding DINO models expect 3 image channels


class IsaacROSGroundingDinoLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

        # DNN Image Encoder parameters
        network_image_width = LaunchConfiguration('network_image_width')
        network_image_height = LaunchConfiguration('network_image_height')
        encoder_image_mean = LaunchConfiguration('encoder_image_mean')
        encoder_image_stddev = LaunchConfiguration('encoder_image_stddev')

        # TensorRT parameters
        model_file_path = LaunchConfiguration('model_file_path')
        engine_file_path = LaunchConfiguration('engine_file_path')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        tensorrt_verbose = LaunchConfiguration('tensorrt_verbose')
        force_engine_update = LaunchConfiguration('force_engine_update')

        # Grounding DINO parameters
        confidence_threshold = LaunchConfiguration('confidence_threshold')

        return {
            'resize_node': ComposableNode(
                name='resize_node',
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                parameters=[{
                    'input_width': interface_specs['camera_resolution']['width'],
                    'input_height': interface_specs['camera_resolution']['height'],
                    'output_width': network_image_width,
                    'output_height': network_image_height,
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
                    'output_image_width': network_image_width,
                    'output_image_height': network_image_height,
                    'padding_type': 'BOTTOM_RIGHT'
                }],
                remappings=[(
                    'image', 'resize/image'
                )]
            ),
            'image_to_tensor_node': ComposableNode(
                name='image_to_tensor_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
                parameters=[{
                    'scale': True,
                    'tensor_name': 'image',
                }],
                remappings=[
                    ('image', 'padded_image'),
                    ('tensor', 'image_tensor'),
                ]
            ),
            'normalize_node': ComposableNode(
                name='normalize_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
                parameters=[{
                    'mean': encoder_image_mean,
                    'stddev': encoder_image_stddev,
                    'input_tensor_name': 'image',
                    'output_tensor_name': 'image'
                }],
                remappings=[
                    ('tensor', 'image_tensor'),
                ],
            ),
            'interleave_to_planar_node': ComposableNode(
                name='interleaved_to_planar_node',
                package='isaac_ros_tensor_proc',
                plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
                parameters=[{
                    'input_tensor_shape': [
                        network_image_height, network_image_width, MODEL_NUM_CHANNELS]
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
                    'output_tensor_name': 'images',
                    'input_tensor_shape': [
                        MODEL_NUM_CHANNELS, network_image_height, network_image_width],
                    'output_tensor_shape': [
                        1, MODEL_NUM_CHANNELS, network_image_height, network_image_width]
                }],
                remappings=[
                    ('tensor', 'planar_tensor')
                ],
            ),
            'grounding_dino_preprocessor': ComposableNode(
                name='grounding_dino_preprocessor',
                package='isaac_ros_grounding_dino',
                plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode',
                parameters=[{
                    'default_prompt': 'trash can on the left.person to the right.plastic bag.',
                }],
                remappings=[
                    ('image_tensor', 'reshaped_tensor')
                ]
            ),
            'grounding_dino_inference_node': ComposableNode(
                name='grounding_dino_inference',
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                parameters=[{
                    'model_file_path': model_file_path,
                    'engine_file_path': engine_file_path,
                    'input_tensor_names': input_tensor_names,
                    'input_binding_names': input_binding_names,
                    'output_tensor_names': output_tensor_names,
                    'output_binding_names': output_binding_names,
                    'verbose': tensorrt_verbose,
                    'force_engine_update': force_engine_update
                }]
            ),
            'grounding_dino_decoder_node': ComposableNode(
                name='grounding_dino_decoder',
                package='isaac_ros_grounding_dino',
                plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode',
                parameters=[{
                    'confidence_threshold': confidence_threshold,
                    'image_width': network_image_width,
                    'image_height': network_image_height,
                }],
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        return {
            'network_image_width': DeclareLaunchArgument(
                'network_image_width',
                default_value='960',
                description='The input image width that the network expects'
            ),
            'network_image_height': DeclareLaunchArgument(
                'network_image_height',
                default_value='544',
                description='The input image height that the network expects'
            ),
            'encoder_image_mean': DeclareLaunchArgument(
                'encoder_image_mean',
                default_value='[0.485, 0.456, 0.406]',
                description='The mean for image normalization'
            ),
            'encoder_image_stddev': DeclareLaunchArgument(
                'encoder_image_stddev',
                default_value='[0.229, 0.224, 0.225]',
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
                default_value=(
                    '["images", "input_ids", "attention_mask", "position_ids", '
                    '"token_type_ids", "text_token_mask"]'),
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value=(
                    '["inputs", "input_ids", "attention_mask", "position_ids", '
                    '"token_type_ids", "text_token_mask"]'),
                description='A list of input tensor binding names (specified by model)'
            ),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["scores", "boxes"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["pred_logits", "pred_boxes"]',
                description='A list of output tensor binding names (specified by model)'
            ),
            'tensorrt_verbose': DeclareLaunchArgument(
                'tensorrt_verbose',
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
                default_value='0.5',
                description='The confidence threshold for detections'
            ),
            'text_tokenizer_node': Node(
                package='isaac_ros_grounding_dino',
                executable='isaac_ros_grounding_dino_text_tokenizer.py',
                name='grounding_dino_text_tokenizer',
                output='screen',
            ),
        }


def generate_launch_description():
    grounding_dino_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='grounding_dino_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSGroundingDinoLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription(
        [grounding_dino_container] +
        IsaacROSGroundingDinoLaunchFragment.get_launch_actions().values())
