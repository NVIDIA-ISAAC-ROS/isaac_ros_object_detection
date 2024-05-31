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

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch the DNN Image encoder, TensorRT node and YOLOv8 decoder node."""
    launch_args = [
        DeclareLaunchArgument(
            'input_image_width',
            default_value='1920',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='1200',
            description='The input image height'),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='640',
            description='The input image width that the network expects'),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='640',
            description='The input image height that the network expects'),
        DeclareLaunchArgument(
            'image_mean',
            default_value='[0.0, 0.0, 0.0]',
            description='The mean for image normalization'),
        DeclareLaunchArgument(
            'image_stddev',
            default_value='[1.0, 1.0, 1.0]',
            description='The standard deviation for image normalization'),
        DeclareLaunchArgument(
            'model_file_path',
            default_value='',
            description='The absolute file path to the ONNX file'),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'input_tensor_names',
            default_value='["input_tensor"]',
            description='A list of tensor names to bound to the specified input binding names'),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value='["images"]',
            description='A list of input tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'output_tensor_names',
            default_value='["output_tensor"]',
            description='A list of tensor names to bound to the specified output binding names'),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value='["output0"]',
            description='A  list of output tensor binding names (specified by model)'),
        DeclareLaunchArgument(
            'verbose',
            default_value='False',
            description='Whether TensorRT should verbosely log or not'),
        DeclareLaunchArgument(
            'force_engine_update',
            default_value='False',
            description='Whether TensorRT should update the TensorRT engine file or not'),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.25',
            description='Confidence threshold to filter candidate detections during NMS'),
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.45',
            description='NMS IOU threshold'),
    ]

    # DNN Image Encoder parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')

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

    image_resize_node_left = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        name='image_resize_node_left',
        parameters=[{
                'output_width': network_image_width,
                'output_height': network_image_height,
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('camera_info', 'front_stereo_camera/left_rgb/camerainfo'),
            ('image', 'front_stereo_camera/left_rgb/image_raw'),
            ('resize/camera_info', 'front_stereo_camera/left_rgb/camerainfo_resize'),
            ('resize/image', 'front_stereo_camera/left_rgb/image_resize')]
    )

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': input_image_width,
            'input_image_height': input_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
            'enable_padding': 'False',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'yolov8_container',
            'dnn_image_encoder_namespace': 'yolov8_encoder',
            'image_input_topic': '/front_stereo_camera/left_rgb/image_resize',
            'camera_info_input_topic': '/front_stereo_camera/left_rgb/camerainfo_resize',
            'tensor_output_topic': '/tensor_pub',
        }.items(),
    )

    tensor_rt_node = ComposableNode(
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
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder_node',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
        }]
    )

    yolov8_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='yolov8_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[image_resize_node_left, tensor_rt_node, yolov8_decoder_node],
        output='screen'
    )

    yolov8_visualizer_node = Node(
        package='isaac_ros_yolov8',
        executable='isaac_ros_yolov8_visualizer.py',
        name='yolov8_visualizer',
        remappings=[('yolov8_encoder/resize/image', 'front_stereo_camera/left_rgb/image_resize')]

    )

    rqt_image_view_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='image_view',
        arguments=['/yolov8_processed_image'],
        parameters=[
                {'my_str': 'rgb8'},
        ]
    )

    return launch.LaunchDescription(launch_args +
                                    [yolov8_container, yolov8_encoder_launch,
                                     yolov8_visualizer_node, rqt_image_view_node])
