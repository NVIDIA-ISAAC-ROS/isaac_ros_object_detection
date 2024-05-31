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

"""
Proof-Of-Life test for the Isaac ROS RT-DETR package.

    1. Sets up DnnImageEncoderNode, RtDetrPreprocessorNode, TensorRTNode, RtDetrDecoderNode
    2. Loads a sample image and publishes it
    3. Subscribes to the relevant topics, waiting for an output from RtDetrDecoderNode
    4. Verifies that the received output sizes and encodings are correct (based on dummy model)

    Note: the data is not verified because the model is initialized with random weights
"""


import os
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

MODEL_FILE_NAME = 'rtdetr_dummy_pol.onnx'

MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10
MODEL_PATH = '/tmp/rtdetr_dummy_pol.plan'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing relevant nodes."""
    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
            'input_width': 640,
            'input_height': 480,
            'output_width': 640,
            'output_height': 640,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }]
    )

    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
            'output_image_width': 640,
            'output_image_height': 640,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[(
            'image', 'resize/image'
        )]
    )

    image_format_node = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': 640,
                'image_height': 640
        }],
        remappings=[
            ('image_raw', 'padded_image'),
            ('image', 'image_rgb')]
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'image_rgb'),
            ('tensor', 'normalized_tensor'),
        ]
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
            'input_tensor_shape': [640, 640, 3]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, 640, 640],
            'output_tensor_shape': [1, 3, 640, 640]
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        remappings=[
            ('encoded_tensor', 'reshaped_tensor')
        ],
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace(),
        parameters=[{
            'model_file_path': f'{os.path.dirname(__file__)}/dummy_model/{MODEL_FILE_NAME}',
            'engine_file_path': MODEL_PATH,
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
        namespace=IsaacROSRtDetrPOLTest.generate_namespace()
    )

    container = ComposableNodeContainer(
        name='rtdetr_container',
        namespace='rtdetr_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            resize_node, pad_node, image_format_node,
            image_to_tensor_node, interleave_to_planar_node, reshape_node,
            rtdetr_preprocessor_node, tensor_rt_node, rtdetr_decoder_node
        ],
        output='screen'
    )

    return IsaacROSRtDetrPOLTest.generate_test_description([container])


class IsaacROSRtDetrPOLTest(IsaacROSBaseTest):
    """Validates that the inference pipeline produces outputs using a RT-DETR-like dummy model."""

    # filepath is required by IsaacROSBaseTest
    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    @IsaacROSBaseTest.for_each_test_case()
    def test_object_detection(self, test_folder):
        """Expect the node to produce detection array given image."""
        self.node._logger.info(f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(MODEL_PATH):
            time_diff = time.time() - start_time
            if time_diff > MODEL_GENERATION_TIMEOUT_SEC:
                self.fail('Model generation timed out')
            if time_diff > wait_cycles*10:
                self.node._logger.info(
                    f'Waiting for model generation to finish... ({time_diff:.0f}s passed)')
                wait_cycles += 1
            time.sleep(1)

        self.node._logger.info(
            f'Model generation was finished (took {(time.time() - start_time)}s)')

        received_messages = {}

        self.generate_namespace_lookup(['image', 'camera_info', 'detections_output'])

        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)
        subs = self.create_logging_subscribers(
            [('detections_output', Detection2DArray)], received_messages)

        try:
            image = JSONConversion.load_image_from_json(
                test_folder / 'image.json')
            camera_info = JSONConversion.load_camera_info_from_json(
                test_folder / 'camera_info.json')
            timestamp = self.node.get_clock().now().to_msg()
            image.header.stamp = timestamp
            camera_info.header.stamp = timestamp

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'detections_output' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, "Didn't receive output on detections_output topic!")

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
