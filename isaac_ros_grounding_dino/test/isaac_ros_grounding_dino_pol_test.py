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

"""
Proof-Of-Life test for the Isaac ROS Grounding DINO package.

    1. Sets up DNN Image Encoder, Text Tokenizer, GroundingDinoPreprocessorNode, TensorRTNode,
       GroundingDinoDecoderNode
    2. Loads a sample image and publishes it
    3. Subscribes to the relevant topics, waiting for an output from GroundingDinoDecoderNode
    4. Verifies that the received output sizes and encodings are correct (based on dummy model)

    Note: The data is not verified because the model is initialized with random weights
"""


import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest, JSONConversion, MockModelGenerator
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image
import torch
from vision_msgs.msg import Detection2DArray


MODEL_ONNX_PATH = '/tmp/model.onnx'
MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10
MODEL_PLAN_PATH = '/tmp/model.plan'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing relevant nodes."""
    # Generate a dummy model with Grounding DINO-like I/O
    MockModelGenerator.generate(
        input_bindings=[
            MockModelGenerator.Binding('images', [-1, 3, 480, 640], torch.float32),
            MockModelGenerator.Binding('input_ids', [-1, 256], torch.int64),
            MockModelGenerator.Binding('attention_mask', [-1, 256], torch.uint8),
            MockModelGenerator.Binding('position_ids', [-1, 256], torch.int64),
            MockModelGenerator.Binding('token_type_ids', [-1, 256], torch.int64),
            MockModelGenerator.Binding('text_token_mask', [-1, 256, 256], torch.uint8),
        ],
        output_bindings=[
            MockModelGenerator.Binding('pred_logits', [-1, 900, 256], torch.float32),
            MockModelGenerator.Binding('pred_boxes', [-1, 900, 4], torch.float32),
        ],
        output_onnx_path=MODEL_ONNX_PATH
    )

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    grounding_dino_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': '640',
            'input_image_height': '480',
            'network_image_width': '640',
            'network_image_height': '480',
            'image_mean': '[0.485, 0.456, 0.406]',
            'image_stddev': '[0.229, 0.224, 0.225]',
            'final_tensor_name': 'images',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'grounding_dino_container',
            'dnn_image_encoder_namespace': IsaacROSGroundingDINOPOLTest.generate_namespace(),
            'image_input_topic': 'image',
            'camera_info_input_topic': 'camera_info',
            'tensor_output_topic': 'input_image_tensor',
            'keep_aspect_ratio': 'True'
        }.items(),
    )

    text_tokenizer_node = Node(
        package='isaac_ros_grounding_dino',
        executable='isaac_ros_grounding_dino_text_tokenizer.py',
        name='grounding_dino_text_tokenizer',
        namespace=IsaacROSGroundingDINOPOLTest.generate_namespace(),
        output='screen',
    )

    grounding_dino_preprocessor_node = ComposableNode(
        name='grounding_dino_preprocessor',
        package='isaac_ros_grounding_dino',
        plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode',
        namespace=IsaacROSGroundingDINOPOLTest.generate_namespace(),
        remappings=[('image_tensor', 'input_image_tensor')],
    )

    tensor_rt_node = ComposableNode(
        name='grounding_dino_inference',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSGroundingDINOPOLTest.generate_namespace(),
        parameters=[{
            'model_file_path': MODEL_ONNX_PATH,
            'engine_file_path': MODEL_PLAN_PATH,
            'input_tensor_names': ['images', 'input_ids', 'attention_mask',
                                   'position_ids', 'token_type_ids', 'text_token_mask'],
            'input_binding_names': ['images', 'input_ids', 'attention_mask',
                                    'position_ids', 'token_type_ids', 'text_token_mask'],
            'output_binding_names': ['pred_logits', 'pred_boxes'],
            'output_tensor_names': ['scores', 'boxes'],
            'verbose': False,
            'force_engine_update': True
        }]
    )

    grounding_dino_decoder_node = ComposableNode(
        name='grounding_dino_decoder',
        package='isaac_ros_grounding_dino',
        plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode',
        namespace=IsaacROSGroundingDINOPOLTest.generate_namespace(),
        parameters=[{
            'confidence_threshold': 0.5,
            'image_width': 640,
            'image_height': 480,
        }]
    )

    container = ComposableNodeContainer(
        name='grounding_dino_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            grounding_dino_preprocessor_node,
            tensor_rt_node,
            grounding_dino_decoder_node
        ],
        output='screen'
    )

    return IsaacROSGroundingDINOPOLTest.generate_test_description([
        text_tokenizer_node,
        container,
        grounding_dino_encoder_launch
    ])


class IsaacROSGroundingDINOPOLTest(IsaacROSBaseTest):
    """Validates that the inference pipeline produces outputs using a dummy model."""

    # filepath is required by IsaacROSBaseTest
    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    @IsaacROSBaseTest.for_each_test_case()
    def test_object_detection(self, test_folder):
        """Expect the node to produce detection array given image."""
        self.node._logger.info(f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(MODEL_PLAN_PATH):
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

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                timestamp = self.node.get_clock().now().to_msg()
                image.header.stamp = timestamp
                camera_info.header.stamp = timestamp

                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'detections_output' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, 'Did not receive output on detections_output topic!')

        finally:
            self.node.destroy_subscription(subs)
            self.node.destroy_publisher(image_pub)
            self.node.destroy_publisher(camera_info_pub)
