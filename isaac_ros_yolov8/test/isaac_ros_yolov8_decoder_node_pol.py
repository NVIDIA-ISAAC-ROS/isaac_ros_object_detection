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
Proof-Of-Life test for the Isaac ROS YOLOV8 Decoder Node package.

    1. Sets up DnnImageEncoderNode, TensorRTNode
    2. Loads a sample image and publishes it
    3. Subscribes to the relevant topics, waiting for an output from YoloV8DecoderNode
    4. Verifies that the output is recieved (based on dummy model)

    Note: the data is not verified because the model is initialized with random weights
"""


import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

_TEST_CASE_NAMESPACE = 'yolov8_decoder_node_test'

MODEL_FILE_NAME = 'dummy_yolov8s.onnx'
MODEL_GENERATION_TIMEOUT_SEC = 300
INIT_WAIT_SEC = 10
engine_file_path = '/tmp/dummy_yolov8s.plan'
input_tensor_names = ['input_tensor']
input_binding_names = ['images']
output_tensor_names = ['output_tensor']
output_binding_names = ['output0']


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing relevant nodes."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir_path = launch_dir_path + '/dummy_model'
    model_name = 'yolov8'
    model_file_path = f'{model_dir_path}/{model_name}/dummy_yolov8s.onnx'
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    yolov8_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': '640',
            'input_image_height': '640',
            'network_image_width': '640',
            'network_image_height': '640',
            'image_mean': str([0.5, 0.6, 0.25]),
            'image_stddev': str([0.25, 0.8, 0.5]),
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'tensor_rt_container',
            'dnn_image_encoder_namespace': IsaacROSYoloV8POLTest.generate_namespace(
                _TEST_CASE_NAMESPACE),
            'tensor_output_topic': 'tensor_pub',
        }.items(),
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        namespace=IsaacROSYoloV8POLTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'output_tensor_names': output_tensor_names,
            'input_tensor_shape': [3, 640, 640],
            'output_tensor_shape': [1, 3, 640, 640]
        }],
        remappings=[
            ('tensor_input_topic', 'tensor_sub')
        ],
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        namespace=IsaacROSYoloV8POLTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'output_binding_names': output_binding_names,
            'output_tensor_names': output_tensor_names,
            'input_tensor_names': input_tensor_names,
            'input_binding_names': input_binding_names,
            'verbose': False,
            'force_engine_update': False
        }]
    )

    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder_node',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        namespace=IsaacROSYoloV8POLTest.generate_namespace(_TEST_CASE_NAMESPACE),
        parameters=[{
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
        }]
    )

    tensor_rt_container = ComposableNodeContainer(
        name='tensor_rt_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            tensor_rt_node, yolov8_decoder_node, reshape_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO'],
        namespace=''
    )

    return IsaacROSYoloV8POLTest.generate_test_description(
        [tensor_rt_container, yolov8_encoder_launch])


class IsaacROSYoloV8POLTest(IsaacROSBaseTest):
    """Validates that the inference pipeline produces outputs using a Yolov8 model."""

    # filepath is required by IsaacROSBaseTest
    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    @IsaacROSBaseTest.for_each_test_case()
    def test_yolov8(self, test_folder):
        """Expect the node to produce detection array given image."""
        self.node._logger.info(f'Generating model (timeout={MODEL_GENERATION_TIMEOUT_SEC}s)')
        start_time = time.time()
        wait_cycles = 1
        while not os.path.isfile(engine_file_path):
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

        self.generate_namespace_lookup(
            ['image', 'detections_output', 'camera_info'], _TEST_CASE_NAMESPACE)

        image_publisher = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)

        received_messages = {}

        yolov8_detections = self.create_logging_subscribers(
            [('detections_output', Detection2DArray)],
            received_messages, accept_multiple_messages=False)

        self.generate_namespace_lookup(['image', 'detections_output'], _TEST_CASE_NAMESPACE)

        try:
            image = JSONConversion.load_image_from_json(test_folder / 'image.json')
            timestamp = self.node.get_clock().now().to_msg()
            image.header.stamp = timestamp
            camera_info = CameraInfo()
            camera_info.header = image.header
            camera_info.distortion_model = 'plumb_bob'

            TIMEOUT = 10
            end_time = time.time() + TIMEOUT
            done = False
            while time.time() < end_time:
                image_publisher.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=1.0)

                if 'detections_output' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, "Didn't receive output on detections_output topic!")

        finally:
            self.node.destroy_subscription(yolov8_detections)
            self.node.destroy_publisher(image_publisher)
