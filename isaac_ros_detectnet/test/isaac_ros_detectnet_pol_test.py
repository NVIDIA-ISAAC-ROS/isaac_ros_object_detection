# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Proof-Of-Life test for the Isaac ROS DetectNet package.

    1. Sets up DnnImageEncoderNode, TensorRTNode, DetectNetDecoderNode
    2. Loads a sample image and publishes it
    3. Subscribes to the relevant topics, waiting for an output from DetectNetDecoderNode
    4. Verifies that the received output sizes and encodings are correct (based on dummy model)

    Note: the data is not verified because the model is initialized with random weights
"""


import os
import pathlib
from pprint import pprint
import subprocess
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

_TEST_CASE_NAMESPACE = 'detectnet_node_test'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description for testing relevant nodes."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir_path = launch_dir_path + '/dummy_model'
    model_name = 'detectnet'
    model_version = 1
    engine_file_path = f'{model_dir_path}/{model_name}/{model_version}/model.plan'

    # Read labels from text file
    labels_file_path = f'{model_dir_path}/{model_name}/labels.txt'
    with open(labels_file_path, 'r') as fd:
        label_list = fd.read().strip().splitlines()

    # Generate engine file using tao-converter
    print('Generating engine file using tao-converter...')
    tao_converter_args = [
        '-k', '"object-detection-from-sim-pipeline"',
        '-d', '3,368,640',
        '-t', 'fp16',
        '-p', 'input_1,1x3x368x640,1x3x368x640,1x3x368x640',
        '-e', engine_file_path,
        '-o', 'output_cov/Sigmoid,output_bbox/BiasAdd',
        f'{model_dir_path}/{model_name}/1/resnet18_detector.etlt'
    ]
    tao_converter_executable = '/opt/nvidia/tao/tao-converter'
    print('Running command:\n' +
          ' '.join([tao_converter_executable] + tao_converter_args))
    start_time = time.time()
    result = subprocess.run(
        [tao_converter_executable] + tao_converter_args,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise Exception(
            f'Failed to convert with status: {result.returncode}.\n'
            f'stderr:\n' + result.stderr.decode('utf-8')
        )
    print(
        f'Finished generating engine file (took {int(time.time() - start_time)}s)')

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    centerpose_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': '640',
            'input_image_height': '368',
            'network_image_width': '640',
            'network_image_height': '368',
            'attach_to_shared_component_container': 'True',
            'component_container_name': 'detectnet_container',
            'dnn_image_encoder_namespace': IsaacROSDetectNetPipelineTest.generate_namespace(
                _TEST_CASE_NAMESPACE),
            'tensor_output_topic': 'tensor_pub',
        }.items(),
    )

    triton_node = ComposableNode(
        name='TritonNode',
        package='isaac_ros_triton',
        namespace=IsaacROSDetectNetPipelineTest.generate_namespace(
            _TEST_CASE_NAMESPACE),
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'detectnet',
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
        name='DetectNetDecoderNode',
        package='isaac_ros_detectnet',
        plugin='nvidia::isaac_ros::detectnet::DetectNetDecoderNode',
        namespace=IsaacROSDetectNetPipelineTest.generate_namespace(
            _TEST_CASE_NAMESPACE),
        parameters=[{
            'label_list': label_list,
            'enable_confidence_threshold': True,
            'enable_bbox_area_threshold': True,
            'enable_dbscan_clustering': True,
            'bounding_box_scale': 35.0,
            'bounding_box_offset': 0.0,
            'confidence_threshold': 0.6,
            'min_bbox_area': 100.0,
            'dbscan_confidence_threshold': 0.6,
            'dbscan_eps': 1.0,
            'dbscan_min_boxes': 1,
            'dbscan_enable_athr_filter': 0,
            'dbscan_threshold_athr': 0.0,
            'dbscan_clustering_algorithm': 1,
        }])

    container = ComposableNodeContainer(
        name='detectnet_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            triton_node,
            detectnet_decoder_node
        ],
        output='screen'
    )

    return IsaacROSDetectNetPipelineTest.generate_test_description(
        [container, centerpose_encoder_launch])


class IsaacROSDetectNetPipelineTest(IsaacROSBaseTest):
    """Validates a DetectNet model with randomized weights with a sample output from Python."""

    # filepath is required by IsaacROSBaseTest
    filepath = pathlib.Path(os.path.dirname(__file__))
    INIT_WAIT_SEC = 10

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_detection(self, test_folder):

        time.sleep(self.INIT_WAIT_SEC)

        self.node._logger.info('Starting to test')

        """Expect the node to segment an image."""
        self.generate_namespace_lookup(
            ['image', 'detectnet/detections', 'camera_info'], _TEST_CASE_NAMESPACE)
        image_pub = self.node.create_publisher(
            Image, self.namespaces['image'], self.DEFAULT_QOS)
        camera_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info'], self.DEFAULT_QOS)
        received_messages = {}
        detectnet_detections = self.create_logging_subscribers(
            [('detectnet/detections', Detection2DArray)],
            received_messages, accept_multiple_messages=False)

        self.generate_namespace_lookup(
            ['image', 'detectnet/detections'], _TEST_CASE_NAMESPACE)

        try:
            image = JSONConversion.load_image_from_json(
                test_folder / 'detections.json')
            image.header.stamp = self.node.get_clock().now().to_msg()
            camera_info = CameraInfo()
            camera_info.header = image.header
            camera_info.distortion_model = 'plumb_bob'
            ground_truth = open(test_folder.joinpath(
                'expected_detections.txt'), 'r')
            expected_detections = []

            for ground_detection in ground_truth.readlines():
                ground_detection_split = ground_detection.split()
                gtd = [float(ground_detection_split[1]), float(ground_detection_split[2]),
                       float(ground_detection_split[3]), float(ground_detection_split[4])]
                expected_detections.append(
                    {'width': int(gtd[0]), 'height': int(gtd[1]),
                        'center': {'x': int(gtd[2]), 'y': int(gtd[3])}}
                )

            TIMEOUT = 60
            end_time = time.time() + TIMEOUT
            done = False
            while time.time() < end_time:
                image_pub.publish(image)
                camera_info_pub.publish(camera_info)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'detectnet/detections' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, "Didn't receive output on detectnet/detections topic!")

            detection_list = received_messages['detectnet/detections'].detections

            pprint(detection_list)

            pixel_tolerance = 2.0

            self.assertGreater(len(detection_list), 0, 'No detections in detection list!')

            self.assertAlmostEqual(detection_list[0].bbox.size_x,
                                   expected_detections[0]['width'],
                                   None, 'Received incorrect width', pixel_tolerance)
            self.assertAlmostEqual(detection_list[0].bbox.size_y,
                                   expected_detections[0]['height'],
                                   None, 'Received incorrect height', pixel_tolerance)
            self.assertAlmostEqual(
                detection_list[0].bbox.center.position.x,
                expected_detections[0]['center']['x'],
                None, 'Received incorrect center', pixel_tolerance)
            self.assertAlmostEqual(
                detection_list[0].bbox.center.position.y,
                expected_detections[0]['center']['y'],
                None, 'Received incorrect center', pixel_tolerance)
        finally:
            self.node.destroy_subscription(detectnet_detections)
            self.node.destroy_publisher(image_pub)
