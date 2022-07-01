# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script prepares the pretrained detectnet model for quick deployment with Triton
# inside the Docker container for the quickstart demo

mkdir -p /tmp/models/detectnet/1
# Move pre-trained ETLT model to appropriate directory
cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection
cp isaac_ros_detectnet/test/dummy_model/detectnet/1/resnet18_detector.etlt \
  /tmp/models/detectnet/1
cd /tmp/models/detectnet/1
# This is the key for the provided pretrained model
# replace with your own key when using a model trained by any other means
export PRETRAINED_MODEL_ETLT_KEY=\"object-detection-from-sim-pipeline\"
/opt/nvidia/tao/tao-converter \
  -k $PRETRAINED_MODEL_ETLT_KEY \
  -d 3,368,640 \
  -p input_1,1x3x368x640,1x3x368x640,1x3x368x640 \
  -t fp16 \
  -e model.plan \
  -o output_cov/Sigmoid,output_bbox/BiasAdd \
  resnet18_detector.etlt
cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection
cp isaac_ros_detectnet/resources/detectnet_sample_config.pbtxt \
  /tmp/models/detectnet/config.pbtxt
