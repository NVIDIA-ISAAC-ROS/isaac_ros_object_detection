#!/bin/bash
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

# Download and convert RT-DETR models to TensorRT engines
# Models will be stored in the isaac_ros_assets directory
# Setup paths
set -e
if [ -n "$TENSORRT_COMMAND" ]; then
  # If a custom tensorrt is used, ensure it's lib directory is added to the LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(readlink -f $(dirname ${TENSORRT_COMMAND})/../../../lib/x86_64-linux-gnu/)"
fi
if [ -z "$ISAAC_ROS_WS" ] && [ -n "$ISAAC_ROS_ASSET_MODEL_PATH" ]; then
  ISAAC_ROS_WS="$(readlink -f $(dirname ${ISAAC_ROS_ASSET_MODEL_PATH})/../../..)"
fi
ASSET_NAME="synthetica_detr"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/synthetica_detr"
ASSET_DIR="${MODELS_DIR}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/sdetr_grasp.plan"


source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}"

# Create directories if they don't exist
mkdir -p ${MODELS_DIR}

# Download SyntheticaDETR model
SYNTHETICA_DETR_URL="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/synthetica_detr/versions/1.0.0_onnx/files/sdetr_grasp.onnx"
SYNTHETICA_DETR_ONNX="${MODELS_DIR}/sdetr_grasp.onnx"
SYNTHETICA_DETR_ENGINE="${MODELS_DIR}/sdetr_grasp.plan"

wget -nv -O "${SYNTHETICA_DETR_ONNX}" "${SYNTHETICA_DETR_URL}"

${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --onnx=${SYNTHETICA_DETR_ONNX} \
    --saveEngine=${SYNTHETICA_DETR_ENGINE}
