#!/bin/bash

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script prepares the pretrained detectnet model for quick deployment with Triton
# inside the Docker container

if [ -n "$TENSORRT_COMMAND" ]; then
  # If a custom tensorrt is used, ensure it's lib directory is added to the LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(readlink -f $(dirname ${TENSORRT_COMMAND})/../../../lib/$(uname -p)-linux-gnu/)"
fi
if [ -z "$ISAAC_ROS_WS" ] && [ -n "$ISAAC_ROS_ASSET_MODEL_PATH" ]; then
  ISAAC_ROS_WS="$(readlink -f $(dirname ${ISAAC_ROS_ASSET_MODEL_PATH})/../../../..)"
fi

# default arguments
ASSET_NAME="peoplenet"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet?version=deployable_quantized_onnx_v2.6.3"
ASSET_DIR="${MODELS_DIR}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/1/model.plan"

MODEL_LINK="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=resnet34_peoplenet.onnx"
MODEL_FILE_NAME="resnet34_peoplenet.onnx"
HEIGHT="544"
WIDTH="960"
LABELS_LINK="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/deployable_quantized_onnx_v2.6.3/files?redirect=true&path=labels.txt"
CONFIG_FILE="peoplenet_config.pbtxt"
PRECISION="int8"
MAX_BATCH_SIZE="16"


function print_parameters() {
  echo
  echo "***************************"
  echo using parameters:
  echo MODEL_LINK : $MODEL_LINK
  echo LABELS_LINK : $LABELS_LINK
  echo MAX_BATCH_SIZE : $MAX_BATCH_SIZE
  echo HEIGHT : $HEIGHT
  echo WIDTH : $WIDTH
  echo CONFIG_FILE : $CONFIG_FILE
  echo "***************************"
  echo
}

function check_labels_files() {
  if [[ ! -f "${1:-.}/labels.txt" ]]
  then
    echo "Labels file does not exist with the model."
    touch ${1:-.}/labels.txt
    echo "Please enter number of labels."
    read N_LABELS
    for (( i=0; i < $N_LABELS ; i=i+1 )); do
      echo "Please enter label string"
      read label
      echo $label >> ${1:-.}/labels.txt
    done
  else
    echo "Labels file received with model."
  fi
}

# Function that extracts the last word in the three-word chain after "models/"
extract_model_name() {
    url="$1"
    # Extract the three-word chain after "models/"
    five_word_chain=$(echo $url | grep -oP 'models/\K([^/]+)(/[^/]+){4}')
    # Extract the last word
    last_word=$(echo $five_word_chain | grep -oP '[^/]+$')
    echo $last_word
}

function setup_model() {
  # Download pre-traine ONNX model to appropriate directory
  # Extract model names from URLs
  model_name_from_model_link=$(extract_model_name "$MODEL_LINK")
  OUTPUT_PATH=${ISAAC_ROS_WS}/isaac_ros_assets/models/$model_name_from_model_link
  echo "Model name from model link: $model_name_from_model_link"
  echo Creating Directory : "${OUTPUT_PATH}/1"
  rm -rf ${OUTPUT_PATH}
  mkdir -p ${OUTPUT_PATH}/1
  echo Downloading .onnx file from $MODEL_LINK
  echo From $MODEL_LINK
  isaac_ros_common_download_asset --url "${MODEL_LINK}" --output-path "${OUTPUT_PATH}/1/${MODEL_FILE_NAME}" --cache-path "${ISAAC_ROS_DETECTNET_MODEL}"
  MODEL_DOWNLOAD_RESULT=$?
  echo Checking if labels.txt exists
  echo From $LABELS_LINK
  isaac_ros_common_download_asset --url "${LABELS_LINK}" --output-path "${OUTPUT_PATH}/1/labels.txt" --cache-path "${ISAAC_ROS_DETECTNET_LABELS}"
  LABELS_DOWNLOAD_RESULT=$?

  if [[ -n ${ISAAC_ROS_ASSETS_TEST} ]]; then
    if [[ ${MODEL_DOWNLOAD_RESULT} -ne 0 ]]; then
      echo "ERROR: Remote model does not match cached model."
      exit 1
    fi
    if [[ ${LABELS_DOWNLOAD_RESULT} -ne 0 ]]; then
      echo "ERROR: Remote labels do not match cached labels."
      exit 1
    fi
    exit 0
  elif [[ ${MODEL_DOWNLOAD_RESULT} -ne 0 ]]; then
    echo "ERROR: Failed to download model."
    exit 1
  elif [[ ${LABELS_DOWNLOAD_RESULT} -ne 0 ]]; then
    echo "ERROR: Failed to download labels."
    exit 1
  fi

  check_labels_files "${OUTPUT_PATH}/1"

  echo Converting .onnx to a TensorRT Engine Plan

  # if model doesnt have labels.txt file, then create one manually
  # create custom model
  ${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --maxShapes="input_1:0":${MAX_BATCH_SIZE}x3x${HEIGHT}x${WIDTH} \
    --minShapes="input_1:0":1x3x${HEIGHT}x${WIDTH} \
    --optShapes="input_1:0":1x3x${HEIGHT}x${WIDTH} \
    --$PRECISION \
    --calib="${OUTPUT_PATH}/1/resnet34_peoplenet_int8.txt" \
    --onnx="${OUTPUT_PATH}/1/${MODEL_FILE_NAME}" \
    --saveEngine="${OUTPUT_PATH}/1/model.plan" \
    --skipInference

  echo Copying .pbtxt config file to ${OUTPUT_PATH}
  if [ -n "$ISAAC_ROS_DETECTNET_CONFIG" ]; then
    cp $ISAAC_ROS_DETECTNET_CONFIG ${OUTPUT_PATH}/config.pbtxt
  else
    export ISAAC_ROS_DETECTNET_PATH=$(ros2 pkg prefix isaac_ros_detectnet --share)
    cp $ISAAC_ROS_DETECTNET_PATH/config/$CONFIG_FILE \
      ${OUTPUT_PATH}/config.pbtxt
  fi
  echo Completed quickstart setup
}

# Parse script-specific args; unknown args (e.g. --show-eula, --no-cache) are
# collected in PASSTHRU_ARGS and forwarded to the EULA helper via set --.
PASSTHRU_ARGS=()
while (( "$#" )); do
    case "$1" in
        -h|--help)
          echo "Usage: $0 [--show-eula|--print-install-paths|--no-cache]"
          echo "  --model-link MODEL_LINK         Override model download URL"
          echo "  --model-file-name MODEL_FILE    Override model file name"
          echo "  --config-file CONFIG_FILE       Override Triton config file (default: peoplenet_config.pbtxt)"
          echo "  --precision PRECISION           Override precision (default: int8)"
          echo "  --max-batch-size BATCH_SIZE     Override max batch size (default: 16)"
          exit 0
          ;;
        -m|--model-link)
          MODEL_LINK="$2"; shift 2
          ;;
        --model-file-name)
          MODEL_FILE_NAME="$2"; shift 2
          ;;
        -c|--config-file)
          CONFIG_FILE="$2"; shift 2
          ;;
        -p|--precision)
          PRECISION="$2"; shift 2
          ;;
        -b|--max-batch-size)
          MAX_BATCH_SIZE="$2"; shift 2
          ;;
        *)
          # Forward unrecognized args (--show-eula, --eula, --print-install-paths,
          # --no-cache) to the EULA helper script.
          PASSTHRU_ARGS+=("$1"); shift
          ;;
    esac
done

# Restore forwarded args for the EULA helper to consume.
set -- "${PASSTHRU_ARGS[@]}"
source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}"

# Print script parameters being used
print_parameters

# Download model and copy files to appropriate location
setup_model
