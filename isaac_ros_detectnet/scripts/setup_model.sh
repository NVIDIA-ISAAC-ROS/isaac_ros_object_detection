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

# default arguments
MODEL_LINK="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_onnx_v2.6.3/zip"
MODEL_FILE_NAME="resnet34_peoplenet.onnx"
HEIGHT="544"
WIDTH="960"
CONFIG_FILE="peoplenet_config.pbtxt"
PRECISION="int8"
MAX_BATCH_SIZE="16"

function print_parameters() {
  echo
  echo "***************************"
  echo using parameters:
  echo MODEL_LINK : $MODEL_LINK
  echo MAX_BATCH_SIZE : $MAX_BATCH_SIZE
  echo HEIGHT : $HEIGHT
  echo WIDTH : $WIDTH
  echo CONFIG_FILE : $CONFIG_FILE
  echo "***************************"
  echo
}

function check_labels_files() {
  if [[ ! -f "labels.txt" ]]
  then
    echo "Labels file does not exist with the model."
    touch labels.txt
    echo "Please enter number of labels."
    read N_LABELS
    for (( i=0; i < $N_LABELS ; i=i+1 )); do
      echo "Please enter label string"
      read label
      echo $label >> labels.txt
    done
  else
    echo "Labels file received with model."
  fi
}

# Function that extracts the last word in the three-word chain after "models/"
extract_model_name() {
    url="$1"
    # Extract the three-word chain after "models/"
    three_word_chain=$(echo $url | grep -oP 'models/\K([^/]+)(/[^/]+){2}')
    # Extract the last word
    last_word=$(echo $three_word_chain | grep -oP '[^/]+$')
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
  cd ${OUTPUT_PATH}/1
  echo Downloading .onnx file from $MODEL_LINK
  echo From $MODEL_LINK
  wget --content-disposition $MODEL_LINK -O model.zip
  echo Unziping network model file .onnx
  unzip -o model.zip
  echo Checking if labels.txt exists
  check_labels_files
  echo Converting .onnx to a TensorRT Engine Plan

  # if model doesnt have labels.txt file, then create one manually
  # create custom model
  /usr/src/tensorrt/bin/trtexec \
    --maxShapes="input_1:0":${MAX_BATCH_SIZE}x3x${HEIGHT}x${WIDTH} \
    --minShapes="input_1:0":1x3x${HEIGHT}x${WIDTH} \
    --optShapes="input_1:0":1x3x${HEIGHT}x${WIDTH} \
    --$PRECISION \
    --calib="${OUTPUT_PATH}/1/resnet34_peoplenet_int8.txt" \
    --onnx="${OUTPUT_PATH}/1/${MODEL_FILE_NAME}" \
    --saveEngine="${OUTPUT_PATH}/1/model.plan" \
    --skipInference

  echo Copying .pbtxt config file to ${OUTPUT_PATH}
  export ISAAC_ROS_DETECTNET_PATH=$(ros2 pkg prefix isaac_ros_detectnet --share)
  cp $ISAAC_ROS_DETECTNET_PATH/config/$CONFIG_FILE \
    ${OUTPUT_PATH}/config.pbtxt
  echo Completed quickstart setup
}

function show_help() {
  IFS=',' read -ra HELP_OPTIONS <<< "${LONGOPTS}"
  echo "Valid options: "
  for opt in "${HELP_OPTIONS[@]}"; do
    REQUIRED_ARG=""
    if [[ "$opt" == *":" ]]; then
      REQUIRED_ARG="${opt//:/}"
    fi
    echo -e "\t--${opt//:/} ${REQUIRED_ARG^^}"
  done
}

# Get command line arguments
OPTIONS=m:mfn:b:p:ol:h
LONGOPTS=model-link:,model-file-name:,max-batch-size:,config-file:,precision:,output-layers:,help

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
    case "$1" in
        -m|--model-link)
          MODEL_LINK="$2"
          shift 2
          ;;
        -mfn|--model-file-name)
          MODEL_FILE_NAME="$2"
          shift 2
          ;;
        -c|--config-file)
          CONFIG_FILE="$2"
          shift 2
          ;;
        -p|--precision)
          PRECISION="$2"
          shift 2
          ;;
        -b|--max-batch-size)
          MAX_BATCH_SIZE="$2"
          shift 2
          ;;
        -h|--help)
          show_help
          exit 0
          ;;
        --)
          shift
          break
          ;;
        *)
          echo "Unknown argument"
          break
          ;;
    esac
done

# Print script parameters being used
print_parameters

# Download model and copy files to appropriate location
setup_model
