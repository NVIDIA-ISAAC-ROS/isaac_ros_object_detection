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
MODEL_LINK="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.5/zip"
MODEL_FILE_NAME="resnet34_peoplenet_int8.etlt"
HEIGHT="632"
WIDTH="1200"
CONFIG_FILE="peoplenet_config.pbtxt"
PRECISION="int8"
OUTPUT_LAYERS="output_cov/Sigmoid,output_bbox/BiasAdd"

function print_parameters() {
  echo
  echo "***************************"
  echo using parameters:
  echo MODEL_LINK : $MODEL_LINK
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
  # Download pre-trained ETLT model to appropriate directory
  # Extract model names from URLs
  model_name_from_model_link=$(extract_model_name "$MODEL_LINK")
  echo "Model name from model link: $model_name_from_model_link"
  echo Creating Directory : ${ISAAC_ROS_WS}/isaac_ros_assets/models/$model_name_from_model_link/1
  rm -rf ${ISAAC_ROS_WS}/isaac_ros_assets/models
  mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/$model_name_from_model_link/1
  cd ${ISAAC_ROS_WS}/isaac_ros_assets/models/$model_name_from_model_link/1
  echo Downloading .etlt file from $MODEL_LINK
  echo From $MODEL_LINK
  wget --content-disposition $MODEL_LINK -O model.zip
  echo Unziping network model file .etlt
  unzip -o model.zip
  echo Checking if labels.txt exists 
  check_labels_files
  echo Converting .etlt to a TensorRT Engine Plan 
  # This is the key for the provided pretrained model
  # replace with your own key when using a model trained by any other means
  export PRETRAINED_MODEL_ETLT_KEY='tlt_encode'
  # if model doesnt have labels.txt file, then create one manually
  # create custom model
  /opt/nvidia/tao/tao-converter \
    -k $PRETRAINED_MODEL_ETLT_KEY \
    -d 3,$HEIGHT,$WIDTH \
    -p input_1,1x3x$HEIGHTx$WIDTH,1x3x$HEIGHTx$WIDTH,1x3x$HEIGHTx$WIDTH \
    -t $PRECISION \
    -e model.plan \
    -o $OUTPUT_LAYERS\
    $MODEL_FILE_NAME
  echo Copying .pbtxt config file to ${ISAAC_ROS_WS}/isaac_ros_assets/models/$model_name_from_model_link
  cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet
  export ISAAC_ROS_DETECTNET_PATH=$(ros2 pkg prefix isaac_ros_detectnet --share)
  cp $ISAAC_ROS_DETECTNET_PATH/config/$CONFIG_FILE \
    ${ISAAC_ROS_WS}/isaac_ros_assets/models/$model_name_from_model_link/config.pbtxt
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
OPTIONS=m:mfn:hgt:wid:c:p:ol:h
LONGOPTS=model-link:,model-file-name:,height:,width:,config-file:,precision:,output-layers:,help

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
        --height)
          HEIGHT="$2"
          shift 2
          ;;
        --width)
          WIDTH="$2"
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
        -ol|--output-layers)
          OUTPUT_LAYERS="$2"
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
