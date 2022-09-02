# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script prepares the pretrained detectnet model for quick deployment with Triton
# inside the Docker container

# default arguments
MODEL_LINK="https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.5/zip"
HEIGHT="632"
WIDTH="1200"
CONFIG_FILE_PATH="isaac_ros_detectnet/resources/peoplenet_config.pbtxt"

function print_parameters() {
  echo
  echo "***************************"
  echo using parameters:
  echo MODEL_LINK : $MODEL_LINK
  echo HEIGHT : $HEIGHT
  echo WIDTH : $WIDTH
  echo CONFIG_FILE_PATH : $CONFIG_FILE_PATH
  echo "***************************"
  echo
}

function setup_model() {
  # Download pre-trained ETLT model to appropriate directory
  echo Creating Directory : /tmp/models/detectnet/1
  rm -rf /tmp/models
  mkdir -p /tmp/models/detectnet/1
  cd /tmp/models/detectnet/1
  echo Downloading .etlt file from $MODEL_LINK
  echo From $MODEL_LINK
  wget --content-disposition $MODEL_LINK -O model.zip
  echo Unziping network model file .etlt
  unzip -o model.zip
  echo Converting .etlt to a TensorRT Engine Plan 
  # This is the key for the provided pretrained model
  # replace with your own key when using a model trained by any other means
  export PRETRAINED_MODEL_ETLT_KEY='tlt_encode'
  /opt/nvidia/tao/tao-converter \
    -k $PRETRAINED_MODEL_ETLT_KEY \
    -d 3,$HEIGHT,$WIDTH \
    -p input_1,1x3x$HEIGHTx$WIDTH,1x3x$HEIGHTx$WIDTH,1x3x$HEIGHTx$WIDTH \
    -t int8 \
    -e model.plan \
    -o output_cov/Sigmoid,output_bbox/BiasAdd \
    resnet34_peoplenet_int8.etlt
  echo Copying .pbtxt config file to /tmp/models/detectnet
  cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet
  cp $CONFIG_FILE_PATH \
    /tmp/models/detectnet/config.pbtxt
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
OPTIONS=m:hgt:wid:c:h
LONGOPTS=model-link:,height:,width:,config-file:,help

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
    case "$1" in
        -m|--model-link)
          MODEL_LINK="$2"
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
          CONFIG_FILE_PATH="$2"
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
