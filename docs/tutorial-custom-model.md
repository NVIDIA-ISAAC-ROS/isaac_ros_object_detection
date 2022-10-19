# Tutorial for DetectNet with a Custom Model

## Overview

This tutorial walks you through how to use a different [DetectNet Model](https://catalog.ngc.nvidia.com/models?filters=&orderBy=dateModifiedDESC&query=detectnet) with [isaac_ros_detectnet](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection) for object detection.

## Tutorial Walkthrough

1. Complete the [Quickstart section](../README.md#quickstart) in the main README.
2. Choose one of the detectnet model that is [listed here](https://catalog.ngc.nvidia.com/models?filters=&orderBy=dateModifiedDESC&query=detectnet&page=0&pageSize=25)
3. Create a config file. Use `resources/quickstart_config.pbtxt` as a template. The datatype can be found in the overview tab of the model page. The `input/dims` should be the size of the raw input images. It can be different for the same model. The `output/dims` dimensions can be calculated as `round(input_dims/max_batch_size)`. Place this config file in the `isaac_ros_detectnet/resources` directory. You can find more information about the config file [here](https://github.com/NVIDIA-AI-IOT/tao-toolkit-triton-apps/blob/main/docs/configuring_the_client.md#configuring-the-detectnet_v2-model-entry-in-the-model-repository)
4. Run the following command with the required input parameters:

    ```bash
    cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet && \
      ./scripts/setup_model.sh --height 720 --width 1280 --config-file resources/isaac_sim_config.pbtxt
    ```

    Parameters:

    `--model-link` : Get the wget link to the specific model version under the file browser tab in the page. Click on the download button on the top right and select WGET. This will copy the commend to you clipboard. Paste this in a text editor and extract only the hyperlink. eg: `https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.5/zip`

    `--model-file-name` : The name of the .etl file found in the file browser tab of the model page. eg: `resnet34_peoplenet_int8.etlt`

    `--height` : height dimension of the input image eg: `632`

    `--width` : width dimension of the input image. eg: `1200`

    `--config-file` : relative path to the config file mentioned in step 3. eg: `isaac_ros_detectnet/resources/peoplenet_config.pbtxt`
    --precision : type/precision of model found in the overview tag of the model page. eg: `int8`

    `--output-layers`: output layers seperated by commas that can be found from the txt file in the file browser tab of the model page. eg: `output_cov/Sigmoid,output_bbox/BiasAdd`
5. Replace lines 32 and 33 in [isaac_ros_detectnet.launch.py](../isaac_ros_detectnet/launch/isaac_ros_detectnet.launch.py#L32-33) with the input image dimensions
6. Run the following command:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py
    ```
