# Isaac ROS Object Detection

<div align="center"><img alt="original image" src="resources/isaac_ros_object_detection_example.png" width="300px"/> <img alt="bounding box predictions using DetectNet" src="resources/isaac_ros_object_detection_example_bbox.png" width="300px"/></div>

## Overview

Isaac ROS Object Detection contains an ROS 2 package to perform object detection. `isaac_ros_detectnet` provides a method for spatial classification using bounding boxes with an input image. Classification is performed by a GPU-accelerated [DetectNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_detectnet_v2) model. The output prediction can be used by perception functions to understand the presence and spatial location of an object in an image.

<div align="center"><img alt="graph of nodes using DetectNet" src="resources/isaac_ros_object_detection_nodegraph.png" width="500px"/></div>

`isaac_ros_detectnet` is used in a graph of nodes to provide a bounding box detection array with object classes from an input image. A [DetectNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_detectnet_v2) model is required to produce the detection array. Input images may need to be cropped and resized to maintain the aspect ratio and match the input resolution of DetectNet; image resolution may be reduced to improve DNN inference performance, which typically scales directly with the number of pixels in the image. `isaac_ros_dnn_image_encoder` provides a DNN encoder to process the input image into Tensors for the DetectNet model. Prediction results are clustered in the DNN decoder to group multiple detections on the same object. Output is provided as a detection array with object classes.

DNNs have a minimum number of pixels that need to be visible on the object to provide a classification prediction. If a person cannot see the object in the image, it’s unlikely the DNN will. Reducing input resolution to reduce compute may reduce what is detected in the image. For example, a 1920x1080 image containing a distant person occupying 1k pixels (64x16) would have 0.25K pixels (32x8) when downscaled by 1/2 in both X and Y. The DNN may detect the person with the original input image, which provides 1K pixels for the person, and fail to detect the same person in the downscaled resolution, which only provides 0.25K pixels for the person.

> **Note**: DetectNet is similar to other popular object detection models such as YOLOV3, FasterRCNN, and SSD, while being efficient at detecting multiple object classes in large images.

<div align="center"><img alt="comparison of bounding box detection to segmentation" src="resources/isaac_ros_object_detection_example_bboxseg.png" width="300px"/></div>

Object detection classifies a rectangle of pixels as containing an object, whereas image segmentation provides more information and uses more compute to produce a classification per pixel. Object detection is used to know if, and where in a 2D image, the object exists. If a 3D spacial understanding or size of an object in pixels is required, use image segmentation.  

### DNN Models

To perform DNN inferencing a DNN model is required. NGC provides [DetectNet pre-trained models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_detectnet_v2) for use in your robotics application. Using [TAO](https://developer.nvidia.com/tao-toolkit) these pre-trained models can be fine-tuned for your application.

### Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

### Performance

The performance results of benchmarking the prepared pipelines in this package on supported platforms are below:

| Sample Graph                                                                                                                                        | Input Size | AGX Orin                                                                                                                                       | Orin NX                                                                                                                                      | Orin Nano 8GB                                                                                                                                        | x86_64 w/ RTX 4060 Ti |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| [DetectNet Object Detection Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_detectnet_graph.py) | 544p       | [252 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-agx_orin.json)<br>8.7 ms | [110 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-orin_nx.json)<br>13 ms | [77.9 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-orin_nano_emul.json)<br>18 ms | --                    |


> **Note**: These numbers are reported with defaults parameter values found in [params.yaml](./isaac_ros_detectnet/config/params.yaml).

These data have been collected per the methodology described [here](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/performance-summary.md#methodology).

### ROS 2 Graph Configuration

To run the DetectNet object detection inference, the following ROS 2 nodes should be set up and running:

![DetectNet output image showing 2 tennis balls correctly identified](resources/ros2_detectnet_node_setup.svg "Tennis balls detected in image using DetectNet")

1. **Isaac ROS DNN Image encoder**: This will take an image message and convert it to a tensor ([`TensorList`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) that can be
   processed by the network.
2. **Isaac ROS DNN Inference - Triton**: This will execute the DetectNet network and take as input the tensor from the DNN Image Encoder.
    > **Note**: The [Isaac ROS TensorRT](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt) package is not able to perform inference with DetectNet models at this time.
  
   The output will be a TensorList message containing the encoded detections. Use the parameters `model_name` and `model_repository_paths` to point to the model folder and set the model name. The `.plan` file should be located at `$model_repository_path/$model_name/1/model.plan`
3. **Isaac ROS Detectnet Decoder**: This node will take the TensorList with encoded detections as input, and output `Detection2DArray` messages for each frame. See the following section for the parameters.

## Table of Contents

- [Isaac ROS Object Detection](#isaac-ros-object-detection)
  - [Overview](#overview)
    - [DNN Models](#dnn-models)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
    - [Performance](#performance)
    - [ROS 2 Graph Configuration](#ros-2-graph-configuration)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try More Examples](#try-more-examples)
    - [Customize your Dev Environment](#customize-your-dev-environment)
  - [Package Reference](#package-reference)
    - [`isaac_ros_detectnet`](#isaac_ros_detectnet)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [Deep Learning Troubleshooting](#deep-learning-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2023-05-25: Performance improvements.

## Supported Platforms

This package is designed and tested to be compatible with ROS 2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS 2 earlier than Humble are **not** supported. This package depends on specific ROS 2 implementation features that were only introduced beginning with the Humble release.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                          | Notes                                                                                                                                                                                   |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack)                                                    | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.8](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note**: All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).
2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ```

3. Pull down a ROS Bag of sample data:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet && \
      git lfs pull -X "" -I "resources/rosbags"
    ```

4. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

5. Inside the container, build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

6. (Optional) Run tests to verify complete and correct installation:

    ```bash
    colcon test --executor sequential
    ```

7. Run the quickstart setup script which will download the [PeopleNet Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) from NVIDIA GPU Cloud(NGC)

    ```bash
    cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet && \
      ./scripts/setup_model.sh --height 632 --width 1200 --config-file resources/quickstart_config.pbtxt
    ```

8. Run the following launch file to spin up a demo of this package:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      ros2 launch isaac_ros_detectnet isaac_ros_detectnet_quickstart.launch.py
    ```

9. Visualize and validate the output of the package in the `rqt_image_view` window. After about a minute, your output should look like this:

    ![DetectNet output image showing a tennis ball correctly identified](resources/rqt_visualizer.png "RQT showing detection boxes of an NVIDIA Mug and a tennis ball from simulation using DetectNet")

## Next Steps

### Try More Examples

To continue your exploration, check out the following suggested examples:

- [Tutorial with Isaac Sim](docs/tutorial-isaac-sim.md)
- [Tutorial with Custom Model](docs/tutorial-custom-model.md) For more info click [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/model-preparation.md)

This package only supports models based on the `Detectnet_v2` architecture. Some of the [supported DetectNet models](https://catalog.ngc.nvidia.com/?filters=&orderBy=scoreDESC&query=DetectNet) from NGC:

| Model Name                                                                      | Use Case                                               |
| ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [TrafficCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet) | Detect and track cars                                  |
| [PeopleNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplenet)         | People counting, heatmap generation, social distancing |
| [DashCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:dashcamnet)       | Identify objects from a moving object                  |
| [FaceDetectIR](https://ngc.nvidia.com/catalog/models/nvidia:tao:facedetectir)   | Detect faces in a dark environment with IR camera      |

### Customize your Dev Environment

To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/modify-dockerfile.md).

## Package Reference

### `isaac_ros_detectnet`

#### Usage

```bash
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py label_list:=<list of labels> enable_confidence_threshold:=<enable confidence thresholding> enable_bbox_area_threshold:=<enable bbox size thresholding> enable_dbscan_clustering:=<enable dbscan clustering> confidence_threshold:=<minimum confidence value> min_bbox_area:=<minimum bbox area value> dbscan_confidence_threshold:=<minimum confidence for dbscan algorithm> dbscan_eps:=<epsilon distance> dbscan_min_boxes:=<minimum returned boxes> dbscan_enable_athr_filter:=<area-to-hit-ratio filter> dbscan_threshold_athr:=<area-to-hit ratio threshold> dbscan_clustering_algorithm:=<choice of clustering algorithm> bounding_box_scale:=<bounding box normalization value> bounding_box_offset:=<XY offset for bounding box>
```

#### ROS Parameters

| ROS Parameter                 | Type       | Default                     | Description                                                                                                                                                                                                             |
| ----------------------------- | ---------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label_list`                  | `string[]` | `{"person", "bag", "face"}` | The list of labels. These are loaded from labels.txt(downloaded with the model)                                                                                                                                         |
| `confidence_threshold`        | `double`   | `0.35`                      | The min value of confidence used to threshold detections before clustering                                                                                                                                              |
| `min_bbox_area`               | `double`   | `100`                       | The min value of bouding box area used to threshold detections before clustering                                                                                                                                        |
| `dbscan_confidence_threshold` | `double`   | `0.35`                      | Holds the epsilon to control merging of overlapping boxes. Refer to OpenCV groupRectangles and DBSCAN documentation for more information on epsilon.                                                                    |
| `dbscan_eps`                  | `double`   | `0.7`                       | Holds the epsilon to control merging of overlapping boxes. Refer to OpenCV groupRectangles and DBSCAN documentation for more information on epsilon.                                                                    |
| `dbscan_min_boxes`            | `int`      | `1`                         | The minimum number of boxes to return.                                                                                                                                                                                  |
| `dbscan_enable_athr_filter`   | `int`      | `0`                         | Enables the area-to-hit ratio (ATHR) filter. The ATHR is calculated as: **ATHR = sqrt(clusterArea) / nObjectsInCluster.**                                                                                               |
| `dbscan_threshold_athr`       | `double`   | `0.0`                       | The `area-to-hit` ratio threshold.                                                                                                                                                                                      |
| `dbscan_clustering_algorithm` | `int`      | `1`                         | The clustering algorithm selection. (`1`: Enables DBScan clustering, `2`: Enables Hybrid clustering, resulting in more boxes that will need to be processed with NMS or other means of reducing overlapping detections. |
| `bounding_box_scale`          | `double`   | `35.0`                      | The scale parameter, which should match the training configuration.                                                                                                                                                     |
| `bounding_box_offset`         | `double`   | `0.0`                       | Bounding box offset for both X and Y dimensions.                                                                                                                                                                        |

#### ROS Topics Subscribed

| ROS Topic    | Interface                                                                                                                                                         | Description                                                     |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `tensor_sub` | [isaac_ros_tensor_list_interfaces/TensorList](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_tensor_list_interfaces/msg/TensorList.msg) | The tensor that represents the inferred aligned bounding boxes. |

#### ROS Topics Published

| ROS Topic              | Interface                                                                                                        | Description                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `detectnet/detections` | [vision_msgs/Detection2DArray](https://github.com/ros-perception/vision_msgs/blob/ros2/msg/Detection2DArray.msg) | Aligned image bounding boxes with detection class. |

## Troubleshooting

### Isaac ROS Troubleshooting

For solutions to problems with Isaac ROS, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

### Deep Learning Troubleshooting

For solutions to problems with using DNN models, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/troubleshooting.md).

## Updates

| Date       | Changes                                                                               |
| ---------- | ------------------------------------------------------------------------------------- |
| 2023-05-25 | Performance improvements                                                              |
| 2023-04-05 | Source available GXF extensions                                                       |
| 2022-10-19 | Updated OSS licensing                                                                 |
| 2022-08-31 | Update to use NITROS for improved performance and to be compatible with JetPack 5.0.2 |
| 2022-06-30 | Support for ROS 2 Humble and miscellaneous bug fixes                                  |
| 2022-03-21 | Initial release                                                                       |
