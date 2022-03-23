# Isaac ROS Object Detection

![DetectNet output image showing 2 tennis balls correctly identified](resources/header-image.png "Tennis balls detected in image using DetectNet")

## Overview
This repository provides a GPU-accelerated package for object detection based on [DetectNet](https://developer.nvidia.com/blog/detectnet-deep-neural-network-object-detection-digits/). Using a trained deep-learning model and a monocular camera, the `isaac_ros_detectnet` package can detect objects of interest in an image and provide bounding boxes. [DetectNet](ttps://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_pretrained_detectnet_v2/version) is similar to other popular object detection models such as YOLOV3, FasterRCNN, SSD, and others while being efficient working with multiple object classes in large images.

Packages in this repository rely on accelerated DNN model inference using [Triton](https://github.com/triton-inference-server/server) from [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference) and a pretrained model from  [NVIDIA GPU Cloud (NGC)](https://docs.nvidia.com/ngc/) or a [custom re-trained DetectNet model](https://docs.nvidia.com/isaac/isaac/doc/tutorials/training_in_docker.html). Please note that **there is no support for the [Isaac ROS TensorRT](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt) package at this time**.

For solutions to known issues, please visit the [Troubleshooting](#troubleshooting) section below.

## System Requirements
This Isaac ROS package is designed and tested to be compatible with ROS2 Foxy on Jetson hardware, in addition to x86 systems with an NVIDIA GPU. On x86 systems, packages are only supported when run within the provided Isaac ROS Dev Docker container.

### Jetson
- [Jetson AGX Xavier or Xavier NX](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)
- [JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack)

### x86_64 (in Isaac ROS Dev Docker Container)
- Ubuntu 20.04+
- CUDA 11.4+ supported discrete GPU
- VPI 1.1.11


**Note**: For best performance on Jetson, ensure that the [power settings](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html#wwpID0EUHA) are configured appropriately.

### Docker
You need to use the Isaac ROS development Docker image from [Isaac ROS Common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common), based on the version 21.08 image from [Deep Learning Frameworks Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

You must first install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) use the Docker container development/runtime environment.

Configure `nvidia-container-runtime` as the default runtime for Docker by editing `/etc/docker/daemon.json` to include the following:
```
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
```
Then restart Docker: `sudo systemctl daemon-reload && sudo systemctl restart docker`

Run the following script in `isaac_ros_common` to build the image and launch the container on x86_64 or Jetson:

`$ scripts/run_dev.sh <optional_path>`

### Dependencies
- [isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_common)
- [isaac_ros_nvengine](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)

## Setup
1. Create a ROS2 workspace if one is not already prepared:
   ```
   mkdir -p your_ws/src
   ```
   **Note**: The workspace can have any name; this guide assumes you name it `your_ws`.

2. Clone the Isaac ROS Object Detection, Isaac ROS DNN Inference, and Isaac ROS Common package repositories to `your_ws/src`. Check that you have [Git LFS](https://git-lfs.github.com/) installed before cloning to pull down all large files:
   ```
   sudo apt-get install git-lfs

   cd your_ws/src
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
   ```

3. Start the Docker interactive workspace:
   ```
   isaac_ros_common/scripts/run_dev.sh your_ws
   ```
   After this command, you will be inside the container at `/workspaces/isaac_ros-dev`. Running this command in different terminals will attach it to the same container.

   **Note**: The rest of this README assumes that you are inside this container.

## Obtaining a Pre-Trained DetectNet Model

The easiest way to obtain a DetectNet model is to download a pre-trained one from NVIDIA's [NGC repository](https://ngc.nvidia.com). This package only supports models based on the `Detectnet_v2` architecture.

[The catalog of pre-trained models can be seen in the NGC documentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/containers/tao-toolkit-tf). Follow the instructions on the documentation for the latest version of the models and datasets you are interested in using. You can find `.etlt` files in the _File browser_ tab for each model's page, along with the _key_ to use when generating a machine-specific `.plan` file in the following steps.

Some of the [supported DetectNet models](https://catalog.ngc.nvidia.com/?filters=&orderBy=scoreDESC&query=DetectNet) from NGC:

| Model Name                                                                      | Use Case                                               |
| ------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [TrafficCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet) | Detect and track cars                                  |
| [PeopleNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplenet)         | People counting, heatmap generation, social distancing |
| [DashCamNet](https://ngc.nvidia.com/catalog/models/nvidia:tao:dashcamnet)       | Identify objects from a moving object                  |
| [FaceDetectIR](https://ngc.nvidia.com/catalog/models/nvidia:tao:facedetectir)   | Detect faces in a dark environment with IR camera      |



## Training a model using simulation

There are multiple ways to train your own `Detectnet_v2` base model. Note that you will need to update parameters, launch files, and more to match your specific trained model.

### Use the TAO toolkit launcher
The `Train and Optimize` tookit from NVIDIA has all the tools you need to prepare a dataset and re-train a detector with an easy to follow Jupyter notebook tutorial.

1. Install the `tao` command line utilities
   ```bash
     pip3 install jupyterlab
     pip3 install nvidia-pyindex
     pip3 install nvidia-tao
   ```
2. Obtain an [NGC API key](https://ngc.nvidia.com/setup/api-key).
3. Install and configure `ngc cli` from [NVIDIA NGC CLI Setup](https://ngc.nvidia.com/setup/installers/cli).
   ```bash
     wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip -o ngccli_linux.zip && chmod u+x ngc
     md5sum -c ngc.md5
     echo "export PATH=\"\$PATH:$(pwd)\"" >> ~/.bash_profile && source ~/.bash_profile
     ngc config set
   ```
4. Download the TAO cv examples to a local folder
   ```bash
     ngc registry resource download-version "nvidia/tao/cv_samples:v1.3.0"
   ```
5. Run the `DetectNet_v2` Jupyter notebook server.
   ```bash
     cd cv_samples_vv1.3.0
     jupyter-notebook --ip 0.0.0.0 --port 8888 --allow-root
   ```
6. Navigate to the DetectNet v2 notebook in `detectnet_v2/detectnet_v2.ipynb` or go to
   ```
     http://0.0.0.0:8888/notebooks/detectnet_v2/detectnet_v2.ipynb
   ```
   And follow the instructions on the tutorial.

### Training object detection in simulation

If you wish to generate training data from simulation using 3D models of the object classes you would like to detect, consider following the tutorial [Training Object detection from Simulation](https://docs.nvidia.com/isaac/isaac/doc/tutorials/training_in_docker.html).

The tutorial will use simulation to create a dataset that can then be used to train a `DetectNet_v2` based model. It's an easy to use tool with full access to customize training parameters in a Jupyter notebook.

Once you follow through the tutorial, you should have an `ETLT` file in `~/isaac-experiments/tlt-experiments/experiment_dir_final/resnet18_detector.etlt`.

Consult the spec file in `~/isaac-experiments/specs/isaac_detect_net_inference.json` for the values to use in the following section when preparing the model for usage with this package.

### Using the included dummy model for testing

In this package, you will find a pre-trained DetectNet model that was trained solely for detecting tennis balls using the described simulation method. Please use this model only for verification or exploring the pipeline.

**Note**: Do not use this tennis ball detection model in a production environment.

You can find the `ETLT` file in `isaac_ros_detectnet/test/dummy_model/detectnet/1/resnet18_detector.etlt` and use the ETLT key `"object-detection-from-sim-pipeline"`, including the double quotes.

```bash
export PRETRAINED_MODEL_ETLT_KEY=\"object-detection-from-sim-pipeline\"
```

## Model Preparation

In order to use a pre-trained DetectNet model, it needs to be processed for Triton. The following assumes that you have a pre-trained model in the current directory named `resnet18_detector.etlt`,that your Triton model repository is located at `/tmp/models` and that you want to name your model as `detectnet` with version `1`.

You should obtain an `.etlt` file and its key used for training using the methods described above along with any parameters you need for input configuration. 

The input image size is `368x640`. **This is not a standard size, since both dimensions must be divisible by 16 for DetectNet to process it.**

Please refer to your DetectNet training or to NGC for the ETLT key and other parameters. The key will be referred to as `$PRETRAINED_MODEL_ETLT_KEY`, so please ensure you set that variable or replace in the commands below.

For information on the options given to the tao-converter tool, please refer to the command line help with
```bash
  /opt/nvidia/tao/tao-converter -h
```

To prepare your model, please run the following commands:

```bash
# Create folder for our model with version number
mkdir -p /tmp/models/detectnet/1

# Create a plan file for Triton
/opt/nvidia/tao/tao-converter \
  -k $PRETRAINED_MODEL_ETLT_KEY \
  -d 3,368,640 \
  -p input_1,1x3x368x640,1x3x368x640,1x3x368x640 \
  -t fp16 \
  -e detectnet.engine \
  -o output_cov/Sigmoid,output_bbox/BiasAdd \
  resnet18_detector.etlt
# Deploy converted model to Triton
cp detectnet.engine /tmp/models/detectnet/1/model.plan
# Open an editor with the model configuration file for Triton.
# Copy the following section content into this file.
nano /tmp/models/detectnet/config.pbtxt
```

These commands will open a new file. The content of the file `/tmp/models/detectnet/config.pbtxt` must be as follows:

```
name: "detectnet"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 368, 640 ]
  }
]
output [
    {
        name: "output_bbox/BiasAdd"
        data_type: TYPE_FP32
        dims: [ 8, 23, 40 ]
    },
    {
        name: "output_cov/Sigmoid"
        data_type: TYPE_FP32
        dims: [ 2, 23, 40 ]
    }
]
dynamic_batching { }
version_policy: {
  specific {
    versions: [ 1 ]
  }
}
```

Please note that some of the values here will change depending on the way you trained your model.

1. `input[0].dims` is a vector with the following. Please note that the input image width and height should be multiples of 16 since DetectNet slices the image in a grid of 16x16 squares. If this is not the case, the bounding boxes will be off on the output and will need to be scaled.
  * The first position should be `3` for 3 RGB bytes.
  * The second position is the height of the input image in pixels.
  * The third position is the width of the input image in pixels.
2. `output[0].dims` is the bounding box output tensor size.
   * The first position is `4*C`, where `C` is the number of classes the network was trained for.
   * The second position is the number of grid rows, meaning `input image height / 16`.
   * The third position is the number of grid columns, meaning `input image width / 16`.
3. `output[1].dims` is the coverage value tensor size.
   * The first position is `C`, the number of classes the network was trained for.
   * The second position is the number of grid rows, meaning `input image height / 16`.
   * The third position is the number of grid columns, meaning `input image width / 16`.

Once you have the models folder configured, you can point the DNN inference node to load models using the `model_repository_paths` parameter, as explained in the following section.

See the [DetectNet documentation](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/detectnet_v2.html#bbox-ground-truth-generator) for more information.

## ROS2 Graph Configuration

To run the DetectNet object detection inference, the following ROS2 nodes should be set up and running:

![DetectNet output image showing 2 tennis balls correctly identified](resources/ros2_detectnet_node_setup.svg "Tennis balls detected in image using DetectNet")

1. **Isaac ROS DNN Image encoder**: This will take an image message and convert it to a tensor ([`TensorList`](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/isaac_ros_nvengine_interfaces/msg/TensorList.msg) that can be
   processed by the network.
2. **Isaac ROS DNN Inference - Triton**: This will execute the DetectNet network and take as input the tensor from the DNN Image Encoder. **Note: The [Isaac ROS TensorRT](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_tensor_rt) package is not able to perform inference with DetectNet models at this time.**
   The output will be a TensorList message containing the encoded detections
   * Use the parameters `model_name` and `model_repository_paths` to point to the model folder and set the model name. The `.plan` file should be located at `$model_repository_path/$model_name/1/model.plan`
3. **Isaac ROS Detectnet Decoder**
   This node will take the TensorList with encoded detections as input, and output Detection2DArray messages
   for each frame. See the following section for the parameters.

## Package reference
### `isaac_ros_detectnet`

#### Overview
The `isaac_ros_detectnet` package offers decoder to interpret inference results of a DetectNet_v2 model from the [`Triton Inference Server node`](https://gitlab-master.nvidia.com/isaac_ros/isaac_ros_dnn_inference/-/tree/dev/isaac_ros_triton).

#### Package Dependencies
- [isaac_ros_dnn_encoders](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_dnn_encoders)
- [isaac_ros_nvengine_interfaces](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/tree/main/isaac_ros_nvengine_interfaces)
- [isaac_ros_triton](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/tree/main/isaac_ros_triton)

#### Available Components
| Component         | Topics Subscribed                                              | Topics Published                                                                                                                                                                                                           | Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DetectNetDecoderNode` | `tensor_sub`: The tensor that represents the inferred aligned bounding boxes | `detectnet/detections`: Aligned image bounding boxes with detection class ([vision_msgs/Detection2DArray](http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/Detection2DArray.html)) | `label_names`: A list of strings with the names of the classes in the order they are used in the model. Keep the order of the list consistent with the training configuration.<br> `coverage_threshold`: A minimum coverage value for the boxes to be considered. Bounding boxes with lower value than this will not be in the output. <br> `bounding_box_scale`: The scale parameter, which should match the training configuration. `bounding_box_offset`: The bounding box parameter, which should match the training configuration.<br> `eps`: Epsilon value to use. Defaults to 0.01. <br>`min_boxes`: The minimum number of boxes to return. Defaults to 1. <br>`enable_athr_filter`: Enables the area-to-hit ratio (ATHR) filter. The ATHR is calculated as: __ATHR = sqrt(clusterArea) / nObjectsInCluster__. Defaults to 0. <br>`threshold_athr`: The `area-to-hit` ratio threshold. Defaults to 0.<br>`clustering_algorithm`: The clustering algorithm selection. Defaults to 1. (`1`: Enables DBScan clustering, `2`: Enables Hybrid clustering, resulting in more boxes that will need to be processed with NMS or other means of reducing overlapping detections. |

To see more information about how these values are used, see the [DetectNet documentation](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/detectnet_v2.html#bbox-ground-truth-generator)

### Example Code
You can use the following example Python code to set up a `detectnet_conainer` node in your application. Once you have this code, you can set up your message publisher to send an `Image` message to the DNN encoder and subscribe to the `detectnet/detections` publisher from the DetectNet decoder.

This code will not run by itself; it should be included in your application and properly started or shut down by ROS2. Use this code as a reference only.

```python

from launch_ros.actions.composable_node_container import ComposableNodeContainer
from launch_ros.descriptions.composable_node import ComposableNode

MODELS_PATH = '/tmp/models'

def generate_launch_description():
    """Generate launch description for running DetectNet inference."""
    launch_dir_path = os.path.dirname(os.path.realpath(__file__))
    config = launch_dir_path + '/../config/params.yaml'

    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': 640,
            'network_image_height': 368,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative',
            'tensor_name': 'input_tensor'
        }],
        remappings=[('encoded_tensor', 'tensor_pub')]
    )

    triton_node = ComposableNode(
        name='triton_node',
        package='isaac_ros_triton',
        plugin='isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': 'detectnet',
            'model_repository_paths': [MODELS_PATH],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'output_tensor_names': ['output_cov', 'output_bbox'],
            'output_binding_names': ['output_cov/Sigmoid', 'output_bbox/BiasAdd']            
        }])

    detectnet_decoder_node = ComposableNode(
        name='detectnet_decoder_node',
        package='isaac_ros_detectnet',
        plugin='isaac_ros::detectnet::DetectNetDecoderNode',
        parameters=[config]
    )

    container = ComposableNodeContainer(
        name='detectnet_container',
        namespace='detectnet_container',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[encoder_node, triton_node, detectnet_decoder_node],
        output='screen'
    )

    return launch.LaunchDescription([container])
```
We have provided a launch file for your convenience. This launch file will load models from `/tmp/models`, as in the instructions above.

## Running the launch files

Included in this repository is a small script that will load an image and generate a visualization using the output from DetectNet.

You can find the script in `isaac_ros_detectnet/scripts/isaac_ros_detectnet_visualizer.py`.

1. Make a models repository.
   ```bash
   mkdir -p /tmp/models/detectnet/1
   ```
2. Use the included ETLT file in `isaac_ros_detectnet/test/dummy_model/detectnet/1/resnet18_detector.etlt`.
   ```bash
   cp src/isaac_ros_object_detection/isaac_ros_detectnet/test/dummy_model/detectnet/1/resnet18_detector.etlt \
     /tmp/models/detectnet/1
   ```
3. Convert the ETLT to ``model.plan``.
   ```bash
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
   ```
4. Edit the `/tmp/models/detectnet/config.pbtxt` file and copy the contents from the [Model Preparation](#model-preparation) section.
5. Make sure that the model repository is configured correctly. You should have the following files:
   ```bash
   ls -R /tmp/models/
   /tmp/models/:
   detectnet

   /tmp/models/detectnet:
   1  config.pbtxt

   /tmp/models/detectnet/1:
   model.plan  resnet18_detector.etlt
   ```
6. Build and install the package from the root of your workspace.
   ```bash
     cd your_ws
     colcon build --symlink-install --packages-up-to isaac_ros_detectnet
     ```
7. Execute the launch file and visualizer demo node along with `rqt` to inspect the messages. Open 3 terminal windows where you will run ros commands. If you are using VSCode with the remote development plugin to connect to the development Docker container you can skip the `docker exec` command.
   1. On the first terminal run:
   ```bash
     cd your_ws
     ./scripts/run_dev.sh
     source install/setup.bash
     ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py
   ```
   2. On the second terminal run:
   ```bash
     cd your_ws
     ./scripts/run_dev.sh
     source install/setup.bash
     ros2 run isaac_ros_detectnet isaac_ros_detectnet_visualizer.py
   ```
   3. On the third terminal run:
   ```bash
     cd your_ws
     ./scripts/run_dev.sh
     source install/setup.bash
     rqt
   ```
8. The `rqt` window may need to be configured to show the graph and the image.
   Enable the `Plugins > Visualization > Image View`  window from the main menu in `rqt` and set the image topic to `/detectnet_processed_image`. You should see something like this:

   ![DetectNet output image showing a tennis ball correctly identified](resources/rqt_visualizer.png "RQT showing detection boxes of an NVIDIA Mug and a tennis ball from simulation using DetectNet")

9. To stop the demo, press `Ctrl-C` on each terminal.

## Troubleshooting
### Nodes crashed on initial launch reporting shared libraries have a file format not recognized
Many dependent shared library binary files are stored in `git-lfs`. These files need to be fetched in order for Isaac ROS nodes to function correctly.

#### Symptoms
```
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so: file format not recognized; treating as linker script
/usr/bin/ld:/workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_common/isaac_ros_nvengine/gxf/lib/gxf_jetpack46/core/libgxf_core.so:1: syntax error
collect2: error: ld returned 1 exit status
make[2]: *** [libgxe_node.so] Error 1
make[1]: *** [CMakeFiles/gxe_node.dir/all] Error 2
make: *** [all] Error 2
```
#### Solution
Run `git lfs pull` in each Isaac ROS repository you have checked out, especially `isaac_ros_common`, to ensure all of the large binary files have been downloaded.

# Updates

| Date       | Changes         |
| ---------- | --------------- |
| 2022-03-21 | Initial release |
