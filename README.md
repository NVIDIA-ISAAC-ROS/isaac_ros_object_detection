# Isaac ROS Object Detection

NVIDIA-accelerated, deep learned model support for object detection including DetectNet.

<div align="center"><img alt="original image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example.png/" width="300px"/>
<img alt="bounding box predictions using DetectNet" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example_bbox.png/" width="300px"/></div>

## Overview

Isaac ROS Object Detection contains ROS 2 packages to perform object
detection.
`isaac_ros_rtdetr`, `isaac_ros_detectnet`, and `isaac_ros_yolov8` each provide a method for spatial
classification using bounding boxes with an input image. Classification
is performed by a GPU-accelerated model of the appropriate architecture:

- `isaac_ros_rtdetr`: [RT-DETR models](https://nvidia-isaac-ros.github.io/concepts/object_detection/rtdetr/index.html)
- `isaac_ros_detectnet`: [DetectNet models](https://nvidia-isaac-ros.github.io/concepts/object_detection/detectnet/index.html)
- `isaac_ros_yolov8`: [YOLOv8 models](https://nvidia-isaac-ros.github.io/concepts/object_detection/yolov8/index.html)

The output prediction can be used by perception functions to
understand the presence and spatial location of an object in an image.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_nodegraph.png/" width="800px"/></a></div>

Each Isaac ROS Object Detection package is used in a graph of nodes to provide a
bounding box detection array with object classes from an input image. A
trained model of the appropriate architecture is required to produce the detection array.

Input images may need to be cropped and resized to maintain the aspect ratio and match the
input resolution of the specific object detection model; image resolution may be reduced to
improve DNN inference performance, which typically scales directly with
the number of pixels in the image. `isaac_ros_dnn_image_encoder`
provides DNN encoder utilities to process the input image into Tensors for the
object detection models.
Prediction results are decoded in model-specific ways,
often involving clustering and thresholding to group multiple detections
on the same object and reduce spurious detections.
Output is provided as a detection array with object classes.

DNNs have a minimum number of pixels that need to be visible on the
object to provide a classification prediction. If a person cannot see
the object in the image, it’s unlikely the DNN will. Reducing input
resolution to reduce compute may reduce what is detected in the image.
For example, a 1920x1080 image containing a distant person occupying 1k
pixels (64x16) would have 0.25K pixels (32x8) when downscaled by 1/2 in
both X and Y. The DNN may detect the person with the original input
image, which provides 1K pixels for the person, and fail to detect the
same person in the downscaled resolution, which only provides 0.25K
pixels for the person.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example_bboxseg.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example_bboxseg.png/" width="800px"/></a></div>

Object detection classifies a rectangle of pixels as containing an
object, whereas image segmentation provides more information and uses
more compute to produce a classification per pixel. Object detection is
used to know if, and where in a 2D image, the object exists. If a 3D
spacial understanding or size of an object in pixels is required, use
image segmentation.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                             | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                                 | Orin NX<br/><br/>                                                                                                                                                 | Orin Nano Super 8GB<br/><br/>                                                                                                                                       | x86_64 w/ RTX 4090<br/><br/>                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [RT-DETR Object Detection Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_rtdetr_benchmark/scripts/isaac_ros_rtdetr_graph.py)<br/><br/><br/>SyntheticaDETR<br/><br/> | 720p<br/><br/><br/><br/> | [56.5 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_rtdetr_graph-agx_orin.json)<br/><br/><br/>29 ms @ 30Hz<br/><br/>    | [33.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_rtdetr_graph-orin_nx.json)<br/><br/><br/>40 ms @ 30Hz<br/><br/>    | [37.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_rtdetr_graph-orin_nano.json)<br/><br/><br/>37 ms @ 30Hz<br/><br/>    | [450 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_rtdetr_graph-x86-4090.json)<br/><br/><br/>5.5 ms @ 30Hz<br/><br/>   |
| [DetectNet Object Detection Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_detectnet_benchmark/scripts/isaac_ros_detectnet_graph.py)<br/><br/><br/><br/>            | 544p<br/><br/><br/><br/> | [55.4 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-agx_orin.json)<br/><br/><br/>37 ms @ 30Hz<br/><br/> | [25.7 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-orin_nx.json)<br/><br/><br/>45 ms @ 30Hz<br/><br/> | [33.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-orin_nano.json)<br/><br/><br/>43 ms @ 30Hz<br/><br/> | [262 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_detectnet_graph-x86-4090.json)<br/><br/><br/>11 ms @ 30Hz<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_detectnet`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_detectnet/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_detectnet/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_detectnet/index.html#try-more-examples)
  * [ROS 2 Graph Configuration](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_detectnet/index.html#ros-2-graph-configuration)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_detectnet/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_detectnet/index.html#api)
* [`isaac_ros_rtdetr`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_rtdetr/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_rtdetr/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_rtdetr/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_rtdetr/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_rtdetr/index.html#api)
* [`isaac_ros_yolov8`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html#api)
  * [Usage](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_yolov8/index.html#usage)

## Latest

Update 2024-12-10: Update to be compatible with JetPack 6.1
