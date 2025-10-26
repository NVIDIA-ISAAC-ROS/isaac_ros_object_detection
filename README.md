# Isaac ROS Object Detection

NVIDIA-accelerated, deep learned model support for object detection including DetectNet.

<div align="center"><img alt="original image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example.png/" width="300px"/>
<img alt="bounding box predictions using DetectNet" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example_bbox.png/" width="300px"/></div>

## Overview

Isaac ROS Object Detection contains ROS 2 packages to perform object
detection.
`isaac_ros_rtdetr`, `isaac_ros_detectnet`, `isaac_ros_yolov8`, and `isaac_ros_grounding_dino` each provide a method for spatial
classification using bounding boxes with an input image. Classification
is performed by a GPU-accelerated model of the appropriate architecture:

- `isaac_ros_rtdetr`: [RT-DETR models](https://nvidia-isaac-ros.github.io/concepts/object_detection/rtdetr/index.html)
- `isaac_ros_detectnet`: [DetectNet models](https://nvidia-isaac-ros.github.io/concepts/object_detection/detectnet/index.html)
- `isaac_ros_yolov8`: [YOLOv8 models](https://nvidia-isaac-ros.github.io/concepts/object_detection/yolov8/index.html)
- `isaac_ros_grounding_dino`: [Grounding DINO models](https://nvidia-isaac-ros.github.io/concepts/object_detection/grounding_dino/index.html)

The output prediction can be used by perception functions to
understand the presence and spatial location of an object in an image.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_nodegraph.png/" width="800px"/></a></div>

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

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example_bboxseg.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_object_detection/isaac_ros_object_detection_example_bboxseg.png/" width="800px"/></a></div>

Object detection classifies a rectangle of pixels as containing an
object, whereas image segmentation provides more information and uses
more compute to produce a classification per pixel. Object detection is
used to know if, and where in a 2D image, the object exists. If a 3D
spacial understanding or size of an object in pixels is required, use
image segmentation.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                                    | Input Size<br/><br/>   | AGX Thor<br/><br/>                                                                                                                                                             | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DetectNet Object Detection Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_detectnet_benchmark/scripts/isaac_ros_detectnet_graph.py)<br/><br/>                      | 544p<br/><br/>         | [143 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_detectnet_graph-agx_thor.json)<br/><br/><br/>27 ms @ 30Hz<br/><br/>       | [242 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_detectnet_graph-x86-5090.json)<br/><br/><br/>17 ms @ 30Hz<br/><br/>      |
| [Grounding DINO Object Detection Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_grounding_dino_benchmark/scripts/isaac_ros_grounding_dino_graph.py)<br/><br/>       | 544p<br/><br/>         | [22.9 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_grounding_dino_graph-agx_thor.json)<br/><br/><br/>70 ms @ 30Hz<br/><br/> | [144 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_grounding_dino_graph-x86-5090.json)<br/><br/><br/>15 ms @ 30Hz<br/><br/> |
| [RT-DETR Object Detection Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_rtdetr_benchmark/scripts/isaac_ros_rtdetr_graph.py)<br/><br/><br/>SyntheticaDETR<br/><br/> | 720p<br/><br/>         | [219 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_rtdetr_graph-agx_thor.json)<br/><br/><br/>23 ms @ 30Hz<br/><br/>          | [457 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_rtdetr_graph-x86-5090.json)<br/><br/><br/>8.0 ms @ 30Hz<br/><br/>        |

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
* [`isaac_ros_grounding_dino`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_grounding_dino/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_grounding_dino/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_grounding_dino/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_grounding_dino/index.html#api)
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

Update 2025-10-24: Added Grounding DINO object detection package
