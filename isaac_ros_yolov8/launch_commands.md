# Custom Nitros YOLOv8

This sample shows how to use your custom model decoder with Isaac ROS Managed Nitros. We consider the task of Object Detection using YOLOv8 with [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference). 

### Launch the image publisher in Terminal 1:
```
ros2 run image_publisher image_publisher_node <path-to-image> --ros-args --remap /image_raw:=/image` 
```

For example (sample image people_cycles.jpg provided in this repo):
```
ros2 run image_publisher image_publisher_node people_cycles.jpg --ros-args --remap /image_raw:=/image
```

### Launch the inference graph in Terminal 2:
```
ros2 launch isaac_ros_yolov8 yolov8_tensor_rt.launch.py engine_file_path:=yolov8s_fp16.plan input_binding_names:=['images'] output_binding_names:=['output0'] network_image_width:=640 network_image_height:=640 model_file_path:=yolov8s_fp16.onnx force_engine_update:=False image_mean:=[0.485,0.456,0.406] image_stddev:=[0.229,0.224,0.225] input_image_width:=640 input_image_height:=640
```

Results will be published to `/detections_output` as Detection2DArray messages.

## Visualizing results:
```
ros2 launch isaac_ros_yolov8 isaac_ros_yolov8_visualize.launch.py engine_file_path:=yolov8s_fp16.plan input_binding_names:=['images'] output_binding_names:=['output0'] network_image_width:=640 network_image_height:=640 model_file_path:=yolov8s_fp16.onnx force_engine_update:=False image_mean:=[0.485,0.456,0.406] image_stddev:=[0.229,0.224,0.225] input_image_width:=640 input_image_height:=640
```

An RQT image window will pop up to display resulting bounding boxes on the input image. These output images are published on the `/yolov8_processed_image` topic.
