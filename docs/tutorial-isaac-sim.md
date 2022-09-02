# Tutorial with Isaac Sim

1. Complete the [Quickstart section](../README.md#quickstart) in the main README.
2. Launch the Docker container using the `run_dev.sh` script:
    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```
3. Inside the container, build and source the workspace:  
    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```
4. Run the setup script to download the [PeopleNet Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) from NVIDIA GPU Cloud(NGC) and convert it to a .etlt file
    ```bash 
    cd /workspaces/isaac_ros-dev/src/isaac_ros_object_detection/isaac_ros_detectnet && \
      ./scripts/setup_model.sh --height 720 --width 1280 --config-file resources/isaac_sim_config.pbtxt
    ```
5. Launch the pre-composed pipeline launchfile: 
    ```bash
    cd /workspaces/isaac_ros-dev && \
      ros2 launch isaac_ros_detectnet isaac_ros_detectnet_isaac_sim.launch.py
    ```
6. Install and launch Isaac Sim following the steps in the [Isaac ROS Isaac Sim Setup Guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/isaac-sim-sil-setup.md)
7. Open up the Isaac ROS Common USD scene (using the "content" window) located at:
   
   `omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd`
   
   Wait for it to load completely.
   > **Note:** To use a different server, replace `localhost` with `<your_nucleus_server>`


8. Change the Translate values for the Transform box inside the Property section of the Carter_ROS object  to 
   `X=0.0 , Y=0.0`
   <div align="center"><img src="../resources/change_translate.png" width="400px"/></div>
9. Press **Play** to start publishing data from Isaac Sim.
   <div align="center"><img src="../resources/isaac_sim_play.png" width="600px"/></div>
10. You should see the image from Isaac Sim with the rectangles overlayed over detected people in the frame:
<div align="center"><img src="../resources/isaac_sim_detectnet_output.png" width="600px"/></div>
