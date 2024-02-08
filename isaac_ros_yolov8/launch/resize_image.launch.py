import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    image_resize_node = ComposableNode(
        name='image_resize_node',
        package='image_proc',
        plugin='image_proc::ResizeNode',
        remappings=[('image/image_raw', 'image_raw'),
                    ('image/camera_info', 'camera_info'), 
                    ('resize/image_raw', 'image')],
        parameters=[{'width': 640, 'height': 640, 'use_scale': False}]
    )
    image_resize_container = ComposableNodeContainer(
        name='argus_mono_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[image_resize_node],
        namespace='',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )
    return launch.LaunchDescription([image_resize_container])
