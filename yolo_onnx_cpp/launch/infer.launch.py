"""Run an exported YOLO model on a camera topic."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('weights'),
        DeclareLaunchArgument('camera', default_value='/camera/image'),
        Node(package='yolo_onnx_cpp', executable='yolo_onnx_node', name='yolo_onnx_cpp',
             output='screen', parameters=[{'model': LaunchConfiguration('weights')}],
             remappings=[('/camera/image', LaunchConfiguration('camera'))]),
    ])
