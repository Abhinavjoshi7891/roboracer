from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='autodrive_roboracer',
            executable='autodrive_bridge',
            name='autodrive_bridge',
            emulate_tty=True,
            output='screen',
        ),
        Node(
            package='autodrive_roboracer',
            executable='local_map_pp',
            name='local_map_pp',
            emulate_tty=True,
            output='screen',
        ),
        Node(
            package='autodrive_roboracer',
            executable='telemetry_logger',
            name='telemetry_logger',
            emulate_tty=True,
            output='screen',
        ),
    ])
