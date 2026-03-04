from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # The WebSocket bridge — connects to the AutoDRIVE Simulator
        Node(
            package='autodrive_roboracer',
            executable='autodrive_bridge',
            name='autodrive_bridge',
            emulate_tty=True,
            output='screen',
        ),
        # Our gap follower algorithm
        Node(
            package='autodrive_roboracer',
            executable='gap_follower',
            name='gap_follower',
            emulate_tty=True,
            output='screen',
        ),
    ])
