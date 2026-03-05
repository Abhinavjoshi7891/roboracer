#!/usr/bin/env python3
"""
Telemetry Logger Node for AutoDRIVE RoboRacer
Subscribes to ground-truth topics (IPS, IMU, etc.) and continuously dumps them 
to a CSV file for live dashboard visualization on the host.
"""

import os
import csv
import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import Point

# We don't have numpy or scipy in this lightweight node by default if it's just passing data,
# but we do need to convert quaternions to yaw.
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z  # in radians

class TelemetryLogger(Node):
    def __init__(self):
        super().__init__('telemetry_logger')

        self.telemetry_path = '/tmp/telemetry/live_telemetry.csv'
        os.makedirs(os.path.dirname(self.telemetry_path), exist_ok=True)
        self.telemetry_file = open(self.telemetry_path, 'w', newline='')
        self.telemetry_writer = csv.writer(self.telemetry_file)
        
        # Header
        self.telemetry_writer.writerow([
            'timestamp', 'x', 'y', 'heading_rad', 'speed', 
            'steering_cmd', 'throttle_cmd', 'lap_count', 
            'lap_time', 'best_lap_time', 'collision_count',
            'min_range_left', 'min_range_front', 'min_range_right'
        ])
        
        self.start_time = time.time()
        
        # Latest State Cache
        self.state = {
            'x': 0.0,
            'y': 0.0,
            'heading_rad': 0.0,
            'speed': 0.0,
            'steering_cmd': 0.0,
            'throttle_cmd': 0.0,
            'lap_count': 0,
            'lap_time': 0.0,
            'best_lap_time': 0.0,
            'collision_count': 0,
            'min_range_left': 0.0,
            'min_range_front': 0.0,
            'min_range_right': 0.0
        }

        # Subscribers
        self.create_subscription(Point, '/autodrive/roboracer_1/ips', self._ips_cb, 10)
        self.create_subscription(Imu, '/autodrive/roboracer_1/imu', self._imu_cb, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/speed', self._speed_cb, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering_command', self._steer_cb, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle_command', self._throttle_cb, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', self._lap_count_cb, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/lap_time', self._lap_time_cb, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/best_lap_time', self._best_lap_time_cb, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', self._collision_count_cb, 10)
        self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self._lidar_cb, 10)

        # Write frequency: 50Hz (every 20ms) - matches fast physics updates
        self.timer = self.create_timer(0.02, self._write_telemetry)
        
        self.get_logger().info(f"Live telemetry logger initialized. Writing to {self.telemetry_path}")

    # Callbacks
    def _ips_cb(self, msg):
        self.state['x'] = msg.x
        self.state['y'] = msg.y

    def _imu_cb(self, msg):
        q = msg.orientation
        self.state['heading_rad'] = euler_from_quaternion(q.x, q.y, q.z, q.w)

    def _speed_cb(self, msg):
        self.state['speed'] = msg.data

    def _steer_cb(self, msg):
        self.state['steering_cmd'] = msg.data

    def _throttle_cb(self, msg):
        self.state['throttle_cmd'] = msg.data
        
    def _lap_count_cb(self, msg):
        self.state['lap_count'] = msg.data

    def _lap_time_cb(self, msg):
        self.state['lap_time'] = msg.data

    def _best_lap_time_cb(self, msg):
        self.state['best_lap_time'] = msg.data

    def _collision_count_cb(self, msg):
        self.state['collision_count'] = msg.data

    def _lidar_cb(self, msg):
        import numpy as np # Import locally if needed, but we can avoid numpy
        # Lidar comes in as 1081 array (270 degrees)
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0: return
        third = n // 3
        
        def safe_min(sector):
            valid = [r for r in sector if math.isfinite(r) and r > 0]
            if not valid: return 30.0 # max default
            return min(valid)
            
        self.state['min_range_right'] = safe_min(ranges[:third])
        self.state['min_range_front'] = safe_min(ranges[third:2*third])
        self.state['min_range_left'] = safe_min(ranges[2*third:])

    def _write_telemetry(self):
        t = time.time() - self.start_time
        
        row = [
            f"{t:.4f}",
            f"{self.state['x']:.3f}",
            f"{self.state['y']:.3f}",
            f"{self.state['heading_rad']:.4f}",
            f"{self.state['speed']:.3f}",
            f"{self.state['steering_cmd']:.4f}",
            f"{self.state['throttle_cmd']:.4f}",
            self.state['lap_count'],
            f"{self.state['lap_time']:.3f}",
            f"{self.state['best_lap_time']:.3f}",
            self.state['collision_count'],
            f"{self.state['min_range_left']:.3f}",
            f"{self.state['min_range_front']:.3f}",
            f"{self.state['min_range_right']:.3f}"
        ]
        
        self.telemetry_writer.writerow(row)
        self.telemetry_file.flush() # Force write to disk so host can read it instantly

    def destroy_node(self):
        if self.telemetry_file:
            self.telemetry_file.close()
            self.get_logger().info('Telemetry file closed.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TelemetryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
