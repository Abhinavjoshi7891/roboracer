#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np


class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')

        # === TUNABLE PARAMETERS ===
        # LiDAR preprocessing
        self.MAX_LIDAR_RANGE = 4.0       # Clip ranges beyond this (meters)
        self.SMOOTHING_WINDOW = 5        # Moving average filter window

        # Safety bubble
        self.SAFETY_BUBBLE_RADIUS = 80   # Indices to zero out around closest obstacle

        # Disparity extender
        self.DISPARITY_THRESHOLD = 0.5   # Range difference (m) to trigger extension
        self.DISPARITY_EXTEND = 40       # Number of indices to extend the shorter reading

        # Gap selection
        self.MIN_GAP_WIDTH = 10          # Minimum consecutive points to count as a gap
        self.FORWARD_BIAS_WEIGHT = 0.3   # How much to bias toward straight ahead (0=none, 1=strong)

        # Speed control
        self.MAX_SPEED = 0.15            # Throttle on straights
        self.MIN_SPEED = 0.04            # Throttle in tight turns
        self.EMERGENCY_DISTANCE = 0.25   # Distance (m) that triggers emergency slowdown
        self.CAUTION_DISTANCE = 0.5      # Distance (m) that triggers moderate slowdown

        # Steering
        self.STEERING_GAIN = 0.8         # Overall steering response multiplier

        # Publishers
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steering_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)

        # Subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, 10)

        self.frame_count = 0
        self.get_logger().info('Gap Follower v2 node started')

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float64)
        num_points = len(ranges)

        # === STEP 1: Preprocess ===
        ranges = np.where(np.isfinite(ranges), ranges, self.MAX_LIDAR_RANGE)
        ranges = np.clip(ranges, 0.0, self.MAX_LIDAR_RANGE)

        # Smooth
        if self.SMOOTHING_WINDOW > 1:
            kernel = np.ones(self.SMOOTHING_WINDOW) / self.SMOOTHING_WINDOW
            ranges = np.convolve(ranges, kernel, mode='same')

        # === STEP 2: Disparity extender ===
        # When adjacent points have a big range difference, a wall edge exists.
        # Extend the shorter reading outward to "thicken" the wall edge,
        # preventing the car from trying to squeeze past.
        processed = ranges.copy()
        for i in range(1, num_points):
            diff = ranges[i] - ranges[i - 1]
            if abs(diff) > self.DISPARITY_THRESHOLD:
                if diff > 0:
                    # ranges[i-1] is closer (wall), ranges[i] is far (open)
                    # Extend the wall reading to the right
                    extend_end = min(num_points, i + self.DISPARITY_EXTEND)
                    processed[i:extend_end] = np.minimum(processed[i:extend_end], ranges[i - 1])
                else:
                    # ranges[i] is closer (wall), ranges[i-1] is far (open)
                    # Extend the wall reading to the left
                    extend_start = max(0, i - self.DISPARITY_EXTEND)
                    processed[extend_start:i] = np.minimum(processed[extend_start:i], ranges[i])
        ranges = processed

        # === STEP 3: Safety bubble ===
        # Re-evaluate closest obstacle without Disparity Extender for safety check
        closest_idx = np.argmin(ranges)
        closest_dist = ranges[closest_idx]
        
        # We NO LONGER zero out the safety bubble in the range array here. 
        # Zeroing it out was causing the asymmetric left-bias.
        # Instead, we just measure it.

        # === STEP 4: Find best gap ===
        # Restrict gap finding to a forward cone (e.g. 135 deg sweep: indices 270 to 810)
        # This prevents the car from finding "gaps" behind it and going into a permanent spin.
        look_start = num_points // 4       # Index 270 (approx -67.5 deg)
        look_end = 3 * num_points // 4     # Index 810 (approx +67.5 deg)
        
        forward_ranges = ranges[look_start:look_end]
        best_gap_start_local, best_gap_end_local = self._find_best_gap(forward_ranges)
        
        best_gap_start = look_start + best_gap_start_local
        best_gap_end = look_start + best_gap_end_local
        
        gap_width = best_gap_end - best_gap_start

        if gap_width < self.MIN_GAP_WIDTH:
            self._publish_commands(0.0, 0.0)
            self.get_logger().warn('No valid gap found! Stopping.')
            return

        # === STEP 5: Select target point WITH forward bias ===
        # Instead of just picking the farthest point, blend between
        # "farthest point in gap" and "center of gap closest to straight ahead"
        gap_ranges = ranges[best_gap_start:best_gap_end]

        # Farthest point in the gap
        farthest_local_idx = np.argmax(gap_ranges)

        # Center of gap
        center_local_idx = len(gap_ranges) // 2

        # Forward-biased index: the point in the gap closest to straight ahead (index 539)
        center_scan = num_points // 2  # index 539 = straight ahead
        # Clamp to gap bounds
        forward_local_idx = np.clip(center_scan - best_gap_start, 0, len(gap_ranges) - 1)

        # Blend: weighted average of farthest point and forward-biased point
        # This prevents the car from chasing gaps that are far off to the side
        w = self.FORWARD_BIAS_WEIGHT
        blended_local_idx = int((1.0 - w) * farthest_local_idx + w * forward_local_idx)
        blended_local_idx = np.clip(blended_local_idx, 0, len(gap_ranges) - 1)

        best_idx = best_gap_start + blended_local_idx

        # Calculate target angle
        target_angle = msg.angle_min + best_idx * msg.angle_increment

        # Final gap-based steering normalized to [-1, 1]
        gap_steering = (target_angle / 0.5236) * self.STEERING_GAIN
        gap_steering = np.clip(gap_steering, -1.0, 1.0)

        # === STEP 6A: Compute Wall Centering Steering ===
        # Read the distances slightly forward of directly left/right
        left_dist = np.mean(msg.ranges[800:820])  # Approx +65 to +70 deg
        right_dist = np.mean(msg.ranges[260:280]) # Approx -70 to -65 deg
        
        # Guard against infinity/nan
        left_dist = min(left_dist, self.MAX_LIDAR_RANGE) if np.isfinite(left_dist) else self.MAX_LIDAR_RANGE
        right_dist = min(right_dist, self.MAX_LIDAR_RANGE) if np.isfinite(right_dist) else self.MAX_LIDAR_RANGE

        # Calculate a correction factor to keep the car centered
        # If right is further away (positive), we should steer right (negative steering)
        # Note: AutoDRIVE has Positive Steering = LEFT
        wall_steer = 0.0
        dist_sum = left_dist + right_dist
        if dist_sum > 0.01:
            # normalized difference in [-1, 1] range
            diff = right_dist - left_dist 
            # Tune the 1.5 multiplier based on how aggressive centering should be
            wall_steer = -(diff / dist_sum) * 1.5 
        wall_steer = np.clip(wall_steer, -1.0, 1.0)

        # === STEP 6B: Blend Steerings ===
        # On straights (gap is mostly straight ahead), rely on wall centering.
        # On corners (gap is pulled to a side), switch to the gap follower.
        gap_magnitude = abs(gap_steering)
        
        # When gap steering is > 0.3, it takes full control. 
        # When near 0, wall centering takes full control.
        blend_factor = np.clip(gap_magnitude / 0.3, 0.0, 1.0)
        
        steering_cmd = (1.0 - blend_factor) * wall_steer + (blend_factor) * gap_steering
        steering_cmd = np.clip(steering_cmd, -1.0, 1.0)

        # === STEP 7: Proximity-based speed control ===
        # Check how close walls are on the left, right, and front
        # Front sector: middle 20% of scan (~indices 432 to 648)
        front_start = int(num_points * 0.4)
        front_end = int(num_points * 0.6)
        front_min = np.min(np.where(
            np.array(msg.ranges[front_start:front_end]) > 0.06,
            msg.ranges[front_start:front_end],
            self.MAX_LIDAR_RANGE))

        # Side sectors for S-curve awareness
        # Right side: indices 0 to 270 (roughly -135° to -67°)
        right_min = np.min(np.where(
            np.array(msg.ranges[:num_points // 4]) > 0.06,
            msg.ranges[:num_points // 4],
            self.MAX_LIDAR_RANGE))
        # Left side: indices 810 to 1079 (roughly +67° to +135°)
        left_min = np.min(np.where(
            np.array(msg.ranges[3 * num_points // 4:]) > 0.06,
            msg.ranges[3 * num_points // 4:],
            self.MAX_LIDAR_RANGE))

        side_min = min(left_min, right_min)

        # Speed determination
        if front_min < self.EMERGENCY_DISTANCE:
            # Emergency: obstacle very close ahead
            throttle = self.MIN_SPEED * 0.5
        elif front_min < self.CAUTION_DISTANCE or side_min < self.CAUTION_DISTANCE:
            # Caution: walls nearby (S-curve situation)
            throttle = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * 0.3
        else:
            # Normal: speed based on steering magnitude
            throttle = self.MAX_SPEED - (self.MAX_SPEED - self.MIN_SPEED) * abs(steering_cmd)

        # === STEP 8: Publish ===
        self._publish_commands(throttle, steering_cmd)

        # Debug logging (every 40th frame ≈ 1/sec)
        self.frame_count += 1
        if self.frame_count % 40 == 0:
            self.get_logger().info(
                f'Steer: {steering_cmd:.3f} (Wall: {wall_steer:.3f}, Gap: {gap_steering:.3f}, B: {blend_factor:.2f}) | '
                f'Throt: {throttle:.4f} | '
                f'Walls: L{left_dist:.2f}/R{right_dist:.2f} Front: {front_min:.2f}'
            )

    def _find_best_gap(self, ranges):
        """Find the longest contiguous sequence of non-zero values."""
        non_zero = (ranges > 0.0).astype(int)
        best_start = 0
        best_length = 0
        current_start = 0
        current_length = 0

        for i in range(len(non_zero)):
            if non_zero[i] == 1:
                if current_length == 0:
                    current_start = i
                current_length += 1
            else:
                if current_length > best_length:
                    best_start = current_start
                    best_length = current_length
                current_length = 0

        if current_length > best_length:
            best_start = current_start
            best_length = current_length

        return best_start, best_start + best_length

    def _publish_commands(self, throttle, steering):
        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)

        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
