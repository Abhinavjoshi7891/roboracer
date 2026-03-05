#!/usr/bin/env python3
"""
LocalMapPP autonomous racing node for AutoDRIVE RoboRacer.
Ported from f1tenth_benchmarks (Evans et al. 2024).

Pipeline per LiDAR scan:
1. Extract local map (track boundaries) from LiDAR
2. Optimize minimum-curvature raceline on local map
3. Generate physics-based speed profile
4. Track raceline with Pure Pursuit
5. Publish steering + throttle commands
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point
import math
import numpy as np
from scipy import interpolate
import trajectory_planning_helpers as tph
from numba import njit
import csv
import time
import os

from autodrive_roboracer.local_map_generator import LocalMapGenerator
from autodrive_roboracer.local_opt_min_curv import local_opt_min_curv


# ============ Vehicle & Planner Parameters ============
# From vehicle_params.yaml and LocalMapPP.yaml in f1tenth_benchmarks
WHEELBASE = 0.33            # meters (confirmed from AutoDRIVE TF tree)
MAX_SPEED = 8.0             # m/s (f1tenth platform max)
VEHICLE_MASS = 3.71         # kg
GRAVITY = 9.81

# LocalMapPP planner params
MU = 0.6                    # Surface friction coefficient
KAPPA_BOUND = 0.8           # Max curvature for raceline optimization
PATH_EXCLUSION_WIDTH = 1.0  # Safety margin subtracted from track widths
CENTRE_LOOKAHEAD = 2.0      # Lookahead for centre-line fallback (m) — increased to reduce oversteer
CONSTANT_LOOKAHEAD = 0.3    # Base lookahead distance for Pure Pursuit
VARIABLE_LOOKAHEAD = 1.6    # Speed-dependent additional lookahead
MAX_LATERAL_ACC = 8.5       # m/s² for speed profile
MAX_LONGITUDINAL_ACC = 8.5  # m/s² for speed profile
MAX_STEER_RAD = 0.4         # Max steering angle in radians

# AutoDRIVE-specific conversion
MAX_STEERING_ANGLE_RAD = 0.5236  # ±30° = ±0.5236 rad (steering_command ±1.0)

# Speed/throttle mapping — MUST BE CALIBRATED
THROTTLE_SPEED_GAIN = 0.05  # throttle = speed_mps * gain (start conservative)
MAX_THROTTLE = 0.5          # Clamp max throttle (increase after testing)

# ============ Track Mapping & Curvature Speed Control ============
MAPPING_SPEED = 1.5             # m/s — slow speed for lap 0 mapping
RACING_SPEED_MAX = 6.0          # m/s — max speed on straights during racing laps
MIN_CORNER_SPEED = 1.5          # m/s — minimum speed even through tightest corners
CURVATURE_LOOKAHEAD_M = 4.0     # meters — how far ahead to check for curvature
A_LAT_MAX_CORNERING = 3.5       # m/s² — max lateral acceleration for speed limiting
MAP_SAMPLE_DIST = 0.15          # meters — minimum distance between map points


class LocalMapPP(Node):
    def __init__(self):
        super().__init__('local_map_pp')

        # Core modules
        self.local_map_generator = LocalMapGenerator()

        # Speed profile generation parameters
        self.ggv = np.array([
            [0, MAX_LONGITUDINAL_ACC, MAX_LATERAL_ACC],
            [MAX_SPEED, MAX_LONGITUDINAL_ACC, MAX_LATERAL_ACC]
        ])
        self.ax_max_machine = np.array([
            [0, MAX_LONGITUDINAL_ACC],
            [MAX_SPEED, MAX_LONGITUDINAL_ACC]
        ])

        # State
        self.current_speed = 0.0
        self.raceline = None
        self.s_raceline = None
        self.vs = None
        self.tck = None
        self.local_track = None
        self.frame_count = 0
        self.use_raceline = False  # Set False to use centre-line only (safer fallback)

        # Collision detection state
        self.collision_count = 0
        self.last_collision_frame = -50  # prevent double-counting same impact
        self.last_throttle_cmd = 0.0

        # Telemetry logger
        self.telemetry_path = '/tmp/race_telemetry.csv'
        self.telemetry_file = open(self.telemetry_path, 'w', newline='')
        self.telemetry_writer = csv.writer(self.telemetry_file)
        self.telemetry_writer.writerow([
            'timestamp', 'frame', 'speed', 'steering_cmd', 'throttle_cmd',
            'steering_angle_deg', 'track_pts', 'collision',
            'min_range_right', 'min_range_front', 'min_range_left',
            'avg_range_right', 'avg_range_front', 'avg_range_left',
            'collision_total'
        ])
        self.start_time = time.time()
        self.last_scan = None

        # Snapshots directory for extraction visualization
        self.snapshots_dir = '/tmp/telemetry/extraction_snapshots'
        os.makedirs(self.snapshots_dir, exist_ok=True)

        # ---- Track mapping state ----
        self.global_x = 0.0
        self.global_y = 0.0
        self.current_lap = 0
        self.track_map = []              # List of (x, y) during lap 1
        self.track_map_np = None         # Numpy array after processing
        self.track_curvatures = None     # Curvature at each map point
        self.track_map_s = None          # Cumulative arc-length along map
        self.mapping_complete = False
        self.last_map_x = None
        self.last_map_y = None

        # Publishers
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steering_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)

        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, 10)
        self.speed_sub = self.create_subscription(
            Float32, '/autodrive/roboracer_1/speed', self.speed_callback, 10)
        self.ips_sub = self.create_subscription(
            Point, '/autodrive/roboracer_1/ips', self._ips_callback, 10)
        self.lap_sub = self.create_subscription(
            Int32, '/autodrive/roboracer_1/lap_count', self._lap_count_callback, 10)

        self.get_logger().info(f'LocalMapPP node started — telemetry logging to {self.telemetry_path}')

    def speed_callback(self, msg):
        self.current_speed = msg.data

    def _ips_callback(self, msg):
        self.global_x = msg.x
        self.global_y = msg.y

        # During lap 0 (warmup/mapping), accumulate position samples
        if not self.mapping_complete and self.current_lap < 1:
            if self.last_map_x is None:
                self.track_map.append((msg.x, msg.y))
                self.last_map_x = msg.x
                self.last_map_y = msg.y
            else:
                dist = math.sqrt((msg.x - self.last_map_x)**2 + (msg.y - self.last_map_y)**2)
                if dist >= MAP_SAMPLE_DIST:
                    self.track_map.append((msg.x, msg.y))
                    self.last_map_x = msg.x
                    self.last_map_y = msg.y

    def _lap_count_callback(self, msg):
        new_lap = msg.data
        if new_lap != self.current_lap:
            old_lap = self.current_lap
            self.current_lap = new_lap
            self.get_logger().info(f'Lap change: {old_lap} -> {new_lap}')

            # Build the track map at any lap transition once we have enough data
            if not self.mapping_complete and len(self.track_map) > 20:
                self._build_track_map()

    def _build_track_map(self):
        """Process accumulated (x,y) samples into a curvature-annotated track map."""
        pts = np.array(self.track_map)
        self.get_logger().info(f'Building track map from {len(pts)} raw points...')

        # Compute element lengths and cumulative arc-length
        diffs = np.diff(pts, axis=0)
        el_lengths = np.linalg.norm(diffs, axis=1)
        s = np.insert(np.cumsum(el_lengths), 0, 0.0)

        # Compute curvature using finite differences of heading
        # heading = atan2(dy, dx)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])

        # Handle angle wrapping in heading differences
        dheading = np.diff(headings)
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        ds = el_lengths[1:]  # distances between heading samples

        # Curvature = d(heading)/ds, pad edges
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.abs(np.where(ds > 0.01, dheading / ds, 0.0))

        # Smooth curvature with a moving average to reduce noise
        kernel_size = 15
        kernel = np.ones(kernel_size) / kernel_size
        curvature_smooth = np.convolve(curvature, kernel, mode='same')

        # Pad to match the original points array (curvature has len(pts)-2 elements)
        curvature_full = np.zeros(len(pts))
        curvature_full[1:-1] = curvature_smooth
        curvature_full[0] = curvature_full[1]
        curvature_full[-1] = curvature_full[-2]

        self.track_map_np = pts
        self.track_curvatures = curvature_full
        self.track_map_s = s
        self.mapping_complete = True

        max_curv = np.max(curvature_full)
        avg_curv = np.mean(curvature_full)
        self.get_logger().info(
            f'Track map built! {len(pts)} pts, '
            f'total length={s[-1]:.1f}m, '
            f'max curvature={max_curv:.2f}, avg curvature={avg_curv:.3f}'
        )

        # Save the track map for offline analysis
        map_path = '/tmp/telemetry/track_map.npz'
        np.savez(map_path, points=pts, curvatures=curvature_full, arc_length=s)
        self.get_logger().info(f'Track map saved to {map_path}')

    def _get_curvature_speed_limit(self):
        """Look ahead on the global track map and compute speed limit from curvature."""
        if self.track_map_np is None or len(self.track_map_np) < 5:
            return RACING_SPEED_MAX

        # Find nearest point on track map to current global position
        dists = np.sqrt(
            (self.track_map_np[:, 0] - self.global_x)**2 +
            (self.track_map_np[:, 1] - self.global_y)**2
        )
        nearest_idx = np.argmin(dists)
        current_s = self.track_map_s[nearest_idx]

        # Look ahead along the track map
        lookahead_s = current_s + CURVATURE_LOOKAHEAD_M
        total_s = self.track_map_s[-1]

        # Handle wrap-around (closed track)
        if lookahead_s > total_s:
            # Check from current to end + start to remainder
            mask1 = self.track_map_s >= current_s
            remainder = lookahead_s - total_s
            mask2 = self.track_map_s <= remainder
            mask = mask1 | mask2
        else:
            mask = (self.track_map_s >= current_s) & (self.track_map_s <= lookahead_s)

        if np.sum(mask) == 0:
            return RACING_SPEED_MAX

        max_curvature = np.max(self.track_curvatures[mask])

        if max_curvature < 0.05:
            # Essentially straight
            return RACING_SPEED_MAX

        # v = sqrt(a_lat / curvature)
        v_limit = math.sqrt(A_LAT_MAX_CORNERING / max_curvature)
        v_limit = np.clip(v_limit, MIN_CORNER_SPEED, RACING_SPEED_MAX)

        return float(v_limit)

    def lidar_callback(self, msg):
        scan = np.array(msg.ranges, dtype=np.float64)
        self.last_scan = scan.copy()

        # Step 1: Generate local map from LiDAR scan
        self.local_track = self.local_map_generator.generate_line_local_map(scan)

        if len(self.local_track) < 4:
            # Not enough track data — stop
            self._publish_commands(0.0, 0.0)
            self.get_logger().warn('Local map too short, stopping')
            return

        # Step 2 & 3: Compute raceline and speed profile
        if self.use_raceline:
            try:
                self._generate_minimum_curvature_path()
                self._generate_max_speed_profile()
                steering_angle, speed = self._pure_pursuit_racing_line()
            except Exception as e:
                # Raceline optimization failed — fall back to centre line
                self.get_logger().warn(f'Raceline failed ({e}), using centre line')
                steering_angle, speed = self._pure_pursuit_center_line()
        else:
            steering_angle, speed = self._pure_pursuit_center_line()

        # Step 4: Convert to AutoDRIVE commands
        # Steering: radians → normalized [-1, 1]
        steering_cmd = np.clip(steering_angle / MAX_STEERING_ANGLE_RAD, -1.0, 1.0)

        # Speed: m/s → throttle [0, MAX_THROTTLE]
        # This mapping MUST be calibrated empirically
        throttle_cmd = np.clip(speed * THROTTLE_SPEED_GAIN, 0.0, MAX_THROTTLE)

        # Step 5: Publish
        self._publish_commands(throttle_cmd, steering_cmd)

        # Collision detection: car has stopped unexpectedly while we commanded throttle
        COLLISION_SPEED_THRESHOLD = 0.3   # m/s — below this = effectively stopped
        COLLISION_COOLDOWN_FRAMES = 20    # ignore re-triggers for ~2s after a hit
        frames_since_last = self.frame_count - self.last_collision_frame
        if (self.last_throttle_cmd > 0.05
                and self.current_speed < COLLISION_SPEED_THRESHOLD
                and frames_since_last > COLLISION_COOLDOWN_FRAMES):
            self.collision_count += 1
            self.last_collision_frame = self.frame_count
            self.get_logger().warn(
                f'[COLLISION #{self.collision_count}] Speed dropped to '
                f'{self.current_speed:.2f} m/s while throttle={self.last_throttle_cmd:.2f} '
                f'| Total collisions so far: {self.collision_count}'
            )
        self.last_throttle_cmd = throttle_cmd

        # ---- Telemetry logging (every frame) ----
        is_collision = 1 if (
            self.last_throttle_cmd > 0.05
            and self.current_speed < COLLISION_SPEED_THRESHOLD
            and frames_since_last > COLLISION_COOLDOWN_FRAMES
        ) else 0
        # LiDAR sector analysis: split 1081 pts into right / front / left thirds
        if self.last_scan is not None:
            n = len(self.last_scan)
            third = n // 3
            valid_scan = np.where(np.isfinite(self.last_scan), self.last_scan, 30.0)
            right_sector = valid_scan[:third]
            front_sector = valid_scan[third:2*third]
            left_sector = valid_scan[2*third:]
            lidar_stats = [
                float(np.min(right_sector)), float(np.min(front_sector)), float(np.min(left_sector)),
                float(np.mean(right_sector)), float(np.mean(front_sector)), float(np.mean(left_sector)),
            ]
        else:
            lidar_stats = [0]*6

        self.telemetry_writer.writerow([
            f'{time.time() - self.start_time:.3f}',
            self.frame_count,
            f'{self.current_speed:.3f}',
            f'{steering_cmd:.4f}',
            f'{throttle_cmd:.4f}',
            f'{np.degrees(steering_angle):.2f}',
            len(self.local_track),
            is_collision,
            *[f'{v:.3f}' for v in lidar_stats],
            self.collision_count
        ])
        self.telemetry_file.flush()

        # Debug logging
        self.frame_count += 1
        if self.frame_count % 40 == 0:
            self.get_logger().info(
                f'Track pts={len(self.local_track)} | '
                f'Steer={steering_cmd:.3f} ({np.degrees(steering_angle):.1f}°) | '
                f'Throttle={throttle_cmd:.3f} (target {speed:.1f} m/s) | '
                f'Speed={self.current_speed:.2f} m/s | '
                f'Collisions={self.collision_count}'
            )
            # Save extraction snapshot for visualization script
            try:
                np.savez(
                    os.path.join(self.snapshots_dir, f'snapshot_{self.frame_count:04d}.npz'),
                    **self.local_map_generator.debug_data
                )
            except Exception as e:
                self.get_logger().error(f"Failed to save snapshot: {e}")

    def _generate_minimum_curvature_path(self):
        """Optimize a minimum-curvature raceline on the local map."""
        track = self.local_track.copy()
        track[:, 2:] -= PATH_EXCLUSION_WIDTH / 2

        try:
            alpha, nvecs = local_opt_min_curv(
                track, KAPPA_BOUND, 0, fix_s=True, fix_e=False
            )
            self.raceline = track[:, :2] + np.expand_dims(alpha, 1) * nvecs
        except Exception:
            # QP solver failed — use centre line as raceline
            self.raceline = track[:, :2]

        self.tck = interpolate.splprep(
            [self.raceline[:, 0], self.raceline[:, 1]], k=min(3, len(self.raceline)-1), s=0
        )[0]

    def _generate_max_speed_profile(self):
        """Generate physics-based speed profile along the raceline."""
        mu = MU * np.ones_like(self.raceline[:, 0])

        raceline_el_lengths = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)
        self.s_raceline = np.insert(np.cumsum(raceline_el_lengths), 0, 0)
        _, raceline_curvature = tph.calc_head_curv_num.calc_head_curv_num(
            self.raceline, raceline_el_lengths, False
        )

        self.vs = tph.calc_vel_profile.calc_vel_profile(
            self.ax_max_machine, raceline_curvature, raceline_el_lengths,
            False, 0, VEHICLE_MASS,
            ggv=self.ggv, mu=mu, v_max=MAX_SPEED,
            v_start=MAX_SPEED, v_end=MAX_SPEED
        )

    def _calculate_zero_point_progress(self):
        """Find the progress parameter (t) on the raceline closest to the vehicle (origin)."""
        n_pts = np.count_nonzero(self.s_raceline < 5)
        s_raceline = self.s_raceline[:n_pts]
        raceline = self.raceline[:n_pts]
        new_points = np.linspace(0, s_raceline[-1], int(s_raceline[-1] * 100))
        xs = np.interp(new_points, s_raceline, raceline[:, 0])
        ys = np.interp(new_points, s_raceline, raceline[:, 1])
        raceline_interp = np.column_stack([xs, ys])
        dists = np.linalg.norm(raceline_interp, axis=1)
        t_new = new_points[np.argmin(dists)] / self.s_raceline[-1]
        return [t_new]

    def _pure_pursuit_racing_line(self):
        """Pure Pursuit tracking of the optimized raceline."""
        lookahead_distance = CONSTANT_LOOKAHEAD + (
            self.current_speed / MAX_SPEED
        ) * VARIABLE_LOOKAHEAD

        current_s = self._calculate_zero_point_progress()
        lookahead_s = current_s[0] + lookahead_distance / self.s_raceline[-1]
        lookahead_point = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_point.shape) > 1:
            lookahead_point = lookahead_point[0]

        exact_lookahead = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(
            lookahead_point, exact_lookahead, WHEELBASE
        )
        steering_angle = np.clip(steering_angle, -MAX_STEER_RAD, MAX_STEER_RAD)

        speed = np.interp(
            current_s, self.s_raceline / self.s_raceline[-1], self.vs
        )[0]

        return steering_angle, speed

    def _pure_pursuit_center_line(self):
        """Fallback: Pure Pursuit on the local map centre line (no optimization)."""
        current_progress = np.linalg.norm(self.local_track[0, 0:2])
        lookahead = CENTRE_LOOKAHEAD + current_progress

        local_el = np.linalg.norm(np.diff(self.local_track[:, 0:2], axis=0), axis=1)
        s_track = np.insert(np.cumsum(local_el), 0, 0)
        lookahead = min(lookahead, s_track[-1])

        xs = np.interp(lookahead, s_track, self.local_track[:, 0])
        ys = np.interp(lookahead, s_track, self.local_track[:, 1])
        lookahead_point = np.array([xs, ys])

        true_lookahead_distance = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(
            lookahead_point, true_lookahead_distance, WHEELBASE
        )
        steering_angle = np.clip(steering_angle, -MAX_STEER_RAD, MAX_STEER_RAD)

        # ---- Curvature-aware speed control ----
        if not self.mapping_complete:
            # Lap 1: drive slowly to map the track
            speed = MAPPING_SPEED
        else:
            # Lap 2+: use global track map curvature for speed
            speed = self._get_curvature_speed_limit()

        # Log periodically (every 100 frames)
        if self.frame_count % 100 == 0:
            mode = 'MAPPING' if not self.mapping_complete else 'RACING'
            self.get_logger().info(
                f'[{mode}] Lap={self.current_lap} | '
                f'Target speed={speed:.2f} m/s | '
                f'Pos=({self.global_x:.1f}, {self.global_y:.1f})'
            )

        return steering_angle, speed

    def _publish_commands(self, throttle, steering):
        throttle_msg = Float32()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)

        steering_msg = Float32()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)


def get_local_steering_actuation(lookahead_point, lookahead_distance, wheelbase):
    """Pure Pursuit steering calculation in local frame (vehicle at origin)."""
    waypoint_y = lookahead_point[1]
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1.0 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    return steering_angle


def main(args=None):
    rclpy.init(args=args)
    node = LocalMapPP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
