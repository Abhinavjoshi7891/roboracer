#!/usr/bin/env python3
"""
Local map extraction from LiDAR scans.
Ported from f1tenth_benchmarks by Benjamin Evans et al.
Paper: "Unifying F1TENTH Autonomous Racing: Survey, Methods and Benchmarks" (2024)
"""

import numpy as np
from scipy import interpolate
import trajectory_planning_helpers as tph

# Constants from the original implementation
DISTANCE_THRESHOLD = 1.4    # m, gap threshold for boundary segmentation
TRACK_WIDTH = 1.8           # m, assumed fixed track width
FOV = 4.7                   # radians, field of view (270°)
BOUNDARY_SMOOTHING = 0.2
MAX_TRACK_WIDTH = 2.5
TRACK_SEPARATION_DISTANCE = 0.4
BOUNDARY_STEP_SIZE = 0.4


class LocalMapGenerator:
    def __init__(self):
        self.z_transform = None
        self.left_longer = None
        self.debug_data = {}

    def generate_line_local_map(self, scan):
        """
        Takes a 1D numpy array of LiDAR ranges and returns a local track.
        Returns: np.ndarray of shape (N, 4) -> [x, y, width_left, width_right]
        """
        if self.z_transform is None or self.z_transform.shape[0] != len(scan):
            angles = np.linspace(-FOV / 2, FOV / 2, len(scan))
            self.z_transform = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        z = scan[:, None] * self.z_transform

        try:
            left_line, right_line = self.extract_track_boundaries(z)
        except Exception as e:
            # Boundary extraction failed — return empty track
            return np.zeros((0, 4))

        left_boundary, right_boundary = self.calculate_visible_segments(left_line, right_line)
        left_extension, right_extension = self.estimate_semi_visible_segments(
            left_line, right_line, left_boundary, right_boundary
        )
        local_track = self.regularise_track(
            left_boundary, right_boundary, left_extension, right_extension
        )

        self.debug_data = {
            'scan': scan,
            'left_line': left_line,
            'right_line': right_line,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'left_extension': left_extension if left_extension is not None else np.zeros((0, 2)),
            'right_extension': right_extension if right_extension is not None else np.zeros((0, 2)),
            'local_track': local_track
        }

        return local_track

    def extract_track_boundaries(self, z):
        z = z[z[:, 0] > -2]
        z = z[np.logical_or(z[:, 0] > 0, np.abs(z[:, 1]) < 2)]
        pt_distances = np.linalg.norm(z[1:] - z[:-1], axis=1)
        inds = np.array(np.where(pt_distances > DISTANCE_THRESHOLD))

        arr_inds = np.arange(len(pt_distances))[inds]
        if np.min(arr_inds) > 2:
            arr_inds = np.insert(arr_inds, 0, -2)
        if np.max(arr_inds) < len(z) - 3:
            arr_inds = np.append(arr_inds, len(z) - 1)

        candidate_lines = [z[arr_inds[i] + 2:arr_inds[i + 1] + 1] for i in range(len(arr_inds) - 1)]
        candidate_lines = [
            line for line in candidate_lines
            if not np.all(line[:, 0] < -0.8) or np.all(np.abs(line[:, 1]) > 2.5)
        ]
        candidate_lines = [line for line in candidate_lines if len(line) > 1]

        left_line = resample_track_points(candidate_lines[0], BOUNDARY_STEP_SIZE, BOUNDARY_SMOOTHING)
        right_line = resample_track_points(candidate_lines[-1], BOUNDARY_STEP_SIZE, BOUNDARY_SMOOTHING)

        self.left_longer = left_line.shape[0] > right_line.shape[0]

        return left_line, right_line

    def calculate_visible_segments(self, left_line, right_line):
        if self.left_longer:
            left_boundary, right_boundary = calculate_boundary_segments(left_line, right_line)
        else:
            right_boundary, left_boundary = calculate_boundary_segments(right_line, left_line)
        return left_boundary, right_boundary

    def estimate_semi_visible_segments(self, left_line, right_line, left_boundary, right_boundary):
        if self.left_longer:
            if len(left_line) - len(left_boundary) < 3:
                return None, None
            right_extension, left_extension = extend_boundary_lines(
                left_line, left_boundary, right_boundary, -1
            )
        else:
            if len(right_line) - len(right_boundary) < 3:
                return None, None
            left_extension, right_extension = extend_boundary_lines(
                right_line, right_boundary, left_boundary, 1
            )
        return left_extension, right_extension

    def regularise_track(self, left_boundary, right_boundary, left_extension, right_extension):
        if left_extension is not None:
            left_boundary = np.append(left_boundary, left_extension, axis=0)
            right_boundary = np.append(right_boundary, right_extension, axis=0)
        track_centre_line = (left_boundary + right_boundary) / 2
        widths = np.ones_like(track_centre_line) * TRACK_WIDTH / 2
        local_track = np.concatenate((track_centre_line, widths), axis=1)

        local_track = interpolate_4d_track(local_track, TRACK_SEPARATION_DISTANCE, 0.01)

        return local_track


# ============ Helper functions (unchanged from original) ============

def interpolate_4d_track(track, point_separation_distance=0.8, s=0):
    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    ss = np.insert(np.cumsum(el_lengths), 0, 0)
    n_points = int(ss[-1] / point_separation_distance + 1)
    order_k = min(3, len(track) - 1)
    tck = interpolate.splprep(
        [track[:, 0], track[:, 1], track[:, 2], track[:, 3]], u=ss, k=order_k, s=s
    )[0]
    track = np.array(interpolate.splev(np.linspace(0, ss[-1], n_points), tck)).T
    return track


def resample_track_points(points, separation_distance=0.2, smoothing=0.2):
    if points[0, 0] > points[-1, 0]:
        points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / separation_distance), 2)
    smooth_line = interpolate_track_new(points, None, smoothing)
    resampled_points = interpolate_track_new(smooth_line, n_pts, 0)
    return resampled_points


def calculate_boundary_segments(long_line, short_line):
    found_normal = False
    long_boundary = np.zeros_like(long_line)
    short_boundary = np.zeros_like(long_line)
    for i in range(long_line.shape[0]):
        distances = np.linalg.norm(short_line - long_line[i], axis=1)
        idx = np.argmin(distances)
        if distances[idx] > MAX_TRACK_WIDTH:
            if found_normal:
                break
        else:
            found_normal = True
        long_boundary[i] = long_line[i]
        short_boundary[i] = short_line[idx]
    return long_boundary[:i], short_boundary[:i]


def extend_boundary_lines(long_line, long_boundary, short_boundary, direction=1):
    long_extension = long_line[len(long_boundary):]
    nvecs = calculate_nvecs(long_extension)
    short_extension = long_extension - nvecs * TRACK_WIDTH * direction

    if len(short_boundary) > 0 and len(long_boundary) > 0:
        centre_line = (long_boundary + short_boundary) / 2
        threshold = np.linalg.norm(short_boundary[-1] - centre_line[-1])
        for z in range(len(short_extension)):
            if np.linalg.norm(short_extension[z] - centre_line[-1]) < threshold:
                short_extension[z] = short_boundary[-1]

    return short_extension, long_extension


def interpolate_track_new(points, n_points=None, s=0):
    if len(points) <= 1:
        return points
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None:
        n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
    return track


def calculate_nvecs(line):
    el_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(line, el_lengths, False)
    nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi)
    return nvecs
