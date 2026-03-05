#!/usr/bin/env python3
"""
Local Map Extraction Visualizer
Generates 4-panel trajectory extraction images from NPZ snapshots.

Usage:
    python3 visualize_extraction.py                         # visualizes the latest snapshot
    python3 visualize_extraction.py telemetry/extraction_snapshots/snapshot_0040.npz
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec


def draw_car(ax, x=0, y=0, heading=0):
    """Draw a simple racing car representation at x,y with heading."""
    length = 0.5
    width = 0.25
    # Car body (triangle pointing forward)
    pts = np.array([
        [length/2, 0],
        [-length/2, width/2],
        [-length/2, -width/2]
    ])
    rot = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    pts = np.dot(pts, rot.T) + np.array([x, y])
    poly = Polygon(pts, color='#ff4757', zorder=10)
    ax.add_patch(poly)


def plot_extraction(npz_path, save_path):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    scan = data['scan']
    left_line = data['left_line']
    right_line = data['right_line']
    left_boundary = data['left_boundary']
    right_boundary = data['right_boundary']
    left_extension = data['left_extension']
    right_extension = data['right_extension']
    local_track = data['local_track']

    # Convert scan to Cartesian (angles from -FOV/2 to FOV/2)
    FOV = 4.7
    angles = np.linspace(-FOV/2, FOV/2, len(scan))
    valid = (scan < 30.0) & (scan > 0.1)
    scan_pts = np.column_stack((scan[valid] * np.cos(angles[valid]), scan[valid] * np.sin(angles[valid])))

    # Common styling styling
    BG_COLOR = '#1a1a2e'
    PANEL_BG = '#16213e'
    TEXT_COLOR = '#e0e0e0'
    GRID_COLOR = '#2a2a4a'
    
    fig = plt.figure(figsize=(20, 5), facecolor=BG_COLOR)
    gs = GridSpec(1, 4, wspace=0.1, hspace=0)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    # Calculate limits based on LiDAR and Track points to keep plots uniform
    # We want a birds-eye view in the car's local frame
    car_x, car_y = 0, 0
    all_x = [0]
    all_y = [0]
    if len(scan_pts) > 0:
        all_x.extend(scan_pts[:, 0])
        all_y.extend(scan_pts[:, 1])
    if len(local_track) > 0:
        all_x.extend(local_track[:, 0])
        all_y.extend(local_track[:, 1])
        
    x_min, x_max = min(all_x)-1, max(all_x)+1
    y_min, y_max = min(all_y)-1, max(all_y)+1

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values(): spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)

    # --- Panel 1: LiDAR Scan ---
    ax = axes[0]
    ax.set_title('1. LiDAR Scan', color=TEXT_COLOR)
    ax.plot(scan_pts[:, 0], scan_pts[:, 1], '.', color='#00d2ff', markersize=3, alpha=0.6, label='LiDAR Scan')
    draw_car(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc='lower right')

    # --- Panel 2: Candidate Boundaries ---
    ax = axes[1]
    ax.set_title('2. Extracted Line Candidates', color=TEXT_COLOR)
    ax.plot(scan_pts[:, 0], scan_pts[:, 1], '.', color='#00d2ff', markersize=1, alpha=0.2)
    if len(left_line) > 0:
        ax.plot(left_line[:, 0], left_line[:, 1], '-o', color='#ffa502', markersize=4, label='Long Boundary' if len(left_line) >= len(right_line) else 'Short Boundary')
    if len(right_line) > 0:
        ax.plot(right_line[:, 0], right_line[:, 1], '-*', color='#ff6b6b', markersize=5, label='Short Boundary' if len(left_line) >= len(right_line) else 'Long Boundary')
    draw_car(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc='lower right')

    # --- Panel 3: Visible & Projected Segments ---
    ax = axes[2]
    ax.set_title('3. Track Boundary Segments', color=TEXT_COLOR)
    ax.plot(scan_pts[:, 0], scan_pts[:, 1], '.', color='#00d2ff', markersize=1, alpha=0.1)
    
    # Plot links between left and right boundaries
    if len(left_boundary) > 0 and len(right_boundary) > 0:
        for i in range(min(len(left_boundary), len(right_boundary))):
            ax.plot([left_boundary[i, 0], right_boundary[i, 0]], 
                    [left_boundary[i, 1], right_boundary[i, 1]], '-', color='#2ed573', linewidth=1, alpha=0.6)
            
    # Plot links for extensions
    if len(left_extension) > 0 and len(right_extension) > 0:
        for i in range(min(len(left_extension), len(right_extension))):
            ax.plot([left_extension[i, 0], right_extension[i, 0]], 
                    [left_extension[i, 1], right_extension[i, 1]], '-', color='#1e90ff', linewidth=1, alpha=0.6)
            
    # Plot the full joined lines
    left_full = np.concatenate([left_boundary, left_extension]) if len(left_extension) > 0 else left_boundary
    right_full = np.concatenate([right_boundary, right_extension]) if len(right_extension) > 0 else right_boundary
    if len(left_full) > 0: ax.plot(left_full[:, 0], left_full[:, 1], '-', color='#eccc68', linewidth=2, label='Boundary Lines')
    if len(right_full) > 0: ax.plot(right_full[:, 0], right_full[:, 1], '-', color='#eccc68', linewidth=2)
            
    # Dummy lines for legends
    ax.plot([], [], '-', color='#2ed573', label='Calculated Segments')
    ax.plot([], [], '-', color='#1e90ff', label='Projected Segments')
    draw_car(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc='lower right')

    # --- Panel 4: Normal Vectors & Centre Line ---
    ax = axes[3]
    ax.set_title('4. Centre Line & Normal Vectors', color=TEXT_COLOR)
    if len(local_track) > 0:
        track = local_track[:, :2]
        widths_left = local_track[:, 2]
        widths_right = local_track[:, 3]
        
        # Approximate normals from track trajectory
        if len(track) > 1:
            dx = np.gradient(track[:, 0])
            dy = np.gradient(track[:, 1])
            nvecs = np.column_stack([-dy, dx])
            norms = np.linalg.norm(nvecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            nvecs = nvecs / norms
            
            l1 = track + nvecs * widths_left[:, None]
            l2 = track - nvecs * widths_right[:, None]
            
            # Plot normal vectors
            for i in range(len(track)):
                ax.plot([l1[i, 0], l2[i, 0]], [l1[i, 1], l2[i, 1]], '-', color='#747d8c', linewidth=1, alpha=0.5)
            
            ax.plot([], [], '-', color='#747d8c', label='Normal Vectors')
        
        ax.plot(track[:, 0], track[:, 1], '-', color='#ff4757', linewidth=3, label='Centre Line')
        
        # Plot safety widths (optional)
        if len(track) > 1:
            ax.plot(l1[:, 0], l1[:, 1], '--', color='#57606f', linewidth=1)
            ax.plot(l2[:, 0], l2[:, 1], '--', color='#57606f', linewidth=1)

    draw_car(ax)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved extraction visualization to: {save_path}")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        snapshots_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'telemetry', 'extraction_snapshots'
        )
        if not os.path.exists(snapshots_dir):
            print(f"Directory {snapshots_dir} not found. Run a race first!")
            sys.exit(1)
            
        files = glob.glob(os.path.join(snapshots_dir, '*.npz'))
        if not files:
            print(f"No snapshot files found in {snapshots_dir}.")
            sys.exit(1)
            
        # Get the latest one by modification time
        target = max(files, key=os.path.getmtime)
        
    # Save to current working directory instead of the root-owned snapshots dir
    filename = os.path.basename(target)
    out_img = os.path.join(os.getcwd(), filename.replace('.npz', '_viz.png'))
    plot_extraction(target, out_img)


if __name__ == '__main__':
    main()
