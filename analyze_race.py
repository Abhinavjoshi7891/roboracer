#!/usr/bin/env python3
"""
Race Performance Analyzer for LocalMapPP.
Reads telemetry CSV from a race run and generates diagnostic plots.

Usage:
    python3 analyze_race.py                         # uses default path
    python3 analyze_race.py path/to/telemetry.csv   # custom file
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend for SSH / Docker
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
from datetime import timedelta


def load_telemetry(path):
    """Load telemetry CSV into dict-of-arrays."""
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: Telemetry file is empty.")
        sys.exit(1)

    for key in rows[0]:
        try:
            data[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            data[key] = np.array([r[key] for r in rows])

    return data


def plot_race_analysis(data, save_dir):
    """Generate 6-panel race analysis figure."""
    t = data['timestamp']
    t_min = t / 60.0  # convert to minutes

    collision_mask = data['collision'].astype(bool)
    collision_times = t_min[collision_mask]
    collision_speeds = data['speed'][collision_mask]

    total_frames = len(t)
    total_collisions = int(data['collision_total'][-1])
    duration_min = t_min[-1]
    avg_speed = np.mean(data['speed'])
    max_speed = np.max(data['speed'])
    collision_rate = total_collisions / max(duration_min, 0.01)

    # --- Color scheme ---
    BG_COLOR = '#1a1a2e'
    PANEL_BG = '#16213e'
    TEXT_COLOR = '#e0e0e0'
    GRID_COLOR = '#2a2a4a'
    SPEED_COLOR = '#00d2ff'
    STEER_COLOR = '#ff6b6b'
    COLLISION_COLOR = '#ff4757'
    LIDAR_RIGHT = '#ffa502'
    LIDAR_FRONT = '#2ed573'
    LIDAR_LEFT = '#1e90ff'
    TRACK_PTS_COLOR = '#a29bfe'

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), facecolor=BG_COLOR)
    fig.suptitle(
        f'Race Performance Analysis  |  Duration: {duration_min:.1f} min  |  '
        f'Collisions: {total_collisions}  |  Rate: {collision_rate:.1f}/min  |  '
        f'Avg Speed: {avg_speed:.1f} m/s',
        color=TEXT_COLOR, fontsize=14, fontweight='bold', y=0.98
    )

    for ax in axes.flat:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.5, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    # ===== Panel 1: Speed + Collisions =====
    ax1 = axes[0, 0]
    ax1.plot(t_min, data['speed'], color=SPEED_COLOR, linewidth=0.6, alpha=0.8)
    ax1.scatter(collision_times, collision_speeds, color=COLLISION_COLOR,
                s=80, zorder=5, marker='x', linewidths=2, label=f'Collisions ({total_collisions})')
    # Shade collision zones
    for ct in collision_times:
        ax1.axvspan(ct - 0.05, ct + 0.05, color=COLLISION_COLOR, alpha=0.15)
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_xlabel('Time (min)')
    ax1.set_title('Speed Profile with Collision Events')
    ax1.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    ax1.set_ylim(-0.2, max_speed + 1)

    # ===== Panel 2: Steering angle =====
    ax2 = axes[0, 1]
    steer_deg = data['steering_angle_deg']
    ax2.plot(t_min, steer_deg, color=STEER_COLOR, linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)
    for ct in collision_times:
        ax2.axvspan(ct - 0.05, ct + 0.05, color=COLLISION_COLOR, alpha=0.15)
    ax2.set_ylabel('Steering Angle (°)')
    ax2.set_xlabel('Time (min)')
    ax2.set_title('Steering Commands')
    # Histogram on right
    ax2_hist = ax2.twinx()
    ax2_hist.hist(steer_deg, bins=60, orientation='horizontal', alpha=0.15,
                  color=STEER_COLOR, density=True)
    ax2_hist.set_xlim(0, ax2_hist.get_xlim()[1] * 3)
    ax2_hist.tick_params(right=False, labelright=False)

    # ===== Panel 3: LiDAR min ranges (wall proximity) =====
    ax3 = axes[1, 0]
    ax3.plot(t_min, data['min_range_right'], color=LIDAR_RIGHT, linewidth=0.6,
             alpha=0.7, label='Right wall')
    ax3.plot(t_min, data['min_range_front'], color=LIDAR_FRONT, linewidth=0.6,
             alpha=0.7, label='Front wall')
    ax3.plot(t_min, data['min_range_left'], color=LIDAR_LEFT, linewidth=0.6,
             alpha=0.7, label='Left wall')
    for ct in collision_times:
        ax3.axvspan(ct - 0.05, ct + 0.05, color=COLLISION_COLOR, alpha=0.15)
    ax3.set_ylabel('Min LiDAR Range (m)')
    ax3.set_xlabel('Time (min)')
    ax3.set_title('Wall Proximity (closer = more danger)')
    ax3.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9, ncol=3)
    ax3.set_ylim(0, min(10, np.percentile(data['min_range_front'], 95) * 1.5))

    # ===== Panel 4: Track Points Stability =====
    ax4 = axes[1, 1]
    ax4.plot(t_min, data['track_pts'], color=TRACK_PTS_COLOR, linewidth=0.6, alpha=0.8)
    ax4.axhline(y=4, color=COLLISION_COLOR, linewidth=1, linestyle='--',
                alpha=0.5, label='Min track threshold')
    for ct in collision_times:
        ax4.axvspan(ct - 0.05, ct + 0.05, color=COLLISION_COLOR, alpha=0.15)
    ax4.set_ylabel('Track Points')
    ax4.set_xlabel('Time (min)')
    ax4.set_title('Local Map Quality (more pts = better boundary extraction)')
    ax4.legend(loc='upper right', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)

    # ===== Panel 5: Collision Side Analysis =====
    ax5 = axes[2, 0]
    if len(collision_times) > 0:
        coll_idx = np.where(collision_mask)[0]
        right_min = data['min_range_right'][coll_idx]
        left_min = data['min_range_left'][coll_idx]
        front_min = data['min_range_front'][coll_idx]

        # Determine which wall is closest at each collision
        sides = []
        for r, f, l in zip(right_min, front_min, left_min):
            m = min(r, f, l)
            if m == r:
                sides.append('Right')
            elif m == f:
                sides.append('Front')
            else:
                sides.append('Left')

        side_counts = {'Right': sides.count('Right'),
                       'Front': sides.count('Front'),
                       'Left': sides.count('Left')}
        colors = [LIDAR_RIGHT, LIDAR_FRONT, LIDAR_LEFT]
        bars = ax5.bar(side_counts.keys(), side_counts.values(), color=colors,
                       edgecolor='white', linewidth=0.5)
        for bar, count in zip(bars, side_counts.values()):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(count), ha='center', color=TEXT_COLOR, fontsize=12, fontweight='bold')
        ax5.set_ylabel('Count')
        ax5.set_title('Collision Wall Side Analysis')
    else:
        ax5.text(0.5, 0.5, 'No collisions! 🎉', transform=ax5.transAxes,
                 ha='center', va='center', fontsize=16, color=TEXT_COLOR)
        ax5.set_title('Collision Wall Side Analysis')

    # ===== Panel 6: Rolling collision rate + cumulative =====
    ax6 = axes[2, 1]
    # Cumulative collision count
    cumulative = np.cumsum(data['collision'])
    ax6.plot(t_min, cumulative, color=COLLISION_COLOR, linewidth=2, label='Cumulative collisions')
    ax6.set_ylabel('Total Collisions', color=COLLISION_COLOR)
    ax6.set_xlabel('Time (min)')
    ax6.set_title('Collision Accumulation Over Time')

    # Rolling collision rate (per minute, 60s window)
    ax6b = ax6.twinx()
    window_sec = 60.0
    rates = []
    for i in range(len(t)):
        window_mask = (t >= t[i] - window_sec) & (t <= t[i])
        rate = np.sum(data['collision'][window_mask])
        rates.append(rate)
    ax6b.plot(t_min, rates, color='#ffeaa7', linewidth=1, alpha=0.7, label='Rate (per 60s window)')
    ax6b.set_ylabel('Collisions / 60s', color='#ffeaa7')
    ax6b.tick_params(colors='#ffeaa7')
    ax6b.spines['right'].set_color('#ffeaa7')

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(save_dir, 'race_analysis.png')
    plt.savefig(out_path, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    print(f'\n✅ Race analysis saved to: {out_path}')
    plt.close()

    # ===== Summary stats to terminal =====
    print(f'\n{"="*50}')
    print(f'  RACE PERFORMANCE SUMMARY')
    print(f'{"="*50}')
    print(f'  Duration:        {duration_min:.1f} min ({total_frames} frames)')
    print(f'  Total Collisions: {total_collisions}')
    print(f'  Collision Rate:   {collision_rate:.2f} per minute')
    print(f'  Avg Speed:        {avg_speed:.2f} m/s')
    print(f'  Max Speed:        {max_speed:.2f} m/s')
    print(f'  Avg Track Points: {np.mean(data["track_pts"]):.1f}')
    if len(collision_times) > 0:
        print(f'  Collision breakdown:')
        for side, count in side_counts.items():
            print(f'    {side}: {count} ({count/total_collisions*100:.0f}%)')
    print(f'{"="*50}\n')


def main():
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'telemetry', 'race_telemetry.csv'
    )
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(csv_path):
        print(f"ERROR: Telemetry file not found: {csv_path}")
        print("Run a race first with ./launch_localmap.sh and Ctrl-C to stop.")
        print("The telemetry CSV will be saved automatically.")
        sys.exit(1)

    print(f"Loading telemetry from: {csv_path}")
    data = load_telemetry(csv_path)
    save_dir = os.path.dirname(csv_path) or '.'
    plot_race_analysis(data, save_dir)


if __name__ == '__main__':
    main()
