#!/usr/bin/env python3
"""
Live Race Telemetry Dashboard
Reads the /tmp/telemetry/live_telemetry.csv tailored by the ROS 2 node
and visualizes the track map, car position, and telemetry gauges in real-time.
"""

import os
import sys
import time
import math
import numpy as np
import pygame
from pygame import gfxdraw

# --- Configuration ---
CSV_PATH = '/home/abhinav/Data_Drive/roboracer/telemetry/live_telemetry.csv'
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
FPS = 30

# --- Colors ---
BG_COLOR = (26, 26, 46)        # #1a1a2e
PANEL_BG = (22, 33, 62)        # #16213e
TEXT_COLOR = (224, 224, 224)   # #e0e0e0
ACCENT_BLUE = (0, 210, 255)    # #00d2ff
ACCENT_RED = (255, 71, 87)     # #ff4757
ACCENT_GREEN = (46, 213, 115)  # #2ed573
ACCENT_YELLOW = (236, 204, 104)# #eccc68
GRID_COLOR = (42, 42, 74)

class LiveDashboard:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("AutoDRIVE Live Race Telemetry")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('courier', 32, bold=True)
        self.font_mid = pygame.font.SysFont('courier', 20, bold=True)
        self.font_small = pygame.font.SysFont('courier', 14)
        
        self.file_pos = 0
        self.history = []     # List of dicts
        self.track_points = [] # For lap 1 mapping
        self.collisions = []   # (x, y)
        self.last_collision_count = 0
        
        # Track bounds and scaling
        self.min_x, self.max_x = -10.0, 10.0
        self.min_y, self.max_y = -10.0, 10.0
        self.scale = 20.0
        self.map_center_offs = (WINDOW_WIDTH//4, WINDOW_HEIGHT//2)

    def _update_scaling(self, x, y):
        """Update map bounds to fit the track dynamically during lap 1."""
        if len(self.track_points) < 2:
            self.min_x = x - 5.0
            self.max_x = x + 5.0
            self.min_y = y - 5.0
            self.max_y = y + 5.0
        else:
            self.min_x = min(self.min_x, x)
            self.max_x = max(self.max_x, x)
            self.min_y = min(self.min_y, y)
            self.max_y = max(self.max_y, y)
            
        # Calculate scale to fit in the left half of the screen (approx 700x800)
        w = max(self.max_x - self.min_x, 10.0)
        h = max(self.max_y - self.min_y, 10.0)
        scale_x = (WINDOW_WIDTH * 0.55 - 100) / w
        scale_y = (WINDOW_HEIGHT - 100) / h
        self.scale = min(scale_x, scale_y)
        
        # Center point
        cx = (self.min_x + self.max_x) / 2
        cy = (self.min_y + self.max_y) / 2
        self.map_center_offs = (
            (WINDOW_WIDTH * 0.3) - cx * self.scale,
            (WINDOW_HEIGHT / 2) + cy * self.scale # Invert Y for screen coords
        )

    def _world_to_screen(self, x, y):
        # Y is inverted in pygame (0 is top)
        sx = int(x * self.scale + self.map_center_offs[0])
        sy = int(-y * self.scale + self.map_center_offs[1])
        return sx, sy

    def read_telemetry(self):
        """Read new rows from the CSV file."""
        if not os.path.exists(CSV_PATH):
            return

        try:
            with open(CSV_PATH, 'r') as f:
                f.seek(self.file_pos)
                lines = f.readlines()
                self.file_pos = f.tell()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('timestamp'):
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 14:
                        state = {
                            't': float(parts[0]),
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'heading': float(parts[3]),
                            'speed': float(parts[4]),
                            'steering': float(parts[5]),
                            'throttle': float(parts[6]),
                            'lap': int(parts[7]),
                            'lap_time': float(parts[8]),
                            'best_lap': float(parts[9]),
                            'collisions': int(parts[10]),
                            'rl': float(parts[11]),
                            'rf': float(parts[12]),
                            'rr': float(parts[13])
                        }
                        
                        self.history.append(state)
                        self.track_points.append((state['x'], state['y'], state['speed']))
                        self._update_scaling(state['x'], state['y'])
                        
                        if state['collisions'] > self.last_collision_count:
                            self.collisions.append((state['x'], state['y']))
                            self.last_collision_count = state['collisions']
                            
                # Keep last 1000 items in history for charts
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]
                    
        except Exception as e:
            print(f"Error reading telemetry: {e}")

    def draw_text(self, text, font, color, x, y, align="left"):
        surf = font.render(text, True, color)
        rect = surf.get_rect()
        if align == "left":
            rect.topleft = (x, y)
        elif align == "center":
            rect.midtop = (x, y)
        elif align == "right":
            rect.topright = (x, y)
        self.screen.blit(surf, rect)

    def draw_bar(self, x, y, w, h, val, min_val, max_val, color):
        pygame.draw.rect(self.screen, GRID_COLOR, (x, y, w, h))
        pct = max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))
        if pct > 0:
            bw = int(w * pct)
            pygame.draw.rect(self.screen, color, (x, y, bw, h))
        pygame.draw.rect(self.screen, TEXT_COLOR, (x, y, w, h), 1)

    def draw_vertical_bar(self, x, y, w, h, val, min_val, max_val, color):
        pygame.draw.rect(self.screen, GRID_COLOR, (x, y, w, h))
        pct = max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))
        if pct > 0:
            bh = int(h * pct)
            pygame.draw.rect(self.screen, color, (x, y + h - bh, w, bh))
        pygame.draw.rect(self.screen, TEXT_COLOR, (x, y, w, h), 1)

    def draw_map(self):
        # Draw track points with speed-based coloring
        if len(self.track_points) > 1:
            for i in range(1, len(self.track_points)):
                p1 = self.track_points[i-1]
                p2 = self.track_points[i]
                
                sp1 = self._world_to_screen(p1[0], p1[1])
                sp2 = self._world_to_screen(p2[0], p2[1])
                
                # Speed coloring (blue = slow, red = fast, 0-4 m/s limits)
                speed = p2[2]
                pct = max(0.0, min(1.0, speed / 4.0))
                r = int(255 * pct)
                b = int(255 * (1-pct))
                color = (r, 50, b)
                
                pygame.draw.line(self.screen, color, sp1, sp2, 3)

        # Draw collisions as red X
        for cx, cy in self.collisions:
            scx, scy = self._world_to_screen(cx, cy)
            s = 6
            pygame.draw.line(self.screen, ACCENT_RED, (scx-s, scy-s), (scx+s, scy+s), 2)
            pygame.draw.line(self.screen, ACCENT_RED, (scx-s, scy+s), (scx+s, scy-s), 2)

        if not self.history: return
        state = self.history[-1]
        
        # Draw Car
        cx, cy = self._world_to_screen(state['x'], state['y'])
        heading = state['heading']
        
        # Car shape (triangle)
        size = 15
        # Heading is inverted for pygame Y
        h_pg = -heading 
        p1 = (cx + size * math.cos(h_pg), cy + size * math.sin(h_pg))
        p2 = (cx + size * 0.6 * math.cos(h_pg + 2.5), cy + size * 0.6 * math.sin(h_pg + 2.5))
        p3 = (cx + size * 0.6 * math.cos(h_pg - 2.5), cy + size * 0.6 * math.sin(h_pg - 2.5))
        
        pygame.draw.polygon(self.screen, ACCENT_YELLOW, [p1, p2, p3])
        pygame.draw.circle(self.screen, ACCENT_YELLOW, (cx, cy), 3)

    def draw_dashboard(self):
        ox = int(WINDOW_WIDTH * 0.6)
        w = WINDOW_WIDTH - ox - 20
        
        # Background panel
        pygame.draw.rect(self.screen, PANEL_BG, (ox, 20, w, WINDOW_HEIGHT - 40), border_radius=10)
        
        self.draw_text("LIVE TELEMETRY", self.font_large, ACCENT_BLUE, ox + w//2, 40, "center")
        
        if not self.history:
            self.draw_text("Waiting for data...", self.font_mid, TEXT_COLOR, ox + w//2, 100, "center")
            return
            
        state = self.history[-1]
        
        # --- Lap Info ---
        ly = 100
        self.draw_text(f"LAP {state['lap']}", self.font_large, TEXT_COLOR, ox + 20, ly)
        self.draw_text(f"Time: {state['lap_time']:.2f}s", self.font_mid, TEXT_COLOR, ox + 20, ly + 40)
        self.draw_text(f"Best: {state['best_lap']:.2f}s", self.font_mid, ACCENT_GREEN, ox + w - 20, ly + 40, "right")
        
        # --- Collisions ---
        cy = ly + 80
        col_color = ACCENT_RED if state['collisions'] > 0 else ACCENT_GREEN
        self.draw_text(f"COLLISIONS: {state['collisions']}", self.font_mid, col_color, ox + 20, cy)
        
        # --- Speed & Commands ---
        sy = cy + 60
        self.draw_text("SPEED (m/s)", self.font_small, TEXT_COLOR, ox + 20, sy)
        self.draw_text(f"{state['speed']:.2f}", self.font_mid, TEXT_COLOR, ox + w - 20, sy, "right")
        self.draw_bar(ox + 20, sy + 20, w - 40, 20, state['speed'], 0, 5.0, ACCENT_BLUE)
        
        ty = sy + 60
        self.draw_text("THROTTLE", self.font_small, TEXT_COLOR, ox + 20, ty)
        self.draw_text(f"{state['throttle']:.2f}", self.font_mid, TEXT_COLOR, ox + w - 20, ty, "right")
        self.draw_bar(ox + 20, ty + 20, w - 40, 20, state['throttle'], 0, 1.0, ACCENT_GREEN)
        
        sty = ty + 60
        self.draw_text("STEERING", self.font_small, TEXT_COLOR, ox + 20, sty)
        self.draw_text(f"{state['steering']:.2f}", self.font_mid, TEXT_COLOR, ox + w - 20, sty, "right")
        
        # Steering is centered bar -1 to 1
        pygame.draw.rect(self.screen, GRID_COLOR, (ox + 20, sty + 20, w - 40, 20))
        center_x = ox + 20 + (w - 40) // 2
        steer_px = int(state['steering'] * (w - 40) / 2) # steering is -1.0 to 1.0
        if steer_px < 0:
            pygame.draw.rect(self.screen, ACCENT_YELLOW, (center_x + steer_px, sty + 20, -steer_px, 20))
        else:
            pygame.draw.rect(self.screen, ACCENT_YELLOW, (center_x, sty + 20, steer_px, 20))
        pygame.draw.line(self.screen, TEXT_COLOR, (center_x, sty + 15), (center_x, sty + 45), 2)
        pygame.draw.rect(self.screen, TEXT_COLOR, (ox + 20, sty + 20, w - 40, 20), 1)

        # --- Wall Proximity (LiDAR) ---
        ly = sty + 80
        self.draw_text("WALL PROXIMITY (m)", self.font_small, TEXT_COLOR, ox + w//2, ly, "center")
        
        bw = 40
        bh = 100
        bx = ox + w//2 - (bw*1.5 + 20)
        by = ly + 30
        
        # Left
        self.draw_vertical_bar(bx, by, bw, bh, state['rl'], 0, 5.0, (100, 100, 255))
        self.draw_text("L", self.font_small, TEXT_COLOR, bx + bw//2, by + bh + 5, "center")
        self.draw_text(f"{state['rl']:.1f}", self.font_small, TEXT_COLOR, bx + bw//2, by - 20, "center")
        
        # Front
        bx += bw + 20
        self.draw_vertical_bar(bx, by, bw, bh, state['rf'], 0, 5.0, (255, 100, 100))
        self.draw_text("F", self.font_small, TEXT_COLOR, bx + bw//2, by + bh + 5, "center")
        self.draw_text(f"{state['rf']:.1f}", self.font_small, TEXT_COLOR, bx + bw//2, by - 20, "center")
        
        # Right
        bx += bw + 20
        self.draw_vertical_bar(bx, by, bw, bh, state['rr'], 0, 5.0, (100, 255, 100))
        self.draw_text("R", self.font_small, TEXT_COLOR, bx + bw//2, by + bh + 5, "center")
        self.draw_text(f"{state['rr']:.1f}", self.font_small, TEXT_COLOR, bx + bw//2, by - 20, "center")

    def run(self):
        running = True
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    
            # Read new telemetry
            self.read_telemetry()
            
            # Draw
            self.screen.fill(BG_COLOR)
            self.draw_map()
            self.draw_dashboard()
            
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == '__main__':
    try:
        app = LiveDashboard()
        app.run()
    except Exception as e:
        print(f"Dashboard error: {e}")
