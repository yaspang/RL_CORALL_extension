import math
from collections import deque

import numpy as np
import pygame

WIDTH, HEIGHT = 800, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
PADDING = 0.8

BG_COLOR     = (10,  10,  30)
GRID_COLOR   = (30,  35,  55)
OWN_COLOR    = (60,  220,  90)
INTR_COLOR   = (220,  60,  60)
SEP_COLOR    = (220, 180,  40,  80)
SEP_EDGE     = (220, 180,  40, 160)

GRID_SPACING_M = 500.0
TRAIL_LEN      = 400

# Ship outline: bow = +x, starboard = -y, normalised to LOA=1
_SHIP_TEMPLATE = np.array([
    [ 0.50,  0.00],
    [ 0.25, -0.18],
    [-0.40, -0.18],
    [-0.50,  0.00],
    [-0.40,  0.18],
    [ 0.25,  0.18],
])


def _vessel_color(idx: int, focus: int = 0) -> tuple[int, int, int]:
    return OWN_COLOR if idx == focus else INTR_COLOR


def _dim(color: tuple, factor: float = 0.35) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)


class Renderer:
    def __init__(self, loa_m: float = 30.0, fps: int = 30):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("CORALL")
        self.clock = pygame.time.Clock()
        self.loa_m = loa_m
        self.fps = fps
        self._zoom = 1.0
        self.events: list = []
        self.alive: bool = True
        self._font = pygame.font.SysFont("monospace", 14)
        self._font_small = pygame.font.SysFont("monospace", 11)
        self._trails: dict[int, deque] = {}

    # ------------------------------------------------------------------
    def _poll_events(self) -> None:
        new_events = pygame.event.get()
        self.events.extend(new_events)
        for event in new_events:
            if event.type == pygame.QUIT:
                self.alive = False
            if event.type == pygame.MOUSEWHEEL:
                self._zoom = max(0.1, self._zoom * (1.1 if event.y > 0 else 0.9))

    def _compute_scale(self, ox: float, oy: float, targets: list) -> float:
        dists = [500.0]
        for tgt in targets:
            dists.append(np.hypot(tgt.state.x - ox, tgt.state.y - oy))
        return (WIDTH / 2 * PADDING) / max(dists)

    def _to_screen(self, x: float, y: float, ox: float, oy: float, scale: float) -> tuple[int, int]:
        return int(CENTER[0] + (x - ox) * scale), int(CENTER[1] - (y - oy) * scale)

    # ------------------------------------------------------------------
    def _draw_grid(self, ox: float, oy: float, scale: float) -> None:
        spacing_px = GRID_SPACING_M * scale
        if spacing_px < 10:
            return
        x_start = math.floor((ox - CENTER[0] / scale) / GRID_SPACING_M) * GRID_SPACING_M
        x = x_start
        while True:
            sx, _ = self._to_screen(x, 0, ox, oy, scale)
            if sx > WIDTH:
                break
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, HEIGHT), 1)
            x += GRID_SPACING_M
        y_start = math.ceil((oy + CENTER[1] / scale) / GRID_SPACING_M) * GRID_SPACING_M
        y = y_start
        while True:
            _, sy = self._to_screen(0, y, ox, oy, scale)
            if sy > HEIGHT:
                break
            pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (WIDTH, sy), 1)
            y -= GRID_SPACING_M

    def _draw_trail(self, idx: int, ox: float, oy: float, scale: float, focus: int = 0) -> None:
        trail = self._trails.get(idx)
        if not trail or len(trail) < 2:
            return
        color = _dim(_vessel_color(idx, focus), 0.5)
        pts = [self._to_screen(x, y, ox, oy, scale) for x, y in trail]
        pygame.draw.lines(self.screen, color, False, pts, 1)

    def _draw_vessel(self, surface, state, idx: int, ox: float, oy: float, scale: float,
                     ghost: bool = False, focus: int = 0) -> None:
        color = _dim(_vessel_color(idx, focus), 0.25) if ghost else _vessel_color(idx, focus)
        sx, sy = self._to_screen(state.x, state.y, ox, oy, scale)
        size = max(8.0, self.loa_m * scale)
        c, s = np.cos(state.psi), np.sin(state.psi)
        rot = np.array([[c, -s], [s, c]])
        pts = [(sx + lx, sy - ly) for lx, ly in (_SHIP_TEMPLATE * size) @ rot.T]
        if not ghost:
            pygame.draw.polygon(surface, color, pts)
        pygame.draw.polygon(surface, color, pts, 1)
        label_color = _dim((255, 255, 255), 0.4) if ghost else (255, 255, 255)
        label = self._font_small.render(str(idx), True, label_color)
        surface.blit(label, (sx - label.get_width() // 2, sy - int(size * 0.7) - label.get_height()))

    def _draw_sep_radius(self, surface, state, ox: float, oy: float, scale: float) -> None:
        sx, sy = self._to_screen(state.x, state.y, ox, oy, scale)
        r = max(4, int(2 * self.loa_m * scale))
        pygame.draw.circle(surface, SEP_COLOR, (sx, sy), r)
        pygame.draw.circle(surface, SEP_EDGE, (sx, sy), r, 1)

    def _draw_waypoint(self, waypoint: np.ndarray, vessel_x: float, vessel_y: float,
                        ox: float, oy: float, scale: float, color: tuple, label_prefix: str = "") -> None:
        dist_km = np.hypot(waypoint[0] - vessel_x, waypoint[1] - vessel_y) / 1000.0
        text = self._font.render(f"{label_prefix}{dist_km:.1f}km", True, color)
        wx, wy = self._to_screen(waypoint[0], waypoint[1], ox, oy, scale)
        if 0 <= wx <= WIDTH and 0 <= wy <= HEIGHT:
            pygame.draw.circle(self.screen, color, (wx, wy), 5)
            self.screen.blit(text, (wx + 6, wy - 8))
        else:
            angle = np.arctan2(waypoint[1] - oy, waypoint[0] - ox)
            margin = 30
            tip = np.array([
                int(CENTER[0] + (WIDTH  / 2 - margin) * np.cos(angle)),
                int(CENTER[1] - (HEIGHT / 2 - margin) * np.sin(angle)),
            ])
            left  = tip + np.array([int(10 * np.cos(angle + 2.4)), int(-10 * np.sin(angle + 2.4))])
            right = tip + np.array([int(10 * np.cos(angle - 2.4)), int(-10 * np.sin(angle - 2.4))])
            pygame.draw.polygon(self.screen, color, [tip, left, right])
            self.screen.blit(text, (tip[0] + 12, tip[1] - 8))

    # ------------------------------------------------------------------
    def draw(
        self,
        own,
        targets: list,
        waypoint: np.ndarray | None = None,
        step: int = 0,
        waypoints: list[np.ndarray | None] | None = None,
        center_vessel=None,
        arrived: set | None = None,
        focus_idx: int = 0,
    ) -> None:
        """
        center_vessel: vessel to center the view on (defaults to own).
        arrived: set of vessel indices (in [own]+targets ordering) to draw as ghosts.
        waypoints: per-vessel list [own_wp, tgt0_wp, ...].
        focus_idx: index of the currently controlled vessel — drawn in green.
        """
        arrived = arrived or set()
        self._poll_events()
        self.screen.fill(BG_COLOR)

        center = center_vessel if center_vessel is not None else own
        ox, oy = center.state.x, center.state.y

        active_targets = [t for i, t in enumerate(targets, start=1) if i not in arrived]
        scale = self._compute_scale(ox, oy, active_targets) * self._zoom

        self._draw_grid(ox, oy, scale)

        all_vessels = [own] + targets

        # Update trails for active vessels only
        for i, v in enumerate(all_vessels):
            if i in arrived:
                continue
            if i not in self._trails:
                self._trails[i] = deque(maxlen=TRAIL_LEN)
            self._trails[i].append((v.state.x, v.state.y))

        # Draw trails
        for i in range(len(all_vessels)):
            if i not in arrived:
                self._draw_trail(i, ox, oy, scale, focus=focus_idx)

        # Draw separation radii for active vessels
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i, v in enumerate(all_vessels):
            if i not in arrived:
                self._draw_sep_radius(overlay, v.state, ox, oy, scale)
        self.screen.blit(overlay, (0, 0))

        # Draw waypoints for active vessels
        if waypoints is not None:
            for i, (v, wp) in enumerate(zip(all_vessels, waypoints)):
                if wp is not None and i not in arrived:
                    self._draw_waypoint(wp, v.state.x, v.state.y, ox, oy, scale,
                                        color=_vessel_color(i, focus_idx), label_prefix=f"{i}:")
        elif waypoint is not None and 0 not in arrived:
            self._draw_waypoint(waypoint, own.state.x, own.state.y, ox, oy, scale,
                                color=_vessel_color(0, focus_idx), label_prefix="0:")

        # Draw vessels — arrived ones as ghosts, focused vessel in green
        for i, v in enumerate(all_vessels):
            self._draw_vessel(self.screen, v.state, i, ox, oy, scale, ghost=(i in arrived), focus=focus_idx)

        step_label = self._font.render(f"t = {step}", True, (160, 160, 160))
        self.screen.blit(step_label, (WIDTH - step_label.get_width() - 10, HEIGHT - 24))

    @property
    def interrupted(self) -> bool:
        return not self.alive or any(e.type == pygame.KEYDOWN for e in self.events)

    def flip(self) -> None:
        pygame.display.flip()
        self.clock.tick(self.fps)

    def reset(self) -> None:
        self._trails.clear()

    def close(self) -> None:
        pygame.quit()
        self.alive = False
