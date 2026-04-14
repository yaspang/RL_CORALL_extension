import numpy as np

def compute_position_scale_from_WP(self, pad_frac: float=0.05) -> float:
    """Flatten all waypoints across all agents to define posiion scale for observations."""

    xs = np.array([x for route in self.Xwpt_all for x in route], dtype=float)
    ys = np.array([y for route in self.Ywpt_all for y in route], dtype=float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    width = x_max - x_min
    height = y_max - y_min
    diag = float(np.hypot(width, height))

    # small padding so values rarely hit exactly -1 or 1 due to potential dynamics overshoot
    diag *= (1 + pad_frac)

    return max(diag, 1.0)  # avoid zero division