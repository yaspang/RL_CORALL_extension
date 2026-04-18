import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OWN_COLOR = "#3cdc5a"
TGT_COLOR = "#dc3c3c"
GOAL_COLOR = "#50ff78"
ARROW_LEN = 150.0


class PlotRenderer:
    def __init__(self, loa_m: float = 30.0):
        self.loa_m = loa_m
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.patch.set_facecolor("#0a0a1e")
        self.ax.set_facecolor("#0a0a1e")
        self.ax.tick_params(colors="gray")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#1e2337")
        self._own_trail: list[tuple] = []
        self._tgt_trails: dict[int, list[tuple]] = {}

    def draw(self, own, targets, waypoint=None, step: int = 0) -> None:
        self._own_trail.append((own.state.x, own.state.y))
        for i, tgt in enumerate(targets):
            self._tgt_trails.setdefault(i, []).append((tgt.state.x, tgt.state.y))

        self.ax.cla()
        self.ax.set_facecolor("#0a0a1e")
        self.ax.set_aspect("equal")
        self.ax.grid(color="#1e2337", linewidth=0.5)
        self.ax.tick_params(colors="gray")

        if len(self._own_trail) > 1:
            xs, ys = zip(*self._own_trail)
            self.ax.plot(xs, ys, color=OWN_COLOR, linewidth=1.0, alpha=0.6)

        for i, tgt in enumerate(targets):
            trail = self._tgt_trails.get(i, [])
            if len(trail) > 1:
                xs, ys = zip(*trail)
                self.ax.plot(xs, ys, color=TGT_COLOR, linewidth=1.0, alpha=0.6)
            self._draw_vessel(tgt.state, TGT_COLOR)
            self.ax.add_patch(plt.Circle(
                (tgt.state.x, tgt.state.y), 2 * self.loa_m,
                color=TGT_COLOR, alpha=0.1, linewidth=0,
            ))

        self._draw_vessel(own.state, OWN_COLOR)
        self.ax.add_patch(plt.Circle(
            (own.state.x, own.state.y), 2 * self.loa_m,
            color=OWN_COLOR, alpha=0.1, linewidth=0,
        ))

        if waypoint is not None:
            self.ax.plot(*waypoint, marker="*", color=GOAL_COLOR, markersize=12)

        self.ax.set_title(f"t = {step}", color="gray", fontsize=10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_vessel(self, state, color) -> None:
        dx = ARROW_LEN * np.cos(state.psi)
        dy = ARROW_LEN * np.sin(state.psi)
        self.ax.annotate(
            "", xy=(state.x + dx, state.y + dy), xytext=(state.x, state.y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
        )
        self.ax.plot(state.x, state.y, "o", color=color, markersize=5)

    def flip(self) -> None:
        pass  # matplotlib updates happen inside draw()

    @property
    def events(self) -> list:
        return []

    @property
    def interrupted(self) -> bool:
        return False

    def reset(self) -> None:
        self._own_trail = []
        self._tgt_trails = {}

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)
