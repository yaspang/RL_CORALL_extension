import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

from dynamics import Vessel, VesselState
from cases import parse_case, MAX_INTRUDERS


KP_YAW = 250.0
KD_YAW = -5000.0

OWN_DIM  = 4                              # [cos ψ, sin ψ, r, u]
INTR_DIM = 7                              # [x_rel, y_rel, vx_rel, vy_rel, sin_dpsi, cos_dpsi, dr]
GOAL_DIM = 2                              # [cos Δψ_goal, sin Δψ_goal]
OBS_DIM  = OWN_DIM + MAX_INTRUDERS * INTR_DIM + GOAL_DIM


class CORALLEnv(gym.Env):
    """Training environment — randomised encounter geometry and waypoint each episode.

    With round_robin=True the agent controls every vessel in turn each timestep.
    Each step() call collects one vessel's action; physics advances once all
    vessels have submitted actions.  Intruder vessels navigate toward a
    waypoint straight ahead of their initial heading.
    """

    metadata = {"render_modes": ["human", "plot"]}

    def __init__(
        self,
        dt: float = 0.5,
        step_size: int = 1,
        render_every: int = 1,
        fps: int = 30,
        render_mode: Optional[str] = None,
        max_heading_delta_deg: float = 25.0,
        spd_min: float = 6.0,
        spd_max: float = 12.0,
        loa_m: float = 200.0,
        max_intruders: int = MAX_INTRUDERS,
        encounter_range_m: float = 10_000.0,
        waypoint_range_m: tuple[float, float] = (5_000.0, 20_000.0),
        round_robin: bool = False,
    ):
        super().__init__()
        self.dt = dt
        self.step_size = step_size
        self.render_every = render_every
        self.fps = fps
        self.render_mode = render_mode
        self.loa_m = loa_m
        self.spd_min = spd_min
        self.spd_max = spd_max
        self.spd_mid = (spd_min + spd_max) / 2.0
        self.max_hdg_delta = np.deg2rad(max_heading_delta_deg)
        self.max_intruders = max_intruders
        self.encounter_range_m = encounter_range_m
        self.waypoint_range_m = waypoint_range_m
        self.round_robin = round_robin

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self._renderer = None
        self._rr_active: list[int] = []
        self._rr_arrived: set[int] = set()
        self._rr_sub_idx: int = 0
        self._rr_actions: dict[int, np.ndarray] = {}
        self._rr_waypoints: list[np.ndarray] = []
        self.current_vessel: int = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.step_count = 0

        if self._renderer is not None:
            self._renderer.reset()

        psi0 = self._rng.uniform(0, 2 * np.pi)
        self.own = Vessel(state=VesselState(x=0.0, y=0.0, psi=psi0, u=self.spd_mid))
        self.waypoint = self._sample_waypoint()
        self.intruders = self._sample_intruders()
        self._prev_dist = np.hypot(self.waypoint[0], self.waypoint[1])

        if self.round_robin:
            self._setup_rr()

        return self._get_obs(), {}

    def _setup_rr(self) -> None:
        n = len(self._all_vessels())
        self._rr_waypoints = [self.waypoint] + [self._waypoint_ahead(v) for v in self.intruders]
        self._rr_active = list(range(n))
        self._rr_arrived = set()
        self._rr_sub_idx = 0
        self._rr_actions = {}
        self.current_vessel = 0

    def _waypoint_ahead(self, vessel: Vessel, dist: float = 30_000.0) -> np.ndarray:
        return np.array([
            vessel.state.x + dist * np.cos(vessel.state.psi),
            vessel.state.y + dist * np.sin(vessel.state.psi),
        ])

    def _all_vessels(self) -> list[Vessel]:
        return [self.own] + self.intruders

    def _sample_waypoint(self) -> np.ndarray:
        r = self._rng.uniform(*self.waypoint_range_m)
        angle = self._rng.uniform(0, 2 * np.pi)
        return np.array([r * np.cos(angle), r * np.sin(angle)])

    def _sample_intruders(self) -> list[Vessel]:
        n = self._rng.integers(1, self.max_intruders + 1)
        wp = self.waypoint
        wp_dist = float(np.hypot(wp[0], wp[1]))
        route_dir = wp / wp_dist if wp_dist > 0 else np.array([1.0, 0.0])
        route_perp = np.array([-route_dir[1], route_dir[0]])

        intruders = []
        for _ in range(n):
            if self._rng.random() < 0.5:
                # Spawn near origin (original behaviour)
                r     = self._rng.uniform(500.0, self.encounter_range_m)
                angle = self._rng.uniform(0, 2 * np.pi)
                x, y  = r * np.cos(angle), r * np.sin(angle)
            else:
                # Spawn along the route at a random progress fraction
                frac    = self._rng.uniform(0.1, 0.9)
                lateral = self._rng.uniform(-self.encounter_range_m / 2, self.encounter_range_m / 2)
                pt      = frac * wp_dist * route_dir + lateral * route_perp
                x, y    = float(pt[0]), float(pt[1])

            psi = self._rng.uniform(0, 2 * np.pi)
            u   = self._rng.uniform(self.spd_min, self.spd_max)
            intruders.append(Vessel(state=VesselState(x=x, y=y, psi=psi, u=u), sigma_b=0.0))
        return intruders

    def _goal_bearing(self) -> float:
        dx = self.waypoint[0] - self.own.state.x
        dy = self.waypoint[1] - self.own.state.y
        return float(np.arctan2(dy, dx))

    # ------------------------------------------------------------------
    def step(self, action):
        if self.round_robin:
            return self._step_rr(action)

        hdg_action, spd_action = np.clip(action, -1.0, 1.0)
        u_c = self.spd_mid + spd_action * (self.spd_max - self.spd_min) / 2.0
        psi_ref = self._goal_bearing() + hdg_action * self.max_hdg_delta

        collision = False
        for substep in range(self.step_size):
            hdg_err = (psi_ref - self.own.state.psi + np.pi) % (2 * np.pi) - np.pi
            tau_c = KP_YAW * hdg_err + KD_YAW * self.own.state.r
            self.own.step(tau_c, u_c, self.dt, self._rng)
            for intr in self.intruders:
                intr.step(0.0, intr.state.u, self.dt, self._rng)
            if self.render_mode == "human" and substep % self.render_every == 0:
                self.render()
            if any(self.own.distance_to(intr) < 2.0 * self.loa_m for intr in self.intruders):
                collision = True
                break

        self.step_count += 1

        dist_to_goal = np.hypot(
            self.waypoint[0] - self.own.state.x,
            self.waypoint[1] - self.own.state.y,
        )
        collision = collision or any(self.own.distance_to(intr) < 2.0 * self.loa_m for intr in self.intruders)
        arrived = dist_to_goal < 2.0 * self.loa_m
        terminated = collision or arrived
        progress = self._prev_dist - dist_to_goal
        self._prev_dist = dist_to_goal
        reward = -100.0 if collision else (100.0 if arrived else -0.1)

        return self._get_obs(), reward, terminated, False, {}

    def _step_rr(self, action):
        all_v = self._all_vessels()
        current = self._rr_active[self._rr_sub_idx]

        self._rr_actions[current] = np.clip(action, -1.0, 1.0)
        self._rr_sub_idx += 1

        # Not all active vessels have acted yet
        if self._rr_sub_idx < len(self._rr_active):
            next_v = self._rr_active[self._rr_sub_idx]
            self.current_vessel = next_v
            return self._obs_for(next_v), 0.0, False, False, {}

        # All active vessels have acted — compute controls and advance physics
        controls = {}
        for i in self._rr_active:
            v = all_v[i]
            hdg_action, spd_action = self._rr_actions[i]
            u_c = self.spd_mid + spd_action * (self.spd_max - self.spd_min) / 2.0
            wp = self._rr_waypoints[i]
            goal_bearing = float(np.arctan2(wp[1] - v.state.y, wp[0] - v.state.x))
            controls[i] = (goal_bearing + hdg_action * self.max_hdg_delta, u_c)

        for substep in range(self.step_size):
            for i, v in enumerate(all_v):
                if i not in self._rr_arrived:
                    psi_ref, u_c = controls[i]
                    hdg_err = (psi_ref - v.state.psi + np.pi) % (2 * np.pi) - np.pi
                    tau_c = KP_YAW * hdg_err + KD_YAW * v.state.r
                    v.step(tau_c, u_c, self.dt, self._rng)
            if self.render_mode == "human" and substep % self.render_every == 0:
                self.render()

        self.step_count += 1
        self._rr_actions = {}

        # Collision only between active (non-arrived) vessels
        active_set = set(self._rr_active)
        collision = any(
            all_v[i].distance_to(all_v[j]) < 2.0 * self.loa_m
            for i in active_set for j in active_set if i < j
        )

        # Check which active vessels newly arrived
        newly_arrived = [
            i for i in self._rr_active
            if np.hypot(self._rr_waypoints[i][0] - all_v[i].state.x,
                        self._rr_waypoints[i][1] - all_v[i].state.y) < 2.0 * self.loa_m
        ]
        for i in newly_arrived:
            self._rr_arrived.add(i)
            self._rr_active.remove(i)

        all_arrived = len(self._rr_active) == 0
        terminated = collision or all_arrived

        reward = -1.0 if collision else (1.0 * len(newly_arrived) - 0.001 * len(self._rr_active))

        # Reset sub-index for next round
        self._rr_sub_idx = 0
        if self._rr_active:
            self.current_vessel = self._rr_active[0]
            return self._obs_for(self._rr_active[0]), reward, terminated, False, {}
        return self._obs_for(0), reward, terminated, False, {}

    # ------------------------------------------------------------------
    def _obs_for(self, vessel_idx: int) -> np.ndarray:
        all_v = self._all_vessels()
        active = all_v[vessel_idx]
        others = [v for i, v in enumerate(all_v)
                  if i != vessel_idx and i not in self._rr_arrived]
        wp = self._rr_waypoints[vessel_idx]
        own_s, intr_s, wp_s = self.own, self.intruders, self.waypoint
        self.own, self.intruders, self.waypoint = active, others, wp
        obs = self._get_obs_base()
        self.own, self.intruders, self.waypoint = own_s, intr_s, wp_s
        return obs

    def _get_obs_base(self) -> np.ndarray:
        dpsi = self._goal_bearing() - self.own.state.psi
        goal_obs = np.array([np.cos(dpsi), np.sin(dpsi)], dtype=np.float32)
        parts = [self.own.obs()]
        for i in range(MAX_INTRUDERS):
            if i < len(self.intruders):
                parts.append(self.own.relative_obs(self.intruders[i]))
            else:
                parts.append(np.zeros(INTR_DIM, dtype=np.float32))
        parts.append(goal_obs)
        return np.concatenate(parts)

    def _get_obs(self) -> np.ndarray:
        if self.round_robin and self._rr_waypoints and self._rr_active:
            return self._obs_for(self._rr_active[self._rr_sub_idx])
        return self._get_obs_base()

    def render(self):
        if self._renderer is None:
            if self.render_mode == "plot":
                from plot_renderer import PlotRenderer
                self._renderer = PlotRenderer(loa_m=self.loa_m)
            else:
                from renderer import Renderer
                self._renderer = Renderer(loa_m=self.loa_m, fps=self.fps)
        if self.round_robin and self._rr_waypoints:
            all_v = self._all_vessels()
            center = all_v[self.current_vessel] if self._rr_active else self.own
            arrived_set = self._rr_arrived
            waypoints = self._rr_waypoints
        else:
            center, arrived_set, waypoints = self.own, set(), None
        self._renderer.draw(self.own, self.intruders, self.waypoint, step=self.step_count,
                            waypoints=waypoints, center_vessel=center, arrived=arrived_set,
                            focus_idx=self.current_vessel)
        self._renderer.flip()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class ImazuEvalEnv(CORALLEnv):
    """Evaluation environment — steps through Imazu cases 1-23 in order."""

    N_CASES = 23

    def __init__(self, case: int | None = None, intr_speed: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self._fixed_case = case
        self._case_idx = 0
        self.intr_speed = intr_speed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.own.state.psi = np.pi / 2

        case = self._fixed_case if self._fixed_case is not None else (self._case_idx % self.N_CASES) + 1
        if self._fixed_case is None:
            self._case_idx += 1

        self.intruders = [
            Vessel(state=VesselState(**cfg), sigma_b=0.0)
            for cfg in parse_case(case, intr_speed=self.intr_speed)
        ]
        self.current_case = case

        if self.round_robin:
            self._setup_rr()

        return self._get_obs(), {"case": case, "vessel": self.current_vessel}
