"""
Procedural encounter environment — fully geometry-based, NOT case-dependent.

Generates encounters backwards from a defined crossing point:
  1. Fix ownship route: (0, 0) heading east at ownship_speed_mps
  2. Choose crossing point x = cross_x_nmi (±20% noise), y = 0
  3. t_cross = crossing_x_m / ownship_speed_mps
  4. Sample target speed and encounter angle theta
  5. Place target at: crossing_pt - t_cross * speed * (cos(theta), sin(theta))
     → target arrives at crossing exactly when ownship does

Imazu case numbers are used ONLY as structural templates (for agent count /
action-observation space sizing). All geometry is replaced with the
backwards-computed positions above, so training is fully case-independent.

Usage:
    env = RandomEncounterEnv(
        ownship_speed_mps=10.0,         # fixed across all episodes
        target_speed_range=(6.0, 14.0), # randomly sampled each episode
        desired_cross_x_nmi=1.0,        # crossing distance (±20% jitter)
        master_seed=42,
        verbose=True,
    )
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, List
import numpy as np
import gymnasium as gym
from pathlib import Path

from .env_single_agent_sb3 import SingleAgentOwnshipEnv


class RandomEncounterEnv(gym.Wrapper):
    """
    Procedural encounter environment with guaranteed crossings.

    Generates encounters backwards from a crossing point — no Imazu case
    geometry is used.  Case numbers serve only as structural templates so that
    the underlying MultiShipParallelEnv has the correct agent count and
    action/observation space dimensions.

    Args:
        ownship_speed_mps:      Fixed ownship speed [m/s] (default 10.0).
        target_speed_range:     (min, max) target speed per episode [m/s].
        desired_cross_x_nmi:    Nominal crossing distance along ownship route
                                [NMI]; jittered ±20% each episode.
        num_agents_curriculum:  Max total agents per curriculum phase
                                (default [2, 3, 4]).
        dt / sim_time / …:      Passed through to the underlying env.
        master_seed:            RNG seed for reproducibility.
        verbose:                Print one summary line per episode reset.
    """

    MAX_OBS_SIZE = 29

    # ---- OLD hard-phase curriculum (commented out, preserved for reference) ----
    # DEFAULT_PHASES = [
    #     (0,       2),   # 0-1M steps: 2-ship only (1 obstacle)
    #     (1000000, 3),   # 1M-2M steps: up to 3-ship (1-2 obstacles, uniform)
    #     (2000000, 4),   # 2M+  steps: up to 4-ship (1-3 obstacles, uniform)
    # ]

    # ---- NEW stochastic mixed curriculum (active) ----
    # Each phase: (step_threshold, [p_1obs, p_2obs, p_3obs])
    #   0-250k :  80% one target, 15% two targets,  5% three targets
    #   250k-750k: 60% one target, 30% two targets, 10% three targets
    #   750k+  :  40% one target, 40% two targets, 20% three targets
    STOCHASTIC_PHASES = [
        (0,       [0.80, 0.15, 0.05]),
        (250_000, [0.60, 0.30, 0.10]),
        (750_000, [0.40, 0.40, 0.20]),
    ]

    def __init__(
        self,
        ownship_speed_mps: float = 10.0,
        target_speed_range: Tuple[float, float] = (6.0, 14.0),
        desired_cross_x_nmi: float = 1.0,
        # num_agents_curriculum: Optional[List[int]] = None,  # old hard-phase arg, replaced by STOCHASTIC_PHASES
        dt: float = 0.5,
        sim_time: float = 490.0,
        n_heading: int = 7,
        n_speed: int = 5,
        max_heading_change_deg: float = 25.0,
        loa_m: float = 30.0,
        route_len_nmi: float = 2.0,
        master_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        # Build a starter base env (case 1, 1 obstacle) for gym.Wrapper init
        base_env = SingleAgentOwnshipEnv(
            case_number=1,
            dt=dt, sim_time=sim_time, n_heading=n_heading,
            n_speed=n_speed,
            max_heading_change_deg=max_heading_change_deg,
            loa_m=loa_m, route_len_nmi=route_len_nmi,
            seed=0,
            desired_cross_x_nmi=desired_cross_x_nmi,
            target_speed_mps=target_speed_range[0],
            ownship_speed_mps=ownship_speed_mps,
        )
        super().__init__(base_env)

        self.ownship_speed_mps = float(ownship_speed_mps)
        self.target_speed_min, self.target_speed_max = float(target_speed_range[0]), float(target_speed_range[1])
        self.desired_cross_x_nmi_base = float(desired_cross_x_nmi)
        self.verbose = bool(verbose)

        self.dt = float(dt)
        self.sim_time = float(sim_time)
        self.n_heading = int(n_heading)
        self.n_speed = int(n_speed)
        self.max_heading_change_deg = float(max_heading_change_deg)
        self.loa_m = float(loa_m)
        self.route_len_nmi = float(route_len_nmi)

        self.rng = np.random.default_rng(master_seed)

        # Old hard-phase init (commented out)
        # if num_agents_curriculum is None:
        #     self.curriculum_phases = self.DEFAULT_PHASES
        # else:
        #     thresholds = [0, 1_000_000, 2_000_000]
        #     self.curriculum_phases = [
        #         (thresholds[i], num_agents_curriculum[i])
        #         for i in range(len(num_agents_curriculum))
        #     ]

        self.current_step = 0
        self.episode_count = 0
        self.current_num_agents = 2
        self.encounter_params = {}

        # Build map: n_obstacles -> template case number (queried once at init)
        self._template_case_map = self._build_template_case_map()

        # Padded observation space
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.5e4, high=1.5e4,
            shape=(self.MAX_OBS_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    # def _get_max_agents_for_phase(self) -> int:
    #     """OLD: Max total agents (ownship + obstacles) for current training step."""
    #     max_agents = 2
    #     for threshold, agents in self.curriculum_phases:
    #         if self.current_step >= threshold:
    #             max_agents = agents
    #     return max_agents

    def _get_n_obstacles_weights(self) -> list:
        """Return [p_1obs, p_2obs, p_3obs] for the current training step."""
        weights = self.STOCHASTIC_PHASES[0][1]  # default: first phase
        for threshold, phase_weights in self.STOCHASTIC_PHASES:
            if self.current_step >= threshold:
                weights = phase_weights
        return weights

    # ------------------------------------------------------------------
    # Template case map (built once at init)
    # ------------------------------------------------------------------

    def _build_template_case_map(self) -> dict:
        """
        Scan Imazu cases 1-22 and return {n_obstacles: first_case_with_that_count}.

        These cases are used ONLY as structural templates (obs/action space
        dimensions).  All geometry is overridden in reset().
        """
        from utils.imazu_cases import get_obstacle_data as _gob
        template: dict = {}
        for case in range(1, 23):
            try:
                Xob, _, _, _ = _gob(case, synchronize_arrivals=False, target_speed_mps=10.0)
                n = len(Xob)
                if n not in template:
                    template[n] = case
            except Exception:
                pass
        return template

    # ------------------------------------------------------------------
    # Geometry generation (backwards from crossing point)
    # ------------------------------------------------------------------

    def _generate_encounter_geometry(
        self,
        n_obstacles: int,
        target_speed_mps: float,
        cross_x_nmi: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Place n_obstacles so each arrives at the crossing point at t_cross.

        Algorithm (per obstacle):
          t_cross = cross_x_m / ownship_speed_mps
          theta   ~ Uniform([15°, 345°])   # avoids pure parallel overtake
          dist    = t_cross * target_speed_mps
          x_start = cross_x_m - dist * cos(theta)
          y_start =          0 - dist * sin(theta)
          heading = theta  (points straight toward crossing)

        Returns (Xob, Yob, Vob, psiob, angles_deg).
        """
        NMI_M = 1852.0
        cross_x_m = cross_x_nmi * NMI_M
        t_cross = cross_x_m / self.ownship_speed_mps
        dist = t_cross * target_speed_mps

        # Angle exclusion zone around θ=0 (pure overtake from behind)
        lo_deg, hi_deg = 15.0, 345.0  # valid range [15°, 345°]

        Xob_list, Yob_list, Vob_list, psiob_list, angles_deg = [], [], [], [], []
        for _ in range(n_obstacles):
            theta_deg = float(self.rng.uniform(lo_deg, hi_deg))
            theta = np.radians(theta_deg)

            x_start = cross_x_m - dist * np.cos(theta)
            y_start = 0.0       - dist * np.sin(theta)

            Xob_list.append(x_start)
            Yob_list.append(y_start)
            Vob_list.append(target_speed_mps)
            psiob_list.append(theta)
            angles_deg.append(round(theta_deg, 1))

        return (
            np.array(Xob_list, dtype=float),
            np.array(Yob_list, dtype=float),
            np.array(Vob_list, dtype=float),
            np.array(psiob_list, dtype=float),
            angles_deg,
        )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset with a freshly generated encounter.

        Steps:
          1. Sample n_obstacles, target_speed, cross_x (±20% jitter)
          2. Pick template case for the right agent count
          3. Create SingleAgentOwnshipEnv with that template
          4. Override _case_cache with backwards-computed geometry
          5. Call env.reset() → init_from_case() uses the new geometry
        """
        # --- curriculum: stochastic mixed sampling ---
        # Old hard-phase sampling (commented out):
        # max_agents = self._get_max_agents_for_phase()
        # n_obstacles = int(self.rng.integers(1, max_agents))   # 1 … max_agents-1
        weights = self._get_n_obstacles_weights()  # [p_1obs, p_2obs, p_3obs]
        n_obstacles = int(self.rng.choice([1, 2, 3], p=weights))

        # --- encounter params ---
        cross_x_nmi = float(self.rng.uniform(
            self.desired_cross_x_nmi_base * 0.8,
            self.desired_cross_x_nmi_base * 1.2,
        ))
        target_speed = float(self.rng.uniform(self.target_speed_min, self.target_speed_max))

        # --- episode seed ---
        ep_seed = int(seed) if seed is not None else int(self.rng.integers(0, 1_000_000))
        self.current_seed = ep_seed
        self.current_num_agents = n_obstacles

        # --- template case (just for agent count / space dimensions) ---
        template_case = self._template_case_map.get(n_obstacles, 1)

        # Create env with template case (geometry will be replaced below)
        self.env = SingleAgentOwnshipEnv(
            case_number=template_case,
            dt=self.dt,
            sim_time=self.sim_time,
            n_heading=self.n_heading,
            n_speed=self.n_speed,
            max_heading_change_deg=self.max_heading_change_deg,
            loa_m=self.loa_m,
            route_len_nmi=self.route_len_nmi,
            seed=ep_seed,
            desired_cross_x_nmi=cross_x_nmi,
            target_speed_mps=target_speed,
            ownship_speed_mps=self.ownship_speed_mps,
        )

        # --- override _case_cache with generated geometry ---
        Xob, Yob, Vob, psiob, angles_deg = self._generate_encounter_geometry(
            n_obstacles=n_obstacles,
            target_speed_mps=target_speed,
            cross_x_nmi=cross_x_nmi,
        )
        em = self.env.env_multi
        em._case_cache['Xob'] = Xob
        em._case_cache['Yob'] = Yob
        em._case_cache['Vob'] = Vob
        em._case_cache['psiob'] = psiob

        # --- reset (init_from_case uses the updated _case_cache) ---
        self.episode_count += 1
        obs, info = self.env.reset(seed=ep_seed)
        obs = self._pad_observation(obs)

        # --- verbose ---
        if self.verbose:
            spd_str = str([round(v, 1) for v in Vob.tolist()])
            ang_str = str(angles_deg)
            print(
                f"episode={self.episode_count:4d} "
                f"n_obstacles={n_obstacles} "
                f"ownship_speed={self.ownship_speed_mps:.1f} "
                f"target_speeds={spd_str} "
                f"crossing_x={cross_x_nmi:.3f} "
                f"angles={ang_str}"
            )

        # Store for inspection
        self.encounter_params = {
            'n_obstacles':        n_obstacles,
            'target_speed_mps':   target_speed,
            'desired_cross_x_nmi': cross_x_nmi,
            'angles_deg':         angles_deg,
            'template_case':      template_case,
        }

        info['num_agents']       = n_obstacles
        info['seed']             = ep_seed
        info['episode']          = self.episode_count
        info['encounter_params'] = self.encounter_params

        return obs, info

    # ------------------------------------------------------------------
    # step / helpers
    # ------------------------------------------------------------------

    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to MAX_OBS_SIZE."""
        n = len(obs)
        if n == self.MAX_OBS_SIZE:
            return obs.astype(np.float32)
        padded = np.zeros(self.MAX_OBS_SIZE, dtype=np.float32)
        padded[:n] = obs
        return padded

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._pad_observation(obs)
        info['num_agents'] = self.current_num_agents
        return obs, reward, terminated, truncated, info

    def update_step(self, step: int):
        """Call from training loop to advance curriculum gating."""
        self.current_step = step

    @property
    def env_multi(self):
        if hasattr(self.env, 'env_multi'):
            return self.env.env_multi
        raise RuntimeError("env_multi not accessible")

