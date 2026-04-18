## Minimal single-agent RL wrapper around CORALL's simulation loop to test installation. Prove structural / mechanical compatibility. 
#
# Goal: keep CORALL dynamics + planning + obstacle propagation intact, but replace reactive_avoidance with RL_action. 
#

# import libraries 
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# import CORALL wrappers
import sys 
import os 

from .path_setup import ensure_paths
ensure_paths()

from navigation.planning import waypoint_selection, planning
from dynamics.controller import controller 
from dynamics.actuator_modeling import actuator_modeling
from dynamics.vessel_dynamics import vessel_dynamics
from core.integration import integration
from navigation.obstacle_sim import obstacle_sim
from risk_assessment.cpa_calculations import cpa_calculations
from risk_assessment.risk_calculations import risk_calculations
from utils.imazu_cases_old import get_obstacle_data


class CORALL_ReactiveAvoidanceGymEnv(gym.Env):
    """
    Baby gymnasium wrapper around CORALL siulation loop to test simulation compatibility with RL frameworks.

    Goal: Keep CORALL planning + controll + dynamics, but replace reactive_avoidance() with an RL action
    """
    metadata = {"render_modes": []}

    def __init__(
            self, 
            case_number: int = 1,
            dt: float = 0.2,
            #sim_time: float = 300.0, 
            max_colav_deg: float = 25.0, 
            K_obstacles: int = 1, 
            u_cmd_mps: float = 10.0, 
            sat_amp_s: float = 20.0, 
            collision_dist_m: float = 60.0,
            goal_tol_nmi: float = 0.02, 
            min_steps: int = 300, 
            max_steps_cap: int = 20000,
            seed: Optional[int] = None,
    ):

        super().__init__()
        self.case_number = case_number
        self.dt = float(dt)
        # self.sim_time = float(sim_time)
        # self.max_steps_cap = int(np.ceil(self.sim_time / self.dt))

        self.max_colav_rad = np.deg2rad(max_colav_deg)                # rad  ---- saturation limit on heading change
        self.colav_rate = np.deg2rad(5)                               # rad/s --- how RL can steer for colav
        self.psi_colav = 0.0                                          # initialize state heading for colav
        self.K = int(K_obstacles)

        self.u_cmd_mps = float(u_cmd_mps)
        self.sat_amp_s = float(sat_amp_s)

        self.collision_dist_m = float(collision_dist_m)
        self.goal_tol_nmi = float(goal_tol_nmi)

        self.min_steps = int(min_steps)
        self.max_steps_cap = int(max_steps_cap)

        self.np_random = np.random.default_rng(seed)

        # 1D continuous action space: heading change in radians
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )

        # observation dim = 6 + K * 5
        obs_dim = 6 + self.K * 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # sim state
        self.t = 0
        self.step_idx = 0

        # ownship state X = [x, y, psi, r, b, u]
        self.X = np.zeros(6, dtype=float)
        self.ui_psi1 = 0.0  # previous commanded heading rate
        self.i_wpt = 1

        # waypoints in nautical miles (interface with CORALL)
        # simple straight light 
        self.Xwpt = [0, 12]
        self.Ywpt = [0, 0]

        # obstacles
        self.Xob = np.array([], dtype=float)
        self.Yob = np.array([], dtype=float)
        self.Vob = np.array([], dtype=float)
        self.psiob = np.array([], dtype=float)

        # obstacle previous positions for CPA analysis
        self.Xob_prev = np.array([], dtype=float)
        self.Yob_prev = np.array([], dtype=float)

        # progress reward helper
        self.prev_goal_dist = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        if not hasattr(self, "_printed_path"):
            print("DEBUG test_env.py path:", os.path.abspath(__file__))
            self._printed_path = True

        if seed is not None: 
            self.np_random = np.random.default_rng(seed)
    
        # reset sim state
        self.t = 0
        self.step_idx = 0
        self.i_wpt = 1
        self.ui_psi1 = 0.0     # previous commanded heading rate
        self.psi_colav = 0.0   

        # ownship state X = [x, y, psi, r, b, u]
        self.X[:] = np.array([0, 0, np.deg2rad(0), 0, 0, 0], dtype=float)

    
        # load obstacle data from Imazu cases
        Xob, Yob, Vob, psiob = get_obstacle_data(self.case_number)
        
        self.Xob = np.array(Xob, dtype=float)
        self.Yob = np.array(Yob, dtype=float)

        self.Vob = np.array(Vob, dtype=float)
        self.psiob = np.array(psiob, dtype=float)

        # debugging safe checks
        
        if self.Xob is None or len(self.Xob) == 0:
            print(f"DEBUG: no obstacles returned for case_number={self.case_number}")
        else:
            print("DEBUG Xob[0],Yob[0] =", self.Xob[0], self.Yob[0])
            
        self._x_prev = float(self.X[0])
        self._y_prev = float(self.X[1])

        # initialize previous obstacle positions
        self.Xob_prev = np.copy(self.Xob)
        self.Yob_prev = np.copy(self.Yob)

        # compute initial distance to goal for reward calculation
        goal_x = self.Xwpt[-1]
        goal_y = self.Ywpt[-1]
        self.prev_goal_dist = np.hypot(goal_x - self.X[0] / 1852, goal_y - self.X[1] / 1852)

        self.max_steps = self.set_adaptive_horizon()   # no assignment!
        print(f"[horizon] max_steps={self.max_steps} dist_nmi={self._dist_to_goal_nmi():.2f} u_cmd_mps={self.u_cmd_mps}")

        # return initial observation
        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Gymnasium returns: obs, reward, terminated, truncated, info

        self.step_idx += 1
        self.t += self.dt
        max_steps = self.max_steps if self.max_steps is not None else float('inf')
        truncated = self.step_idx >= max_steps

        # store previous ownship, previous obstacles (for CPA)
        x_prev, y_prev, psi_prev = float(self.X[0]), float(self.X[1]), float(self.X[2])
        self._x_prev = x_prev
        self._y_prev = y_prev 
        self.Xob_prev = self.Xob.copy()
        self.Yob_prev = self.Yob.copy()

        # -- waypoint planner (kept as CORALL)
        x_nmi = self.X[0] / 1852
        y_nmi = self.X[1] / 1852
        self.i_wpt = waypoint_selection(self.Xwpt, self.Ywpt, x_nmi, y_nmi, self.i_wpt)
        psi_wp = planning(self.Xwpt, self.Ywpt, x_nmi, y_nmi, self.i_wpt)

        # RL replaces the reactive_avoidance module
        a = float(np.clip(action[0], -1, 1))  # ensure action is in [-1, 1]

        # update colav heading contribution like a rate command (integrator)
        dpsi = a * float(self.colav_rate) * float(self.dt)
        self.psi_colav = float(np.clip(self.psi_colav + dpsi, -float(self.max_colav_rad), float(self.max_colav_rad)))

        # combine into heading command
        psi_wp = psi_wp if psi_wp is not None else 0.0
        psi_p = psi_wp + self.psi_colav

        # -- controller + actuator + vessel dynamics (kept as CORALL)
        psi, r, b = float(self.X[2]), float(self.X[3]), float(self.X[4])
        tau_c, v_c, self.ui_psi1 = controller(psi_p, psi, r, self.u_cmd_mps, b, self.ui_psi1, self.dt)
        tau_ac = actuator_modeling(tau_c, self.sat_amp_s)

        X_dot = vessel_dynamics(self.X.copy(), [tau_ac, v_c])
        self.X = integration(self.X.copy(), X_dot, self.dt)

        # obstacle propagation (CORALL)
        self.Xob, self.Yob, self.Vxob, self.Vyob = obstacle_sim(
            self.Xob, self.Yob, self.Vob, self.psiob, self.dt
        )

        # risk / CPA for reward + termination
        risk, dcpa, tcpa, dist, bearing = self._compute_risk(
            x=float(self.X[0]), y=float(self.X[1]), x_prev=x_prev, y_prev=y_prev, dt=self.dt
        )

        collision = bool(np.any(dist < self.collision_dist_m))
        reached_goal = self._dist_to_goal_nmi() < self.goal_tol_nmi
        terminated = collision or reached_goal

        # safety shaping 
        dmin = float(np.min(dist)) if dist.size else 1e9 # meters

        # starter reward function: delta-progress + risk + collision
        goal_dist = self._dist_to_goal_nmi()
        progress = (self.prev_goal_dist - goal_dist)
        self.prev_goal_dist = goal_dist
        
        # progress term
        r_progress = 50.0 * progress

        # risk penalty 
        # r_risk = -2 * float(np.max(risk)) if risk.size else 0.0

        # distance to obstacle penalty
        # r_distance = -5 * np.exp(-(dmin - 300.0)/50.0) if dist.size else 0.0

        # cross track error penalty (penalize lateral deviation from waypoint path)
        y_nmi = float(self.X[1]) / 1852.0
        r_cte = -0.5 * abs(y_nmi)

        # corridor penalty (stay around path, penalize if too far from path)
        # cte = abs(float(self.X[1] / 1852))
        # corridor = 1.0
        # r_corr = -2 * max(0.0, cte - corridor)**2

        # collision term
        r_collision = -100.0 if collision else 0.0

        # completion / goal term 
        r_goal = 150.0 if reached_goal else 0.0
        # time penalty
        r_time = -0.01

        # control term (penalize large steering commands)
        # r_control = -0.05 * abs(a)

        # bias magnitude penalty encourages returning to track after passing obstacle
        # r_bias = -0.05 * abs(self.psi_colav)


        reward = float(r_progress + r_cte + r_collision)

        obs = self._get_obs()

        info = {
            "psi_wp": float(psi_wp) if psi_wp is not None else 0.0, 
            "psi_colav": float(self.psi_colav),
            "psi_p": float(psi_p),
            "risk": risk.astype(float) if risk.size else np.zeros(0, dtype=float), 
            "risk_max": float(np.max(risk)) if risk.size else 0.0,
            "collision": collision, 
            "reached_goal": reached_goal,
        }

        if not np.all(np.isfinite(obs)):
            info["bad_obs"] = True
            # terminate with penalty so training does not explode
            reward = -200.0
            terminated = True
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


        # safety: never return non-finite reward
        if not np.isfinite(reward):
            reward = -200.0
            terminated = True
            info["bad_reward"] = True


        if not np.isfinite(obs).all():
            raise ValueError("Non-finite obs detected")
        if not np.isfinite(reward):
            raise ValueError("Non-finite reward detected")
        
        if terminated or truncated:
            info["episode_collision"] = int(collision)
            info["episode_goal_reached"] = int(reached_goal)
            info["episode_min_dist_m"] = float(dmin)
            info["episode_final_goal_dist_nmi"] = float(goal_dist)

        return obs, reward, terminated, truncated, info
    
    def _dist_to_goal_nmi(self) -> float:
        x_nmi = float(self.X[0]) / 1852
        y_nmi = float(self.X[1]) / 1852
        xg = float(self.Xwpt[-1])
        yg = float(self.Ywpt[-1])
        return float(np.sqrt((x_nmi - xg) ** 2 + (y_nmi - yg) ** 2))

    def _compute_risk(
            self, 
            x: float, 
            y: float,
            x_prev: float,
            y_prev: float,
            dt: float
    ):
        n = len(self.Xob)
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        dist = np.zeros(n, dtype=float)
        bearing = np.zeros(n, dtype=float)
        dcpa = np.zeros(n, dtype=float)
        tcpa = np.zeros(n, dtype=float)
        risk = np.zeros(n, dtype=float)

        psi = float(self.X[2])

        for j in range(n):
            dx = float(self.Xob[j] - x) 
            dy = float(self.Yob[j] - y)
            dist[j] = np.sqrt(dx**2 + dy**2)
            los = np.arctan2(dy, dx)
            bearing[j] = psi - los

            dcpa_j, tcpa_j, vrel_j, alpha_j, psi_vrel_j = cpa_calculations(
                x, y, x_prev, y_prev, 
                float(self.Xob[j]), float(self.Yob[j]),
                float(self.Xob_prev[j]), float(self.Yob_prev[j]), dt ) 
            
            dcpa[j] = float(dcpa_j)
            tcpa[j] = float(tcpa_j)
            risk[j] = float(risk_calculations(dist[j], dcpa[j], tcpa[j], vrel_j))

        return risk, dcpa, tcpa, dist, bearing
    
    def set_adaptive_horizon(self, max_cap=20000, min_cap = 500, horizon_factor=1.2) -> int:
        """
        Set self.max_steps based on initial distance to goal and commanded speed. 
        Uses nmi for distance (waypoints in plots stored in nmi), m/s for speed, dt for step length, and horizon_factor as a multiplier to ensure the horizon is long enough for the agent to reach the goal. 
        """

        dist_nmi = self._dist_to_goal_nmi()
        u_mps = float(self.u_cmd_mps)

        # convert m/s speed to nmi per step
        speed_nmi_per_step = max( 1e-6,(u_mps * self.dt) / 1852.0 )
        steps_needed = int(np.ceil((dist_nmi / speed_nmi_per_step) * horizon_factor))

        # clip to min and max steps
        self.max_steps = int(np.clip(steps_needed, min_cap, max_cap))

        return self.max_steps


    def _get_obs(self) -> np.ndarray:
        # ownship state: x, y, psi, r, b, u
        own = self.X.astype(np.float32)

        # compute obstacle features, select top-K most "dangerous"
        if len(self.Xob) == 0: 
            obs = np.concatenate([own, np.zeros(self.K * 5, dtype=np.float32)])
            return obs 
        
        # compute using current geometry + CPA/risk
        risk, dcpa, tcpa, dist, bearing = self._compute_risk(
            x=float(self.X[0]), y=float(self.X[1]), 
            x_prev=float(self._x_prev), y_prev=float(self._y_prev), dt=self.dt

        )

        # rank obstacles: highest risk first (fallback to closest distance)
        if risk.size: 
            idx = np.argsort(-risk)
        else:
            idx = np.argsort(dist)
        
        feats = []
        for k in range(self.K):
            if k < len(self.Xob):
                j = idx[k]
                feats.extend([
                    dist[j], bearing[j], dcpa[j], tcpa[j], risk[j]
                ])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.concatenate([own, np.array(feats, dtype=np.float32)])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs

if __name__ == "__main__":
    # test: random policy rollout
    env = CORALL_ReactiveAvoidanceGymEnv(case_number=1, dt = 0.2, K_obstacles=1)
    obs, info = env.reset(seed=0)
    ep_ret = 0.0

    for t in range(500):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        ep_ret += r
        if term or trunc:
            print(f"done at t={t}, return={ep_ret:.2f}, term={term}, trunc={trunc}, info={info}")
            break
    
    print("obs_dim:", obs.shape)