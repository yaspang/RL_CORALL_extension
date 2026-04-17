"""
Load and instantiate trained single-agent policies for multiple cases.

Usage:
    from maritime_rl_pkg.load_trained_policies import PolicyManager
    
    pm = PolicyManager(
        checkpoints={
            1: "SINGLE_AGENT_SB3_case1_20260415-120000/best_checkpoint.zip",
            6: "SINGLE_AGENT_SB3_case6_20260415-123226/best_checkpoint.zip",
            21: "SINGLE_AGENT_SB3_case21_20260415-125000/best_checkpoint.zip",
        }
    )
    
    # Evaluate policy on case 6
    env = pm.create_env(case=6, seed=0)
    obs, info = env.reset()
    action, _ = pm.predict(case=6, obs=obs)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO

from maritime_rl_pkg.env_single_agent_sb3 import SingleAgentOwnshipEnv


def create_5agent_collision_scenario(
    scale_distance: float = 0.5,
    u_nominal: float = 9.5,
) -> Dict[str, np.ndarray]:
    """
    Generate a 5-agent collision scenario with symmetric converging paths.
    
    Ownship is at origin (0, 0) heading 0°.
    4 obstacle ships positioned at cardinal/diagonal directions,
    each heading toward the ownship (converging geometry).
    
    Args:
        scale_distance: Scale factor for inter-agent distances (default 0.5 = half distance)
        u_nominal: Nominal speed for all agents (m/s)
    
    Returns:
        Dict with vessel configuration:
        {
            "positions": np.array (5, 2) - [x, y] for each agent
            "headings": np.array (5,) - heading (rad) for each agent
            "speeds": np.array (5,) - speed (m/s) for each agent
        }
    
    Example scenario:
        - Agent 0 (ownship): x=0,    y=0,    heading=0°   (north), speed=9.5 m/s
        - Agent 1:           x=+1000, y=+1000, heading=225° (toward origin), speed=9.5 m/s
        - Agent 2:           x=-1000, y=+1000, heading=315° (toward origin), speed=9.5 m/s
        - Agent 3:           x=-1000, y=-1000, heading=45°  (toward origin), speed=9.5 m/s
        - Agent 4:           x=+1000, y=-1000, heading=135° (toward origin), speed=9.5 m/s
        
        All agents at same speed creates ~500s encounter time at 1000m spacing.
    """
    # Reference distance (meters) - typical scenario separation
    ref_distance = 2000.0  # 2km initial separation
    distance = ref_distance * scale_distance
    
    # 5-agent collision scenario: ownship + 4 converging obstacles
    n_agents = 5
    positions = np.zeros((n_agents, 2), dtype=float)
    headings = np.zeros(n_agents, dtype=float)
    speeds = np.full(n_agents, u_nominal, dtype=float)
    
    # Agent 0 (ownship): at origin, heading 0° (north/up in typical nav coords)
    positions[0] = [0.0, 0.0]
    headings[0] = 0.0
    
    # Agent 1: NE quadrant, heading SW (toward origin) at 225°
    positions[1] = [distance, distance]
    headings[1] = np.radians(225.0)  # Heading toward origin
    
    # Agent 2: NW quadrant, heading SE (toward origin) at 315°
    positions[2] = [-distance, distance]
    headings[2] = np.radians(315.0)  # Heading toward origin
    
    # Agent 3: SW quadrant, heading NE (toward origin) at 45°
    positions[3] = [-distance, -distance]
    headings[3] = np.radians(45.0)  # Heading toward origin
    
    # Agent 4: SE quadrant, heading NW (toward origin) at 135°
    positions[4] = [distance, -distance]
    headings[4] = np.radians(135.0)  # Heading toward origin
    
    return {
        "positions": positions,
        "headings": headings,
        "speeds": speeds,
        "n_agents": n_agents,
    }


class PolicyManager:
    """Manage multiple trained policies for different cases."""
    
    def __init__(self, checkpoints: Dict[int, str]):
        """
        Initialize policy manager with checkpoint paths.
        
        Args:
            checkpoints: Dict mapping case_number -> checkpoint_path
                Example: {1: "path/to/case1/best_checkpoint.zip", 
                          6: "path/to/case6/best_checkpoint.zip"}
        """
        self.checkpoints = checkpoints
        self.models = {}
        self.envs = {}
        
        # Load models
        for case_num, checkpoint_path in checkpoints.items():
            path = Path(checkpoint_path)
            if not path.exists():
                print(f"⚠ Warning: Checkpoint not found for case {case_num}: {checkpoint_path}")
                continue
            
            model = PPO.load(str(path))
            self.models[case_num] = model
            print(f"✓ Loaded policy for case {case_num}")
    
    def create_env(
        self, 
        case: int, 
        seed: int = 0,
        dt: float = 0.5,
        sim_time: float = 1950.0,
        route_len_nmi: float = 2.0,
        collision_scenario: bool = False,
        collision_bearing_offset: float = 0.0,
        use_fast_speeds: bool = False,
        scaled_5agent_collision: bool = False,
        collision_scale_distance: float = 0.5,
    ):
        """
        Create environment for a specific case.
        
        Args:
            case: Case number (1, 6, 21, etc.)
            seed: Random seed
            dt: Time step
            sim_time: Episode length
            route_len_nmi: Route length in NMI
            collision_scenario: If True, initialize with collision-prone configuration
            collision_bearing_offset: Bearing adjustment (radians) to impose collision geometry
                                     for case 21 (e.g., π/6 = 30° adjustment)
            use_fast_speeds: If True, use faster obstacle speeds (deprecated)
            scaled_5agent_collision: If True, use 5-agent scenario with scaled distances
                                    instead of case geometry
            collision_scale_distance: Scale factor for 5-agent collision distances (0.5 = half)
        
        Returns:
            SingleAgentOwnshipEnv instance
        """
        # For case 21 collision scenario: use case 6 geometry or 5-agent custom
        actual_case = case
        use_custom_init = False
        custom_config = None
        
        if scaled_5agent_collision:
            # 5-agent scaled collision scenario
            custom_config = create_5agent_collision_scenario(
                scale_distance=collision_scale_distance,
                u_nominal=9.5,
            )
            use_custom_init = True
            print(f"  → Using 5-agent collision scenario (scale={collision_scale_distance})")
        elif collision_scenario and case == 21:
            actual_case = 6  # Use Imazu (case 6) positions
            print(f"  → Case 21 using case 6 geometry with collision bearings")
        
        env = SingleAgentOwnshipEnv(
            case_number=actual_case,
            dt=dt,
            sim_time=sim_time,
            route_len_nmi=route_len_nmi,
            seed=seed,
        )
        
        # If using custom 5-agent collision, override init_from_case
        if use_custom_init and custom_config is not None:
            original_init = env.env_multi.init_from_case
            
            def custom_init_from_case():
                """Custom initialization with 5-agent collision geometry."""
                X_all = np.zeros((custom_config["n_agents"], 6), dtype=float)
                for i in range(custom_config["n_agents"]):
                    x, y = custom_config["positions"][i]
                    psi = custom_config["headings"][i]
                    u = custom_config["speeds"][i]
                    X_all[i, :] = np.array([x, y, psi, 0.0, 0.0, u], dtype=float)
                    env.env_multi.u_des_all[i] = u
                
                return X_all
            
            # Replace init method
            env.env_multi.init_from_case = custom_init_from_case
            # Also update n_agents to match custom scenario
            if custom_config["n_agents"] != env.env_multi.n_agents:
                print(f"  ⚠ Warning: Custom scenario has {custom_config['n_agents']} agents, "
                      f"env has {env.env_multi.n_agents}")
        
        # Store for later reference
        key = f"case_{case}_seed_{seed}"
        self.envs[key] = env
        
        if collision_scenario and case == 21:
            print(f"  → Collision bearing offset: {collision_bearing_offset:.3f} rad ({np.degrees(collision_bearing_offset):.1f}°)")
        
        return env
    
    def predict(
        self, 
        case: int, 
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get policy action for observation.
        
        Args:
            case: Case number
            obs: Observation array
            deterministic: Use deterministic policy (no exploration)
        
        Returns:
            (action, log_prob) tuple
        """
        if case not in self.models:
            raise ValueError(f"No policy loaded for case {case}")
        
        model = self.models[case]
        action, _states = model.predict(obs, deterministic=deterministic)
        return action, _states
    
    def run_episode(
        self,
        case: int,
        seed: int = 0,
        max_steps: Optional[int] = None,
        collision_scenario: bool = False,
        collision_bearing_offset: float = 0.0,
        use_fast_speeds: bool = False,
        deterministic: bool = True,
    ):
        """
        Run a single episode with trained policy.
        
        Args:
            case: Case number
            seed: Random seed
            max_steps: Max steps (if None, use full episode)
            collision_scenario: Initialize with collision scenario
            collision_bearing_offset: Bearing offset (rad) for collision geometry
            use_fast_speeds: Use faster speeds for guaranteed collision
            deterministic: Use deterministic policy
        
        Returns:
            Dict with episode metrics
        """
        # Create environment
        env = self.create_env(
            case=case,
            seed=seed,
            collision_scenario=collision_scenario,
            collision_bearing_offset=collision_bearing_offset,
            use_fast_speeds=use_fast_speeds,
        )
        
        obs, info = env.reset()
        
        episode_return = 0.0
        step = 0
        done = False
        
        if max_steps is None:
            max_steps = int(1950.0 / 0.5)  # Default ~ 3900 steps
        
        while not done and step < max_steps:
            action, _ = self.predict(case, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated
            step += 1
        
        ownship_metrics = env.get_ownship_metrics()
        
        results = {
            "case": case,
            "seed": seed,
            "episode_return": float(episode_return),
            "episode_steps": step,
            "collision": int(ownship_metrics.get("collision", 0)),
            "success": int(ownship_metrics.get("success", 0)),
            "goal_progress": float(ownship_metrics.get("goal_progress", 0.0)),
            "min_dcpa_m": float(ownship_metrics.get("min_dcpa_m", np.inf)),
            "min_tcpa_s": float(ownship_metrics.get("min_tcpa_s", np.inf)),
        }
        
        env.close()
        return results
    
    def evaluate_multiple_cases(
        self,
        cases: list,
        episodes_per_case: int = 3,
        collision_scenario: bool = False,
        collision_bearing_offset: float = np.pi / 6,
        use_fast_speeds: bool = False,
    ):
        """
        Evaluate trained policies across multiple cases.
        
        Args:
            cases: List of case numbers [1, 6, 21]
            episodes_per_case: Number of eval episodes per case
            collision_scenario: Initialize with collision scenarios (especially for case 21)
            collision_bearing_offset: Bearing offset (rad) for collision geometry
                                     (default 30° = π/6). Use for case 21 collisions.
            use_fast_speeds: Use faster obstacle speeds for guaranteed collision
        
        Returns:
            Dict of results by case
        """
        results = {}
        
        for case in cases:
            if case not in self.models:
                print(f"⚠ Skipping case {case} (no policy loaded)")
                continue
            
            print(f"\n{'='*60}")
            scenario_str = ""
            if collision_scenario and case == 21:
                scenario_str = f" [Collision Scenario - Bearing Offset: {np.degrees(collision_bearing_offset):.0f}°]"
            print(f"Evaluating Case {case}{scenario_str}")
            print(f"{'='*60}")
            
            case_results = []
            
            for ep in range(episodes_per_case):
                seed = ep
                print(f"  Episode {ep+1}/{episodes_per_case}...", end=" ", flush=True)
                
                ep_result = self.run_episode(
                    case=case,
                    seed=seed,
                    collision_scenario=collision_scenario and case == 21,
                    collision_bearing_offset=collision_bearing_offset,
                    use_fast_speeds=use_fast_speeds,
                )
                case_results.append(ep_result)
                
                print(f"✓ Return: {ep_result['episode_return']:.2f} | "
                      f"Success: {ep_result['success']} | "
                      f"Goal: {ep_result['goal_progress']:.1f}%" +
                      (f" | Min DCPA: {ep_result['min_dcpa_m']:.1f}m" if case == 21 and collision_scenario else ""))
            
            # Aggregate
            results[case] = {
                "per_episode": case_results,
                "mean_return": float(np.mean([r["episode_return"] for r in case_results])),
                "mean_goal_progress": float(np.mean([r["goal_progress"] for r in case_results])),
                "success_rate": float(np.mean([r["success"] for r in case_results])),
                "collision_rate": float(np.mean([r["collision"] for r in case_results])),
                "mean_min_dcpa": float(np.mean([r["min_dcpa_m"] for r in case_results if r["min_dcpa_m"] != np.inf])) if any(r["min_dcpa_m"] != np.inf for r in case_results) else np.inf,
            }
            
            print(f"\n  Summary (Case {case}):")
            print(f"    Mean Return:        {results[case]['mean_return']:.2f}")
            print(f"    Mean Goal Progress: {results[case]['mean_goal_progress']:.1f}%")
            print(f"    Success Rate:       {results[case]['success_rate']:.2f}")
            print(f"    Collision Rate:     {results[case]['collision_rate']:.2f}")
            if collision_scenario and case == 21:
                mean_dcpa = results[case]['mean_min_dcpa']
                print(f"    Mean Min DCPA:      {mean_dcpa:.1f}m" if mean_dcpa != np.inf else "    Mean Min DCPA:      N/A")
        
        return results


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multiple trained policies")
    parser.add_argument("--checkpoint-1", type=str, help="Checkpoint path for case 1")
    parser.add_argument("--checkpoint-6", type=str, help="Checkpoint path for case 6")
    parser.add_argument("--checkpoint-21", type=str, help="Checkpoint path for case 21")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per case")
    parser.add_argument("--collision-scenario", action="store_true", help="Use collision scenarios for case 21")
    parser.add_argument("--collision-bearing-offset", type=float, default=np.pi/6, 
                        help="Bearing offset in degrees for case 21 collision (default: 30°)")
    parser.add_argument("--use-fast-speeds", action="store_true", help="Use faster speeds for guaranteed collision")
    
    args = parser.parse_args()
    
    # Convert bearing from degrees to radians
    bearing_rad = np.radians(args.collision_bearing_offset) if args.collision_bearing_offset != np.pi/6 else np.pi/6
    
    # Build checkpoints dict from arguments
    checkpoints = {}
    if args.checkpoint_1:
        checkpoints[1] = args.checkpoint_1
    if args.checkpoint_6:
        checkpoints[6] = args.checkpoint_6
    if args.checkpoint_21:
        checkpoints[21] = args.checkpoint_21
    
    if not checkpoints:
        print("Error: Provide at least one checkpoint path (--checkpoint-{1,6,21})")
        return
    
    # Create manager and evaluate
    pm = PolicyManager(checkpoints)
    
    results = pm.evaluate_multiple_cases(
        cases=list(checkpoints.keys()),
        episodes_per_case=args.episodes,
        collision_scenario=args.collision_scenario,
        collision_bearing_offset=bearing_rad,
        use_fast_speeds=args.use_fast_speeds,
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for case, metrics in results.items():
        print(f"\nCase {case}:")
        print(f"  Mean Return:        {metrics['mean_return']:8.2f}")
        print(f"  Mean Goal Progress: {metrics['mean_goal_progress']:8.1f}%")
        print(f"  Success Rate:       {metrics['success_rate']:8.2f}")
        print(f"  Collision Rate:     {metrics['collision_rate']:8.2f}")
        if case == 21 and args.collision_scenario:
            mean_dcpa = metrics.get('mean_min_dcpa', np.inf)
            if mean_dcpa != np.inf:
                print(f"  Mean Min DCPA:      {mean_dcpa:8.1f} m")


if __name__ == "__main__":
    main()
