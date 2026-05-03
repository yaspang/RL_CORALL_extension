#!/usr/bin/env python3
"""
Quick diagnostic to verify observation shapes, values, and normalization.
"""
import numpy as np
import sys
sys.path.insert(0, ".")

from maritime_rl_pkg.env_random_case_sb3 import RandomCaseEnv
from stable_baselines3.common.vec_env import DummyVecEnv

print("[OBSERVATION DIAGNOSTIC TEST]")
print("=" * 60)

# Create environment with curriculum enabled
env = RandomCaseEnv(
    cases_to_train=[1, 6, 21],  # All three curriculum cases
    num_seeds=100,
    enable_curriculum=False,  # Disable curriculum for this test
)

# Create VecEnv wrapper (like training does)
vec_env = DummyVecEnv([lambda: env])

print("\n[1] Testing reset()...")
obs, info = env.reset()
print(f"    Observation shape: {obs.shape}")
print(f"    Observation dtype: {obs.dtype}")
print(f"    Observation min: {np.min(obs):.4f}, max: {np.max(obs):.4f}")
print(f"    Observation mean: {np.mean(obs):.4f}, std: {np.std(obs):.4f}")
print(f"    Case: {info.get('case', '?')}, Seed: {info.get('seed', '?')}")

# Check for NaNs or Infs
if np.any(np.isnan(obs)):
    print("    WARNING: NaN values detected in observation!")
if np.any(np.isinf(obs)):
    print("    WARNING: Inf values detected in observation!")

print("\n[2] Testing multiple episodes (20 steps)...")
obs_samples = []
for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    obs_samples.append(obs)
    
    if step % 5 == 0:
        print(f"    Step {step:2d}: shape={obs.shape}, range=[{np.min(obs):7.3f}, {np.max(obs):7.3f}], reward={reward:7.2f}")
        if info.get('case'):
            print(f"              case={info.get('case')}, has_collision={info.get('collision', False)}")
    
    if terminated or truncated:
        obs, info = env.reset()
        print(f"    Episode reset - Case: {info.get('case')}, Seed: {info.get('seed')}")

obs_array = np.array(obs_samples)
print(f"\n[3] Observation statistics across {len(obs_samples)} steps:")
print(f"    Overall min: {np.min(obs_array):.4f}")
print(f"    Overall max: {np.max(obs_array):.4f}")
print(f"    Overall mean: {np.mean(obs_array):.4f}")
print(f"    Overall std: {np.std(obs_array):.4f}")

# Check each dimension
print(f"\n[4] Per-dimension ranges (should be ~[-1, 1] for normalized):")
for i in range(obs.shape[0]):
    dim_values = obs_array[:, i]
    print(f"    Dim {i:2d}: [{np.min(dim_values):7.3f}, {np.max(dim_values):7.3f}]", end="")
    if np.min(dim_values) < -1.5 or np.max(dim_values) > 1.5:
        print(" <- OUT OF RANGE!")
    else:
        print()

print("\n[5] Testing VecEnv wrapper...")
vec_obs = vec_env.reset()
print(f"    VecEnv observation shape: {vec_obs.shape}")
print(f"    VecEnv observation range: [{np.min(vec_obs):.4f}, {np.max(vec_obs):.4f}]")

for i in range(5):
    actions = vec_env.action_space.sample()
    vec_obs, rewards, dones, infos = vec_env.step([actions])
    if i == 0:
        print(f"    Step {i}: obs_shape={vec_obs.shape}, reward={rewards[0]:.2f}, done={dones[0]}")

print("\n[6] Curriculum status at step 0:")
if hasattr(env, 'current_step'):
    print(f"    Current step: {env.current_step}")
if hasattr(env, '_get_curriculum_available_cases'):
    available = env._get_curriculum_available_cases()
    print(f"    Available cases at step 0: {available}")

print("\n" + "=" * 60)
print("[DIAGNOSTIC COMPLETE]")
