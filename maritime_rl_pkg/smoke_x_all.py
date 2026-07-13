"""Smoke test: verify target_speeds and angles vary per episode."""
from maritime_rl_pkg.env_procedural_encounter_sb3 import RandomEncounterEnv

env = RandomEncounterEnv(ownship_speed_mps=10.0, desired_cross_x_nmi=1.0, master_seed=42, verbose=True)
print("=== 8-episode procedural smoke test ===")
for i in range(8):
    obs, info = env.reset()
print("=== DONE ===")
