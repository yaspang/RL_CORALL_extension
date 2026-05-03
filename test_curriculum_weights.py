#!/usr/bin/env python3
"""
Test curriculum weighting to verify anti-forgetting mechanism.
"""
import sys
sys.path.insert(0, ".")

from maritime_rl_pkg.env_random_case_sb3 import RandomCaseEnv

env = RandomCaseEnv(
    cases_to_train=[1, 6, 21],
    enable_curriculum=True,
)

print("[CURRICULUM WEIGHT TEST]")
print("=" * 60)

test_steps = [
    (0, "Phase 1 start (2-ship)"),
    (100000, "Phase 1 mid (2-ship)"),
    (400000, "Phase 1 end (2-ship)"),
    (450000, "Transition 1 start (blend 2-ship/3-ship)"),
    (500000, "Phase 2 start (blend 2-ship/3-ship)"),
    (600000, "Phase 2 mid (mostly 3-ship)"),
    (950000, "Transition 2 start (blend all)"),
    (1000000, "Phase 3 start (all)"),
    (1200000, "Phase 3 mid (all)"),
]

for step, description in test_steps:
    env.current_step = step
    
    available = env._get_curriculum_available_cases()
    weights = env._get_case_sampling_weights()
    
    print(f"\nStep {step:7d} ({step/1e6:.1f}M): {description}")
    print(f"  Available cases: {available}")
    print(f"  Sampling weights: ", end="")
    for case in [1, 6, 21]:
        w = weights.get(case, 0.0)
        print(f"Case {case}: {w:.1%}  ", end="")
    print()

print("\n" + "=" * 60)
print("[TEST COMPLETE]")
