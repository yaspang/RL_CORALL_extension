"""
HARD-CASE OVERSAMPLING WITH FAILURE REPLAY
===========================================

This guide explains how to implement sensitive training with hard-case oversampling
to prevent the policy from "solving" some cases while forgetting others.

PROBLEM ADDRESSED
=================
When training on all 22 cases uniformly, risk-75 improved some cases (e.g., 13, 18, 20, 21)
but FORGOT others (e.g., 1-4, 15-17). This is due to catastrophic forgetting—the policy
prioritizes recently-seen cases and loses skill on others.

Hard-case oversampling prevents this by:
1. Identifying problematic cases (cases 13, 18, 20, 21 are known trouble)
2. Tracking evaluation failures dynamically
3. Oversampling failures during training (70% hard seeds, 30% random)
4. Maintaining a stratified distribution:
   - 50% curriculum cases (normal phase progression)
   - 30% hard/failure cases (Imazu + recent failures)
   - 20% easy cases (anti-forgetting on baselines)

WORKFLOW
========

STEP 1: TRAIN WITH HARD-CASE OVERSAMPLING ENABLED
--------------------------------------------------

python -m maritime_rl_pkg.train_generalized_policy_sb3 \\
    --num_steps 3000000 \\
    --train_batch 256 \\
    --rollout_frag 256 \\
    --lr 1e-4 \\
    --master_seed 42 \\
    --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \\
    --hard_case_oversampling \\
    --hard_case_pool_file "hard_case_pool_initial.json"

This will:
  - Train with 50% curriculum, 30% hard (13,18,20,21), 20% easy (1,5,12)
  - Create/load hard_case_pool_initial.json to track failures
  - Sample 70% of hard cases from the tracked failure pool (when available)

Output:
  - GENERALIZED_SB3_TIMESTAMP/ (training directory)
  - hard_case_pool_initial.json (updated during training)

---

STEP 2: EVALUATE ALL CASES
---------------------------

For each case (1-22), run evaluation:

for case in {1..22}; do
    python -m maritime_rl_pkg.eval_generalized_policy_sb3 \\
        --checkpoint "GENERALIZED_SB3_TIMESTAMP/best_checkpoint.zip" \\
        --case $case \\
        --episodes 100 \\
        --seed 0 \\
        --desired_cross_x_nmi 1.05 \\
        --target_speed_mps 10.0 \\
        --ownship_speed_mps 10.0
done

Output:
  - policy_eval_generalized_sb3_case{X}_{TIMESTAMP}/seed_0/policy_eval_summary.json (per case)

---

STEP 3: EXTRACT FAILURES AND UPDATE POOL
-----------------------------------------

After evaluation, extract failures from each case:

python -m maritime_rl_pkg.extract_failures_for_hard_case_pool \\
    --eval_dir "policy_eval_generalized_sb3_case22_TIMESTAMP/seed_0/" \\
    --output_pool "hard_case_pool_updated.json" \\
    --collision_threshold 0.0 \\
    --dcpa_threshold 100.0

Repeat for each case, updating the same pool file each time:

for case in {1..22}; do
    python -m maritime_rl_pkg.extract_failures_for_hard_case_pool \\
        --eval_dir "policy_eval_generalized_sb3_case${case}_TIMESTAMP/seed_0/" \\
        --output_pool "hard_case_pool_updated.json" \\
        --collision_threshold 0.0 \\
        --dcpa_threshold 100.0
done

This will:
  - Read policy_eval_summary.json from each eval directory
  - Extract episodes with:
    * collision_flag > 0 (collision occurred), OR
    * DCPA < 100.0 m (near-collision/high-risk)
  - Add (case, seed) pairs to hard_case_pool_updated.json
  - Display statistics for each case

Output:
  - hard_case_pool_updated.json with failures from all 22 cases

Example output:
    Case  1:   8 failed seeds
    Case 13:  25 failed seeds
    Case 18:  19 failed seeds
    Case 20:  31 failed seeds
    Case 21:  28 failed seeds
    ...

---

STEP 4: TRAIN AGAIN WITH UPDATED FAILURE POOL
----------------------------------------------

Start a new training run using the failures discovered in evaluation:

python -m maritime_rl_pkg.train_generalized_policy_sb3 \\
    --num_steps 3000000 \\
    --train_batch 256 \\
    --rollout_frag 256 \\
    --lr 1e-4 \\
    --master_seed 42 \\
    --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \\
    --hard_case_oversampling \\
    --hard_case_pool_file "hard_case_pool_updated.json"

This will:
  - Initialize hard-case pool from hard_case_pool_updated.json
  - Use stratified sampling (50% curriculum, 30% hard, 20% easy)
  - Oversample the failure cases identified in evaluation
  - Update the pool during training as new failures are discovered

Repeat steps 2-4 iteratively until performance converges across all cases.

---

KEY PARAMETERS
==============

--hard_case_oversampling
  Enable hard-case oversampling. When set, uses stratified sampling instead of
  uniform curriculum-based sampling.
  Default: disabled (False)

--hard_case_pool_file
  Path to JSON file tracking (case, seed) pairs. When provided:
  - If file exists: loads failures from previous eval sweep
  - If file missing: initializes with known hard cases (13, 18, 20, 21)
  - During training: updates with newly discovered failures
  Default: None (uses hardcoded HARD_CASES list)

Extract failures parameters:

--collision_threshold
  Episodes with collision_flag >= this are marked as failures.
  collision_flag=1 means collision occurred.
  Default: 0.0 (count all collisions)

--dcpa_threshold
  Episodes with DCPA < this (meters) are marked as failures.
  DCPA = Distance to Closest Point of Approach.
  Smaller DCPA = closer/riskier encounter.
  Default: 100.0 m (mark high-risk as failures)

---

SAMPLING DISTRIBUTION
======================

When hard_case_oversampling=True, training uses:

  50% curriculum_cases
    └─ Normal curriculum progression (phase-based)
       Phase 1 (0-500k): 2-ship only
       Phase 2 (500k-1M): 2-ship + 3-ship
       Phase 3 (1M+): 2-ship + 3-ship + 4-ship

  30% hard_cases
    └─ Cases 13, 18, 20, 21 (Imazu scenarios)
    └─ Plus any cases in hard_case_pool_file
    └─ For each hard case: 70% sample from failed seeds, 30% random seeds

  20% easy_cases
    └─ Cases 1, 5, 12 (representative baselines)
    └─ Prevents catastrophic forgetting on "solved" cases

---

POOL FILE FORMAT
================

hard_case_pool.json is a JSON dictionary mapping case number to list of failed seeds:

{
  "1": [3, 7, 12, 45, ...],     # Case 1: seeds that failed
  "13": [0, 1, 2, ..., 99],     # Case 13: many failures (known problem)
  "18": [5, 10, 15, 20, ...],   # Case 18: some failures
  "20": [2, 7, 11, 33, ...],    # Case 20: some failures
  "21": [1, 8, 22, 55, ...]     # Case 21: some failures
}

When training:
  - If (case, seed) is in pool, 70% chance to sample it directly
  - Otherwise, sample seed uniformly [0, 100)

---

EXPECTED IMPROVEMENTS
=====================

Iteration 0 (baseline uniform training):
  - Case 1:  ✓✓✓ excellent
  - Case 13: ✗✗✗ many collisions
  - Case 18: ✗✗✗ many collisions
  - Case 20: ✗✗ good but not great
  - Case 21: ✗✗✗ many collisions
  → Forgetting observed: fixing hard cases breaks easy ones

Iteration 1 (hard-case oversampling):
  - Focus on cases 13, 18, 20, 21 more
  - Maintain anti-forgetting on cases 1, 5, 12
  - Result: improvement on hard cases, less forgetting on easy

Iteration 2+ (iterative refinement):
  - Extract new failures from eval sweep
  - Train with updated failure pool
  - Gradually improve across all cases

---

DEBUGGING / MONITORING
======================

During training, you'll see:

  [HardCaseOversampling] Loaded pool from hard_case_pool_updated.json: 6 cases
  
This confirms the pool was loaded. The env will track how many times it samples
from each pool during training.

To inspect the pool file:
  cat hard_case_pool_updated.json | python -m json.tool
  
To see summary statistics:
  grep -A 100 "per_episode_results" policy_eval_summary.json | head -20

---

ADVANCED: MANUAL POOL UPDATES
==============================

You can manually edit hard_case_pool.json to prioritize certain cases:

{
  "1": [0, 1, 2],           # Add case 1 to hard pool manually
  "13": [0, 1, ..., 99],    # Keep all seeds of case 13
  "18": [5, 10, 15]         # Only sample specific seeds of case 18
}

Then train with:
  --hard_case_pool_file "hard_case_pool.json"

---

HYPERPARAMETER TUNING
======================

If performance is still not balanced:

Increase hard-case fraction:
  Edit _sample_case_with_hard_oversampling() in env_random_case_sb3.py:
    - Change 0.5 (50% curriculum) to 0.4 (40% curriculum)
    - Change 0.8 (80% cumulative) to 0.75 (75% cumulative)
    - Result: 40% curriculum, 40% hard, 20% easy

Decrease anti-forgetting on easy cases:
  Edit EASY_CASES = [1, 5, 12] to subset like [1, 5]

Adjust failure seed sampling probability:
  Edit "if local_rng.random() < 0.7:" to 0.5 or 0.8
  (lower = more random seeds, higher = focus on known failures)

---

CITATIONS & REFERENCES
======================

This approach is inspired by:
- Curriculum learning (Bengio et al., 2009)
- Hard example mining / importance sampling
- Catastrophic forgetting mitigation (continual learning literature)
- Risk-sensitive RL with prioritized experience replay

For your maritime RL scenario:
- Risk-aware training prevents critical failures in collision avoidance
- Multi-case generalization requires explicit failure-focused training
"""

if __name__ == "__main__":
    print(__doc__)
