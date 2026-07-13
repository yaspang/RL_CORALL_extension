# RL_CORALL_extension

Deep Reinforcement Learning for Maritime Collision Avoidance using CORALL Simulator

## Overview

This project extends the CORALL maritime collision avoidance simulator with Reinforcement Learning (RL) training and evaluation capabilities. A single generalized PPO policy is trained using curriculum learning to handle collision avoidance across a range of multi-ship encounter scenarios. The project supports two training modes: a **case-based mode** using the 22 fixed Imazu encounter geometries, and a **procedural encounter mode** that generates fully randomized encounters at runtime with no fixed geometry.

**Key Features:**
- **Two Training Modes**: Fixed Imazu case rotation (`RandomCaseEnv`) or fully procedural randomized encounters (`RandomEncounterEnv`)
- **Curriculum Learning**: Policy trained progressively — 1 target → 1–2 targets → 1–3 targets
- **Generalized Policy**: Single checkpoint evaluated across all 22 CORALL cases
- **Backwards-from-Crossing Geometry**: Procedural mode places targets by sampling speed and angle independently, then computing start positions so all ships arrive at the crossing point simultaneously
- **Constraint-Based Checkpoint Selection**: Checkpoints ranked by safety constraints (collision-free + full success required) before secondary metrics
- **Normalized Observations**: 29-dimensional state space with stable normalization
- **Comprehensive Metrics**: Collision rate, success rate, risk exposure, minimum separation, completion time

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│           Single-Agent RL Training                              │
│         (Multi-Agent Infrastructure)                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Agent 0 (Ownship) - RL Controlled                             │
│  └─ PPO Policy from Stable-Baselines3                          │
│     Action: Heading command (7 discrete bins)                  │
│     Observation: 29-dim state vector                           │
│     Reward: 5-component shaping function                       │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────────────────┐
        │                                         │
   ┌────▼───────────────────────────┐    ┌────────▼───────────────────┐
   │ Mode A: Case-Based             │    │ Mode B: Procedural          │
   │ RandomCaseEnv                  │    │ RandomEncounterEnv          │
   │                                │    │                             │
   │ Rotates across 22 fixed Imazu  │    │ Generates fully randomized  │
   │ encounter geometries each      │    │ encounters each episode:    │
   │ episode. Curriculum selects    │    │ - Target speed: 6–14 m/s   │
   │ from 2-ship → 3-ship → 4-ship  │    │ - Angle: 15°–345°          │
   │ cases as training progresses.  │    │ - Crossing dist: 1.0 ±20%  │
   │                                │    │ - N obstacles: 1 → 1–2 →   │
   │ Uses fixed Imazu geometry.     │    │   1–3 (curriculum by step)  │
   └────────────────────────────────┘    └─────────────────────────────┘
        │                                         │
        └──────────────────┬──────────────────────┘
                           │
              SingleAgentOwnshipEnv
              (env_single_agent_sb3.py)
                           │
              env_multi_agent_ppo.py
              (PettingZoo, ownship only)
                           │
              PPO (Stable-Baselines3)
```

---

## Environment Stack

### Core Environments

**1. Multi-Agent Base Environment** (`env_multi_agent_ppo.py`)
- PettingZoo `ParallelEnv` supporting all agents
- Agent 0 = ownship (RL-controlled); Agents 1–K = obstacles (scripted CORALL trajectories)
- Handles vehicle dynamics, CPA computation, collision detection, per-agent reward shaping
- Reward components: progress toward goal, risk penalty, separation bonus, collision penalty, success bonus
- AABB broad-phase filtering available for 5+ agent scalability

**2. Single-Agent Wrapper** (`env_single_agent_sb3.py`)
- Wraps `env_multi_agent_ppo.py` into a single-agent Gymnasium interface (ownship only)
- Used as the base environment for both training modes
- Observation size varies by number of obstacles (14–26 dims before padding)

**3. Case-Based Curriculum Wrapper** (`env_random_case_sb3.py`)
- **Training mode**: `--cases` (Imazu cases 1–22)
- Each episode samples a random case from the current curriculum phase:
  - Phase 1 (0–500k steps): 2-ship cases only
  - Phase 2 (500k–1M steps): 2-ship and 3-ship blend
  - Phase 3 (1M–2.5M steps): All 22 cases (2/3/4-ship blend)
- Observation padded to 29-dim for consistent policy input

**4. Procedural Encounter Wrapper** (`env_procedural_encounter_sb3.py`)
- **Training mode**: `--use_procedural_encounters`
- Generates novel encounter geometry every episode using **backwards-from-crossing placement**:
  - Ownship travels east at fixed speed; crossing point is at `desired_cross_x_nmi` ahead
  - `t_cross = cross_x_m / ownship_speed_mps` — time for ownship to reach crossing
  - Each obstacle: sample speed (6–14 m/s) and angle θ (15°–345°), compute start as `t_cross` back along θ from crossing point
  - Guarantees all ships arrive at crossing simultaneously, producing genuine encounter geometry
- No fixed Imazu geometry used — every episode is unique
- Curriculum by training step:
  - Steps 0–999k: 1 target
  - Steps 1M–1.999M: 1–2 targets
  - Steps 2M+: 1–3 targets
- Verbose logging prints per-episode geometry: `episode=N n_obstacles=M ownship_speed=10.0 target_speeds=[V] crossing_x=C angles=[θ]`

**5. Baseline Environment** (`env_baseline.py`)
- CORALL's rule-based reactive avoidance guidance
- Ownship controlled by CORALL waypoint planner + reactive avoidance
- Obstacles follow fixed CORALL-scripted trajectories
- Used for fair side-by-side comparison with RL policy

---

## Training Pipeline

### Script: `train_generalized_policy_sb3.py`

**Algorithm**: Stable-Baselines3 PPO  
**Network**: 2-layer MLP [256, 256], Tanh activation  
**Hyperparameters**:
- Learning rate: `1e-4`
- N-steps (rollout fragment): `256`
- Batch size: `256` (match rollout to avoid SB3 warning)
- Gamma: `0.99` | GAE Lambda: `0.95` | Clip range: `0.2`
- Episode horizon: `--sim_time 900.0` seconds (default)

**Reward Function**:
```python
{
    'progress':   200.0,   # Delta progress toward goal each step
    'risk':       -15.0,   # Risk exposure penalty (CPA-based)
    'separation':   2.0,   # Bonus for maintaining safe separation
    'collision': -600.0,   # Terminal penalty on collision
    'success':   250.0,    # Terminal bonus on reaching goal
}
```

**Mode A — Case-Based Training** (Imazu cases):
```bash
python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --num_steps 3000000 \
    --train_batch 256 \
    --rollout_frag 256 \
    --lr 1e-4 \
    --master_seed 42 \
    --ownship_speed_mps 10.0 \
    --desired_cross_x_nmi 1.0 \
    --sim_time 900.0 \
    --checkpoint_freq 50000
```

**Mode B — Procedural Encounter Training** (recommended for generalization):
```bash
python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --num_steps 3000000 \
    --train_batch 256 \
    --rollout_frag 256 \
    --lr 1e-4 \
    --master_seed 42 \
    --use_procedural_encounters \
    --ownship_speed_mps 10.0 \
    --desired_cross_x_nmi 1.0 \
    --sim_time 900.0 \
    --checkpoint_freq 50000
```

**Output**: `GENERALIZED_SB3_YYYYMMDD-HHMMSS/`
- `checkpoints/generalized_checkpoint_*_steps.zip` — periodic checkpoints
- `training_config.json` — full hyperparameter record (`sim_time`, `dt`, `route_len_nmi`, etc.)

---

## Checkpoint Ranking

### Script: `rank_checkpoints_weighted.py`

After training, all saved checkpoints are evaluated on a representative set of cases and ranked using a **constraint-based** selection scheme:

1. **Acceptable checkpoints**: `collision_rate == 0.0` AND `success_rate == 1.0`
   - Sorted by: `risk_exposure` ↑ → `min_sep_m` ↓ → `completion_time_s` ↑
2. **Unacceptable checkpoints**: listed below, sorted by `collision_rate` ↑

This ensures safety constraints are hard requirements, not tradeoff weights.

```bash
python -m maritime_rl_pkg.rank_checkpoints_weighted \
    --training_dir "GENERALIZED_SB3_YYYYMMDD-HHMMSS" \
    --quick_eval \
    --desired_cross_x_nmi 1.00 \
    --ownship_speed_mps 10.0 \
    --target_speed_mps 10.0 \
    --sim_time 900.0
```

**Output columns**: `checkpoint`, `collision_rate`, `success_rate`, `risk_exposure`, `min_sep_m`, `completion_time_s`, `acceptable`

---

## Policy Evaluation

### Script: `eval_generalized_policy_sb3.py`

Evaluates a single checkpoint on a specific Imazu case.

```bash
python -m maritime_rl_pkg.eval_generalized_policy_sb3 \
    --checkpoint GENERALIZED_SB3_YYYYMMDD-HHMMSS/checkpoints/generalized_checkpoint_300000_steps.zip \
    --case 6 \
    --episodes 100 \
    --seed 0 \
    --desired_cross_x_nmi 1.0 \
    --target_speed_mps 10.0 \
    --ownship_speed_mps 10.0 \
    --sim_time 900.0 \
    --save_histories
```

**Output**: `policy_eval_generalized_sb3_case{X}_YYYYMMDD-HHMMSS/seed_0/`
- `policy_eval_per_episode.csv` — per-episode metrics (`collision_any`, `success_ownship`, `min_actual_sep_m_ownship`, `risk_exposure_ownship`, `completion_time_s_ownship`)
- `policy_eval_summary.json` — aggregated statistics
- `episode_histories/*.npz` — trajectory data for visualization (if `--save_histories`)

---

## Baseline Evaluation

### Script: `eval_baseline_with_hist.py`

Evaluates CORALL's rule-based guidance on a single case for comparison.

```bash
python -m maritime_rl_pkg.eval_baseline_with_hist \
    --case 6 \
    --episodes 100 \
    --seed 0 \
    --save_histories
```

**Output**: `corall_baseline_case{X}_YYYYMMDD-HHMMSS/seed_0/` — same structure as RL evaluation.

---

## Comparison & Analysis

### Script: `compare_case_metrics.py`

Generates side-by-side comparison charts of CORALL baseline vs RL policy across all 22 cases.

```bash
python -m maritime_rl_pkg.compare_case_metrics \
    --base_dir . \
    --case_numbers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --output_dir comparison_results_fixed
```

**Generated Charts**:
1. `01_min_separation_by_case.png` — Minimum separation distance per case
2. `02_path_length_by_case.png` — Path efficiency
3. `03_time_by_case.png` — Navigation completion time
4. `04_risk_exposure_by_case.png` — Integrated risk exposure
5. `05_collision_rate_by_case.png` — Collision rate (0% is target)
6. `06_success_rate_by_case.png` — Success rate (100% is target)
7. `07_guaranteed_success_summary.png` — Aggregated by complexity (2/3/4-ship)
8. `08_separation_scaling_by_ships.png` — Separation vs encounter complexity
9. `09_scaling_analysis_lines.png` — Trend lines across scenarios

---

## Episode Visualization

### Scripts: `generate_trajectory_overlays.py`, `batch_animate_eval.py`

```bash
# Single episode
python -m maritime_rl_pkg.generate_trajectory_overlays \
    policy_eval_generalized_sb3_case6_YYYYMMDD-HHMMSS/seed_0/episode_histories/case6_seed0_ep000.npz \
    --output_dir overlays/

# All episodes from one evaluation run
python -m maritime_rl_pkg.batch_animate_eval \
    policy_eval_generalized_sb3_case6_YYYYMMDD-HHMMSS/
```

---

## Observation Space

### 29-Dimensional Feature Vector

All observations are padded to a consistent 29-dim vector for policy compatibility across cases:

```
[0–7]   Own State (8 dims):
        x_norm, y_norm, sin(ψ), cos(ψ), r_norm, u_x_norm, u_y_norm, b_norm

[8–10]  Goal Features (3 dims):
        sin(goal_bearing_rel), cos(goal_bearing_rel), goal_distance_norm

[11–28] Obstacle Information (18 dims = 3 obstacles × 6 dims, zero-padded):
        dx_norm, dy_norm, sin(bearing_rel), cos(bearing_rel), du_x_norm, du_y_norm
```

**Normalization bounds**: position ±15,000 m; velocity ±15 m/s; turn rate ±0.5 rad/s. All features clipped to [−1, 1].

---

## Complete Workflow

```
1. Train
   python -m maritime_rl_pkg.train_generalized_policy_sb3
       --use_procedural_encounters --num_steps 3000000 --sim_time 900.0 ...
         └─ GENERALIZED_SB3_YYYYMMDD-HHMMSS/checkpoints/

2. Rank Checkpoints
   python -m maritime_rl_pkg.rank_checkpoints_weighted
       --training_dir GENERALIZED_SB3_YYYYMMDD-HHMMSS --sim_time 900.0 ...
         └─ Select best checkpoint (collision=0, success=1.0)

3. Evaluate RL Policy (all 22 cases)
   python -m maritime_rl_pkg.eval_generalized_policy_sb3
       --checkpoint <best>.zip --case N --episodes 100 --sim_time 900.0 ...
         └─ policy_eval_generalized_sb3_caseN_YYYYMMDD-HHMMSS/

4. Evaluate Baseline (all 22 cases)
   python -m maritime_rl_pkg.eval_baseline_with_hist
       --case N --episodes 100 ...
         └─ corall_baseline_caseN_YYYYMMDD-HHMMSS/

5. Compare Results
   python -m maritime_rl_pkg.compare_case_metrics
       --base_dir . --output_dir comparison_results_fixed

6. Visualize Episodes
   python -m maritime_rl_pkg.batch_animate_eval <eval_dir>/
```

---

## Requirements

- Python 3.8+
- Stable-Baselines3 (PPO)
- Gymnasium
- NumPy, Pandas, Matplotlib
- tqdm, rich (optional — enables `progress_bar=True` during training)
- CORALL simulator (in `third_party/`)

See `requirements.txt` for full dependencies.