# RL_CORALL_extension

Deep Reinforcement Learning for Maritime Collision Avoidance using CORALL Simulator

## Overview

This project extends the CORALL maritime collision avoidance simulator with Reinforcement Learning (RL) training and evaluation capabilities. A single generalized PPO policy is trained using curriculum learning to handle collision avoidance across 22 different multi-ship encounter scenarios (2-ship, 3-ship, and 4-ship cases).

**Key Features:**
- **Curriculum Learning**: Policy trained progressively on 2-ship → 3-ship → 4-ship scenarios
- **Generalized Policy**: Single checkpoint that works across all 22 CORALL cases
- **Baseline Comparison**: CORALL's rule-based guidance vs RL policy
- **Normalized Observations**: 29-dimensional state space with stable normalization
- **Comprehensive Metrics**: Collision rate, success rate, path length, separation distances, risk exposure

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
│     Reward: 6-component shaping function                       │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼──────────────────┐    ┌───▼────────────────────┐
   │ Environment Stack     │    │ Obstacle Ships         │
   │ (MultiShipParallelEnv)│    │ (Agents 1-K)           │
   │                       │    │                        │
   │ RandomCaseEnv         │    │ Fixed Heading & Speed  │
   │ (curriculum wrapper)  │    │ CORALL Imazu Cases     │
   │        ↓              │    │                        │
   │ env_multi_agent_ppo   │    │ (1-22 scenarios)       │
   │ (PettingZoo)          │    │                        │
   │        ↓              │    │ Infrastructure ready   │
   │ SingleAgentOwnshipEnv │    │ for future MARL        │
   │ (base env per case)   │    │                        │
   │                       │    │                        │
   └───────────────────────┘    └────────────────────────┘
        │
        │ Observation: 29-dim [own_state + goal_features + obstacles]
        │ Reward: Progress + Risk + Separation + Collision + Success
        │
```

---

## Environment Stack

### Core Environments

**1. Single-Agent Base Environment** (`env_single_agent_sb3.py`)
- Wraps individual CORALL case (1-22)
- Handles vehicle dynamics, collision detection, CPA calculations
- Observation size varies by case (14-26 dims for obstacles only)
- Ownship: RL-controlled | Obstacles: CORALL scripted

**2. Random Case Curriculum Wrapper** (`env_random_case_sb3.py`)
- **Purpose**: Enable multi-case training in a single policy
- **Curriculum Phases**:
  - Phase 1 (0-500k steps): Cases 1-4 (2-ship only) - 100% probability
  - Phase 2 (500k-1M steps): Cases 1-11 (2-ship + 3-ship) - Blend: 30% 2-ship, 70% 3-ship
  - Phase 3 (1M-2.5M steps): Cases 1-22 (all) - Blend: 20% 2-ship, 30% 3-ship, 50% 4-ship
- **Observation Padding**: All cases normalized to 29-dim (8 own + 3 goal + 18 obstacles padded)
- **Randomization**: Each reset samples random case + random seed

**3. Multi-Agent Environment** (`env_multi_agent_ppo.py`)
- **Purpose**: Foundation for future Multi-Agent RL (MARL) extension
- **Architecture**: PettingZoo ParallelEnv with all agents controllable
- **Current Use**: Single-agent training (agent 0 = ownship only)
- **Design**: Agents 1-K (obstacles) can accept actions but use scripted behavior for stability
- **Key Features**:
  - Sophisticated reward shaping with 6 components (progress, risk, separation, warning, collision, success)
  - Per-agent metrics tracking for multi-agent evaluation
  - Agent-level termination/truncation (individual agents can reach goal separately)
  - Centralized pairwise CPA cache for efficient multi-agent risk computation
  - AABB broad-phase filtering for scalable collision detection (5+ agents)

**4. Baseline Environment** (`env_baseline.py`)
- CORALL's rule-based reactive avoidance guidance
- Same observation/reward space as RL for fair comparison
- Ownship: CORALL waypoint planner + reactive avoidance
- Obstacles: Same CORALL scripted trajectories

---

## Single vs Multi-Agent Training Strategy

### Current Approach: Single-Agent with Multi-Agent Infrastructure

The project uses `env_multi_agent_ppo.py` for **single-agent RL training** (agent 0 controls ownship only):

```
Single-Agent Training Pipeline:
┌─────────────────────────────────────────┐
│  env_multi_agent_ppo.py                 │
│  (Multi-Agent Infrastructure)           │
├─────────────────────────────────────────┤
│ Agent 0 (Ownship):                      │
│   - RL-controlled with PPO              │
│   - Receives observation + reward       │
│   - Learns collision avoidance          │
│                                         │
│ Agents 1-K (Obstacles):                 │
│   - Action space available (unused)     │
│   - Fixed scripted behavior             │
│   - CORALL trajectories                 │
│   - No RL training                      │
└─────────────────────────────────────────┘
        ↓
  RandomCaseEnv wrapper
        ↓
  SB3 PPO training
```

---

## Training Pipeline

### Single-Agent RL Policy Training

**Script**: `train_generalized_policy_sb3.py`

**Configuration**:
- Algorithm: Stable-Baselines3 PPO
- Network: 2-layer MLP [256, 256] with Tanh activation
- Hyperparameters:
  - Learning rate: 1e-4 (reduced from 3e-4 for stability at curriculum transitions)
  - N-steps: 256 | Batch size: 256
  - Gamma: 0.99 | GAE Lambda: 0.95
  - Clip range: 0.2
- Total training: 2.5M steps (~10-12 hours GPU)

**Training Workflow**:
```python
RandomCaseEnv (curriculum wrapper)
    ↓
SingleAgentOwnshipEnv (base environment)
    ↓
RewardNormalizerByShipCount (normalize by case difficulty)
    ↓
Monitor (SB3 for episode tracking)
    ↓
PPO.learn() (training loop with GeneralizedTrainingMetricsCallback)
```

**Reward Function**:
```python
{
    'progress': 200.0,       # Delta progress toward goal
    'risk': -15.0,           # Moderate risk penalty
    'separation': 2.0,       # Safe separation bonus
    'collision': -600.0,     # Hard collision penalty
    'success': 250.0,        # Goal completion bonus
}
```

**Training Command**:
```bash
python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --num_steps 2500000 \
    --train_batch 256 \
    --rollout_frag 256 \
    --lr 1e-4 \
    --cases 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --checkpoint_freq 50000
```

**Output**: `GENERALIZED_SB3_YYYYMMDD-HHMMSS/`
- `checkpoints/generalized_checkpoint_*.zip` (periodic checkpoints)
- `best_checkpoint.zip` (best policy by moving average return)
- `training_metrics.json` (convergence curves)

---

## Baseline Evaluation

### CORALL Rule-Based Guidance

**Script**: `eval_baseline_with_hist.py`

Evaluates the CORALL simulator's built-in rule-based guidance on a single case.

**Environment Flow**:
```python
CORALLComparisonEnv (baseline environment)
    ↓
Ownship: CORALL planning + reactive_avoidance guidance
Obstacles: Fixed CORALL trajectories
    ↓
Monitor per-episode metrics (collision, success, path length, etc.)
    ↓
Save trajectory histories (NPZ) for visualization
```

**Baseline Command** (single case):
```bash
python -m maritime_rl_pkg.eval_baseline_with_hist \
    --case 18 \
    --episodes 100 \
    --seed 0 \
    --save_histories
```

**Output**: `corall_baseline_case{X}_YYYYMMDD-HHMMSS/seed_{seed}/`
- `policy_eval_per_episode.csv` - Per-episode metrics with columns:
  - `episode_return`, `collision_any`, `success_ownship`, `min_actual_sep_m_ownship`, `risk_exposure_ownship`
- `episode_histories/case{X}_seed{seed}_ep{Y}.npz` - Trajectory data for visualization
- `policy_eval_summary.json` - Aggregated statistics

---

## RL Policy Evaluation

### Trained Policy Evaluation

**Script**: `eval_generalized_policy_sb3.py`

Evaluates the trained generalized policy on any CORALL case (1-22).

**Environment Flow**:
```python
RandomCaseEnv (to handle 29-dim observation normalization)
    ↓
SingleAgentOwnshipEnv (base environment for specific case)
    ↓
PPO.predict(obs, deterministic=True) (policy inference)
    ↓
Monitor per-episode metrics
    ↓
Save trajectory histories (NPZ) for visualization
```

**RL Evaluation Command** (single case):
```bash
python -m maritime_rl_pkg.eval_generalized_policy_sb3 \
    --checkpoint GENERALIZED_SB3_20260510-134752/best_checkpoint.zip \
    --case 18 \
    --episodes 100 \
    --seed 0 \
    --save_histories
```

**Batch Evaluation** (all 22 cases):
```bash
# Evaluate on all cases with curriculum distribution
for case in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    python -m maritime_rl_pkg.eval_generalized_policy_sb3 \
        --checkpoint GENERALIZED_SB3_20260510-134752/best_checkpoint.zip \
        --case $case --episodes 100 --seed 0 --save_histories
done
```

**Output**: `policy_eval_generalized_sb3_case{X}_YYYYMMDD-HHMMSS/seed_{seed}/`
- Same structure as baseline evaluation
- Per-episode metrics with collision data from `collision_any` CSV column

---

## Comparison & Analysis

### Metrics Comparison

**Script**: `maritime_rl_pkg/compare_case_metrics.py`

Compares baseline vs RL policy across all 22 cases with 9 visualization charts.

**Data Pipeline** (corrected):
```
Per-Episode CSV Files (policy_eval_per_episode.csv)
    ↓
Read 'collision_any' and 'success_ownship' columns
    ↓
Compute mean collision_rate, success_rate per case
    ↓
Generate comparison charts with proper axis scaling
```

**Comparison Command**:
```bash
python -m maritime_rl_pkg.compare_case_metrics \
    --base_dir . \
    --case_numbers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --output_dir comparison_results_fixed
```

**Generated Charts**:
1. `01_min_separation_by_case.png` - Min separation distance (higher is safer)
2. `02_path_length_by_case.png` - Path efficiency (shorter is better)
3. `03_time_by_case.png` - Navigation time (shorter is better)
4. `04_risk_exposure_by_case.png` - Risk exposure integration (lower is better)
5. `05_collision_rate_by_case.png` - **Collision rate per case** (0% is target)
6. `06_success_rate_by_case.png` - Success rate per case (100% is target)
7. `07_guaranteed_success_summary.png` - Aggregated metrics by complexity (2/3/4-ship)
8. `08_separation_scaling_by_ships.png` - Separation vs complexity
9. `09_scaling_analysis_lines.png` - Trend analysis across scenarios

---

## Episode Visualization

### Trajectory Overlays

**Script**: `generate_trajectory_overlays.py` or `batch_animate_eval.py`

Generates animated visualization of specific episodes showing:
- Ship trajectories
- Collision detection zones
- Risk visualization
- Separation distances

**Animation Command** (single episode):
```bash
python -m maritime_rl_pkg.generate_trajectory_overlays \
    policy_eval_generalized_sb3_case18_20260511-110001/seed_0/episode_histories/case18_seed0_ep000.npz \
    --output_dir overlays/
```

**Batch Animation** (all episodes from evaluation):
```bash
python -m maritime_rl_pkg.batch_animate_eval \
    policy_eval_generalized_sb3_case18_20260511-110001/
```

---

## Observation Space

### 29-Dimensional Feature Vector

All cases normalized to consistent 29-dim observation space:

```
[0-7]   Own State (8 dims):
        - x_norm, y_norm (clipped to [-1, 1])
        - sin(psi), cos(psi) (circular heading representation)
        - r_norm (turn rate, clipped to [-1, 1])
        - u_x_norm, u_y_norm (velocity components, clipped to [-1, 1])
        - b_norm (actuator bias, clipped to [-1, 1])

[8-10]  Goal Features (3 dims):
        - sin(goal_bearing_rel), cos(goal_bearing_rel) (relative bearing to goal)
        - goal_distance_norm (normalized to [-1, 1])

[11-28] Obstacle Information (18 dims = 3 obstacles × 6 dims):
        For each obstacle (up to 3, zero-padded):
        - dx_norm, dy_norm (relative position, clipped to [-1, 1])
        - sin(bearing_rel), cos(bearing_rel) (relative bearing to obstacle)
        - du_x_norm, du_y_norm (relative velocity, clipped to [-1, 1])
```

**Normalization Bounds**:
- Position: ±15,000m (scenario extent)
- Velocity: ±15 m/s (max typical ship speeds)
- Turn rate: ±0.5 rad/s
- All features clipped to [-1, 1] for stable learning

---

## Complete Training & Evaluation Workflow

### Step 1: Train Generalized Policy
```bash
# Start training (2.5M steps, ~12 hours on GPU)
python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --num_steps 2500000 \
    --train_batch 256 \
    --rollout_frag 256 \
    --lr 1e-4 \
    --checkpoint_freq 50000

# Output: GENERALIZED_SB3_YYYYMMDD-HHMMSS/best_checkpoint.zip
```

### Step 2: Evaluate RL Policy on All Cases
```bash
# Batch evaluate on all 22 cases
for case in {1..22}; do
    python -m maritime_rl_pkg.eval_generalized_policy_sb3 \
        --checkpoint GENERALIZED_SB3_20260510-134752/best_checkpoint.zip \
        --case $case --episodes 100 --seed 0 --save_histories
done
```

### Step 3: Generate Baseline Results (if needed)
```bash
# Baseline evaluation for comparison
for case in {1..22}; do
    python -m maritime_rl_pkg.eval_baseline_with_hist \
        --case $case --episodes 100 --seed 0 --save_histories
done
```

### Step 4: Compare Results (performance analysis)
```bash
# Generate comparison charts
python -m maritime_rl_pkg.compare_case_metrics \
    --base_dir . \
    --case_numbers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --output_dir comparison_results_final
```

### Step 5: Visualize Best Episodes (if desired)
```bash
# Generate trajectory overlays for visualization
python -m maritime_rl_pkg.batch_animate_eval \
    policy_eval_generalized_sb3_case18_20260511-110001/
```
---

## Requirements

- Python 3.8+
- Stable-Baselines3 (PPO)
- Gymnasium
- NumPy, Pandas, Matplotlib
- CORALL simulator (in `third_party/`)

See `requirements.txt` for full dependencies.