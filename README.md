# RL_CORALL_extension

LLM-Guided Reinforcement Learning for Maritime Collision Avoidance

*Extension of the CORALL maritime collision-avoidance framework*

## Overview

This repository provides a generalized Proximal Policy Optimization (PPO) policy for multi-ship maritime collision avoidance, an LLM-guided supervisory control layer that delivers explainable COLREGs maneuver intent during deployment, and a reliability evaluation framework for measuring LLM intent quality against a deterministic reference. It extends the [CORALL](https://github.com/Klins101/CORALL) maritime collision-avoidance framework.

The RL policy operates independently from the LLM and proposes discrete heading and speed actions from the observed encounter state. At specified decision intervals, the LLM evaluates nearby target-ship geometry and returns an explainable COLREGs maneuver intent, which is parsed as K_{dir} ∈ {−1, 0, +1}, representing port, stand-on/maintain, and starboard maneuver guidance respectively. An arbitration layer then constrains the PPO-proposed maneuver according to valid LLM intent and higher-priority safety logic before the final action is applied to the ownship.

The project supports both fixed Imazu encounter geometries and procedurally generated multi-ship encounters.

### Key Features

- **Generalized PPO Policy:** Single policy trained across multi-ship collision-avoidance encounters
- **Two Training Modes:** Fixed Imazu case rotation (`RandomCaseEnv`) or procedurally randomized encounters (`RandomEncounterEnv`)
- **Curriculum Learning:** Encounter complexity progresses from one target to as many as three targets
- **29-Dimensional Observation Space:** Ownship, goal, and relative target-ship features
- **Discrete Heading and Speed Control:** `MultiDiscrete([7, 5])` action space
- **LLM-Guided Maneuver Intent:** COLREGs reasoning based on range, DCPA, TCPA, relative bearing, and collision risk
- **Intent-Action Arbitration:** Valid LLM intent constrains the PPO-proposed heading direction without directly modifying the PPO network input
- **LLM Reliability Evaluation:** Measures Valid Intent Rate and Strict Action Accuracy against a deterministic geometry/COLREGs reference
- **Safety-Oriented Evaluation:** Collision rate, success rate, risk exposure, minimum separation, and completion time
- **Constraint-Based Checkpoint Selection:** Collision-free and fully successful checkpoints prioritized before secondary metrics

---

## Reported Results

The final PPO policy was evaluated over the fixed Imazu encounter set using 100 episodes per case.

| Metric | Result |
|---|---|
| Collision rate | 0.0% |
| Ownship success rate | 100.0% |
| LLM valid intent rate | 72.7% |
| LLM strict action accuracy | 96.0% |

LLM reliability results were computed over 9,507 total LLM queries across Cases 1, 6, and 18. Strict action accuracy was evaluated on the high-confidence single-label subset.

---

## Result Data Availability

Full episode-level evaluation outputs and trajectory histories are not included in this repository because of their size. The aggregate results reported in the associated paper are summarized in the [Reported Results](#reported-results) section above. The pretrained PPO checkpoint used for evaluation is provided in `pretrained/`, and the included evaluation scripts can be used to regenerate the policy and CORALL baseline results.

---

## Pretrained Policy

The pretrained Stable-Baselines3 PPO checkpoint used for the reported evaluation is included in this repository at `pretrained/generalized_checkpoint_850000_steps.zip`.

- Training steps: 850,000
- Observation: 29 dimensions
- Action space: `MultiDiscrete([7, 5])` — heading (7 bins) + speed (5 bins)
- Actor: [256, 256], Tanh
- Critic: [256, 256], Tanh

---

## Reproducing the Reported Results

### PPO Policy Evaluation

The reported policy results use the PPO checkpoint at 850,000 training steps. Run for each Imazu case (1–22):

```bash
python -m src.eval_generalized_policy_sb3 \
    --checkpoint pretrained/generalized_checkpoint_850000_steps.zip \
    --case 6 \
    --episodes 100 \
    --seed 0 \
    --desired_cross_x_nmi 1.0 \
    --target_speed_mps 10.0 \
    --ownship_speed_mps 10.0 \
    --sim_time 900.0 \
    --save_histories
```

Repeat with `--case N` for cases 1–22.

### CORALL Baseline

```bash
python -m src.baseline_eval.eval_baseline_with_hist \
    --case 6 \
    --episodes 100 \
    --seed 0 \
    --save_histories
```

Repeat with `--case N` for cases 1–22.

### LLM Reliability Evaluation

```bash
python -m src.llm_integration.eval_llm_reliability \
    --eval_dir <LLM-evaluation-directory>
```

Primary reported metrics: **Valid Intent Rate** and **Strict Action Accuracy**. LLM reliability metrics can be regenerated from `llm_intent_log.csv` files produced during an LLM-enabled evaluation run. Because these experiments rely on an external API, future call-level responses may differ from those reported in the paper.

---

## LLM-Guided Control Architecture

The LLM is used as a supervisory intent source rather than as the primary low-level controller.

```
                     Encounter State
                     /             \
                    /               \
                   v                 v
             PPO Policy       LLM Decision Module
                   |                 |
          Proposed action       Parsed intent
         [heading, speed]          K_dir
                   \                 /
                    \               /
                     v             v
                   Intent-Action Arbiter
                          |
                    Final action
                          |
                       Ownship
```

The PPO policy receives only the environment observation and is not conditioned directly on K&#8336;&#7433;&#7523;. During evaluation, the LLM is queried periodically and produces a high-level maneuver recommendation. When a valid intent is available, the arbiter constrains the heading component of the PPO proposal to remain consistent with the prescribed maneuver direction. If the PPO heading is already compatible with the intent, the proposal is retained. If the LLM output is unavailable or invalid, control falls back to the PPO proposal; deterministic emergency safety logic retains higher authority.

### LLM Inputs
For target ships within the local encounter region (≤ 3 nmi), the LLM receives:
- Range, DCPA, TCPA
- Relative bearing
- Collision-risk value
- Encounter phase
- Deterministic COLREGs rule hint (from CORALL `decision_making()`)

The LLM response contains a proposed maneuver, the applicable COLREGs rule, and supporting rationale. The maneuver is parsed to:
```
K_dir = -1  -> port
K_dir =  0  -> stand on / maintain
K_dir = +1  -> starboard
```

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
│     Action: MultiDiscrete([7, 5])                              │
│       └─ Heading: 7 discrete bins                              │
│       └─ Speed:   5 discrete bins                              │
│     Observation: 29-dim normalized state vector                │
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

## PPO Network Architecture

The PPO policy is implemented using Stable-Baselines3 `MlpPolicy`. The actor and critic use separate feed-forward hidden branches, each with two fully connected layers of 256 neurons and Tanh activation.

```
PPO Policy from Stable-Baselines3
├─ Observation: 29-D normalized state
├─ Action: MultiDiscrete([7,5])
│  ├─ Heading: 7 discrete bins
│  └─ Speed:   5 discrete bins
├─ Actor:  [256, 256], Tanh
└─ Critic: [256, 256], Tanh
```

The actor produces categorical heading and speed decisions, while the critic estimates the scalar state value V(s).

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
python -m src.train_generalized_policy_sb3 \
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
python -m src.train_generalized_policy_sb3 \
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
python -m src.rank_checkpoints_weighted \
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
python -m src.eval_generalized_policy_sb3 \
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
python -m src.baseline_eval.eval_baseline_with_hist \
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
python -m src.performance_eval.compare_case_metrics \
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
python -m src.visualizations.generate_trajectory_overlays \
    policy_eval_generalized_sb3_case6_YYYYMMDD-HHMMSS/seed_0/episode_histories/case6_seed0_ep000.npz \
    --output_dir overlays/

# All episodes from one evaluation run
python -m src.visualizations.batch_animate_eval \
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
1. Train PPO
   python -m src.train_generalized_policy_sb3
       --use_procedural_encounters --num_steps 3000000 --sim_time 900.0 ...
         └─ GENERALIZED_SB3_YYYYMMDD-HHMMSS/checkpoints/

2. Rank Checkpoints
   python -m src.rank_checkpoints_weighted
       --training_dir GENERALIZED_SB3_YYYYMMDD-HHMMSS --sim_time 900.0 ...
         └─ Select best checkpoint (collision=0, success=1.0)

3. Evaluate PPO across encounter cases
   python -m src.eval_generalized_policy_sb3
       --checkpoint <best>.zip --case N --episodes 100 --sim_time 900.0 ...
         └─ policy_eval_generalized_sb3_caseN_YYYYMMDD-HHMMSS/

4. Evaluate CORALL baseline
   python -m src.baseline_eval.eval_baseline_with_hist
       --case N --episodes 100 ...
         └─ corall_baseline_caseN_YYYYMMDD-HHMMSS/

5. Compare RL and baseline performance
   python -m src.performance_eval.compare_case_metrics
       --base_dir . --output_dir comparison_results/

6. Evaluate PPO + LLM integration
   python -m src.eval_generalized_policy_sb3
       --checkpoint <best>.zip --case N --episodes 100 --llm --llm_provider openai
       --llm_interval 10 --llm_env_file third_party/CORALL/.env --save_histories
         └─ results_llmapi_caseN_interval10/

7. Evaluate LLM reliability
   python -m src.llm_integration.eval_llm_reliability
       --eval_dir results_llmapi_caseN_interval10/
         └─ seed_0/results_llm_reliability/

8. Generate trajectory and reliability figures
   python -m src.visualizations.generate_trajectory_overlays ...
   python -m src.llm_integration.plot_llm_reliability_table ...
```

> **Note:** Steps 6–8 are evaluation-time extensions. The LLM is never used during PPO training and does not influence the trained policy weights.

---

## LLM Reliability Evaluation

### Script: `eval_llm_reliability.py`

The reliability evaluator operates on `llm_intent_log.csv` and scores one LLM decision per query interval. Rows belonging to the same query are collapsed and the highest-risk target is used as the primary encounter for reference scoring.

Two primary metrics are reported:

**Valid Intent Rate** — fraction of all LLM queries that both return successfully and produce a parseable maneuver intent:

```
Valid Intent Rate = N(successful response and parseable K_dir) / N(LLM queries)
```

This measures availability of usable LLM guidance and does not imply that the maneuver itself is geometrically correct.

**Strict Action Accuracy** — for valid intents where the deterministic COLREGs/geometry reference defines a high-confidence single expected maneuver:

```
Strict Action Accuracy = N(correct parsed intents) / N(valid high-confidence single-label intents)
```

This metric measures agreement against the deterministic COLREGs reference rather than formal COLREGs compliance verification. The evaluator additionally records diagnostic metrics including parse success, maneuver consistency, per-rule accuracy, missed maneuvers, unnecessary maneuvers, and confusion counts.

---

## Requirements

- Python 3.8+
- Stable-Baselines3 (PPO)
- Gymnasium
- NumPy, Pandas, Matplotlib
- tqdm, rich (optional — enables `progress_bar=True` during training)
- CORALL simulator (in `third_party/`)

See `requirements.txt` for full dependencies.

---

## Upstream Project and Attribution

This work builds upon and adapts components of the **CORALL** framework developed for explainable COLREGs-compliant maritime collision avoidance:

**CORALL:** https://github.com/Klins101/CORALL

Components adapted or extended from CORALL include portions of the maritime encounter simulation, collision-risk formulation, COLREGs decision logic, and LLM decision-making infrastructure. The reinforcement-learning training pipeline, generalized PPO policy, procedural encounter generation, LLM-to-RL arbitration layer, and associated reliability evaluation were developed as extensions for this project.

Users of this repository should also cite the original CORALL work where appropriate. See the repository license and any third-party license notices for attribution requirements associated with adapted CORALL source code.

---

## Citation

If you use this repository, please cite the associated publication and the original CORALL work on which portions of the simulator and decision-making framework are based.

### Upstream CORALL Repository

https://github.com/Klins101/CORALL
