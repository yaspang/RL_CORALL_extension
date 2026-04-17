# Single-Agent PPO Training Setup (Stable-Baselines3)

1. **[env_single_agent_sb3.py](maritime_rl_pkg/env_single_agent_sb3.py)** — Gymnasium wrapper around multi-agent env defined from PettingZoo (future build up to MARL env)
   - Only ownship (agent_0) receives RL actions
   - Obstacles use scripted actions (maintain fixed heading/speed toward ownship)
   - Compatible with Stable-Baselines3 single-agent algorithms

2. **[train_single_agent_sb3.py](maritime_rl_pkg/train_single_agent_sb3.py)** — Single-agent PPO trainer (SB3)
   - Uses Stable-Baselines3 PPO implementation
   - Saves checkpoints every 10k steps + best checkpoint
   - Logs training metrics and convergence plots

3. **[eval_single_agent_sb3.py](maritime_rl_pkg/eval_single_agent_sb3.py)** — Evaluation script
   - Deterministic rollouts (no exploration)
   - Supports both single-agent case-specific and generalized policies
   - Saves per-episode metrics + episode histories (optional)
   - Computes ownship-only metrics for comparison

---

## Quick Start

### Train (100k steps, ~30-50 min per case)
```bash
python -m maritime_rl_pkg.train_single_agent_sb3 \
  --case 6 \
  --num_steps 100000 \
  --seed 0
```

Output: `SINGLE_AGENT_SB3_case6_YYYYMMDD-HHMMSS/`
- `checkpoints/checkpoint_*.zip` (periodic saves every 10k steps)
- `best_checkpoint.zip` (highest validation reward)
- `training_return.png` (convergence plot)
- `training_metrics.json` (training logs)

### Evaluate (100 episodes, save best + all histories)
```bash
python -m maritime_rl_pkg.eval_single_agent_sb3 \
  --checkpoint "SINGLE_AGENT_SB3_case6_YYYYMMDD-HHMMSS/best_checkpoint.zip" \
  --case 6 \
  --episodes 100 \
  --seed 0 \
  --save_histories \
  --save_all_histories
```

Output: `policy_eval_single_sb3_case6_YYYYMMDD-HHMMSS/seed_0/`
- `policy_eval_per_episode.csv` — Per-episode metrics
- `policy_eval_summary.json` — Aggregated mean/std + per-episode
- `episode_histories/trained_case6_seed0_*.npz` — Episodes for visualization

### Visualize & Analyze (same tools as before!)
```bash
# Generate GIFs for all evaluation episodes
python -m maritime_rl_pkg.batch_animate_eval \
  --eval_dir "policy_eval_single_sb3_case6_YYYYMMDD-HHMMSS/seed_0" \
  --fps 20 --stride 4

# Analyze metrics across episodes
python -m maritime_rl_pkg.analyze_eval_metrics \
  --csv "policy_eval_single_sb3_case6_YYYYMMDD-HHMMSS/seed_0/policy_eval_per_episode.csv"
```

---

## Training All Cases (Recommended Sequence)

```bash
# Case 1 (loose encounters, 100k steps)
python -m maritime_rl_pkg.train_single_agent_sb3 --case 1 --num_steps 100000 --seed 0

# Case 6 (medium difficulty, 250k steps)
python -m maritime_rl_pkg.train_single_agent_sb3 --case 6 --num_steps 250000 --seed 0

# Case 21 (tight encounters, 250k-500k steps recommended)
python -m maritime_rl_pkg.train_single_agent_sb3 --case 21 --num_steps 500000 --seed 0
```

---

## Key Differences from Multi-Agent

| Aspect | Multi-Agent | Single-Agent (SB3) |
|--------|-------------|-------------------|
| **Training** | All agents learn cooperatively | Only ownship learns |
| **Obstacles** | Learn collision avoidance together | Scripted (fixed heading + speed) |
| **Observation** | Each agent sees others | Ownship sees all others |
| **Reward** | Per-agent + inter-agent coordination | Ownship-only rewards |
| **Metrics** | Per-agent + aggregate | Ownship-only |
| **Framework** | RLlib (multi-agent) | Stable-Baselines3 (single-agent) |

---

## Reward Function

Located in [env_multi_agent_ppo.py](maritime_rl_pkg/env_multi_agent_ppo.py) `compute_rewards_and_dones()`:

- `w_along = 1.0` — Progress toward waypoint (sparse, per step)
- `w_risk = -50.0` — Risk penalty (strong to encourage proactive avoidance)
- `w_collision = -100.0` — Collision penalty (reacti penalty)
- `w_time = -0.005` — Small per-step cost (encourage efficiency)
- `w_success = 25.0` — Goal completion bonus

**Tuning tips:**
- Increase `w_risk` → More conservative avoidance behavior
- Decrease `w_along` → Less aggressive waypoint chasing
- Add proximity penalty → Discourage loitering in near-miss zones

---

## Troubleshooting

**ImportError: No module named 'stable_baselines3'**  
→ Install: `pip install stable-baselines3`

**Observation shape mismatch**  
→ Model expects 12-dim (Case 1), got 22-dim (generalized)
→ Solution: Use correct checkpoint for case, or retrain with correct generalization

**Collisions still occurring**  
→ Likely reward imbalance: try increasing `w_risk`, decreasing `w_along`
→ Or train longer: increase `num_steps` to 250k-500k

**No episode histories saved**  
→ Use `--save_histories` or `--save_all_histories` flags

# Compare CSV files side-by-side
python -c "
import pandas as pd
rl = pd.read_csv('policy_eval_single_case6_.../seed_0/policy_eval_per_episode.csv')
baseline = pd.read_csv('baseline_eval/seed_0/metrics.csv')
print('RL goal_progress:', rl['goal_progress_ownship'].mean())
print('Baseline goal_progress:', baseline['goal_progress'].mean())
"
```

---

## Architecture

```
SingleAgentOwnshipEnv (Gymnasium API)
    ↓ wraps ↓
MultiShipParallelEnv (PettingZoo multi-agent)
    ↓
CORALL simulation (vessel_dynamics, CPA, risk)

Training:
- RL action for ownship
- Scripted actions for obstacles (maintain heading)
- Step multi-agent env with both
- Extract ownship obs/reward/done
- PPO updates on ownship policy only
```

---

## Metrics Explanation

**Ownship-only metrics** (comparable across RL and baseline):

- `episode_return_ownship` — Cumulative discounted reward
- `goal_progress_ownship` — Fraction of route completed (0-1)
- `success_ownship` — 1 if reached goal collision-free, else 0
- `min_dcpa_m_ownship` — **Predicted** minimum Distance at Closest Point of Approach (forward-looking)
  - What would happen if both vessels maintain current course/speed
  - reflects avoidance timing
- `min_actual_sep_m_ownship` — **Actual** minimum separation distance (range) observed during episode
  - Lowest `pair_dist[k]` across all timesteps and obstacle pairs
- `min_tcpa_s_ownship` — Time until closest predicted approach (seconds)
- `collision_any` — 1 if actual collision occurred (range ≤ 0m), else 0
- `near_miss_any` — 1 if near-miss threshold breached (range < 1.5×LOA), else 0
- `path_length_m_ownship` — Actual distance traveled (longer = less efficient)
- `risk_exposure_ownship` — Cumulative risk metric (sum of pairwise risk across episode)

**Interpretation:**
- High `min_actual_sep_m_ownship` (>500m) + `success_ownship`=1 → Proactive avoidance (predicted conflicts avoided early)
- Low `min_actual_sep_m_ownship` (<200m) + `success_ownship`=1 → Reactive avoidance (waited until late to maneuver)
- `min_actual_sep_m_ownship` ≈ `min_dcpa_m_ownship` → Late reaction (actual matched predicted)
- `min_actual_sep_m_ownship` >> `min_dcpa_m_ownship` → Early avoidance (actual stayed well above predicted)
