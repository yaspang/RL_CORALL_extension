# Training Improvements: Reward Normalization & Hyperparameter Tuning

## Changes Implemented

### 1. **Reward Normalization by Scenario Type** ✅
- **File**: `maritime_rl_pkg/env_reward_normalizer.py` (new)
- **How it works**: 
  - Maintains running statistics (mean, std) of episode returns for 2-ship, 3-ship, and 4-ship cases separately
  - Normalizes individual step rewards using Z-score normalization: `(r - mean) / (std + eps)`
  - Adapts statistics online as training progresses
  - Ensures all scenario types contribute equally despite different reward magnitudes

- **Expected benefits**:
  - Smoother convergence across heterogeneous difficulty levels
  - Reduces extreme variance in per-episode returns
  - Prevents policy from specializing on easy (2-ship) vs hard (4-ship) cases
  - Makes training plots more interpretable

- **Usage**:
  ```bash
  # With reward normalization (default, recommended)
  python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --num_steps 3000000 \
    --train_batch 512 \
    --rollout_frag 256
  
  # Disable for ablation studies
  python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --no_normalize_rewards \
    --num_steps 3000000
  ```

### 2. **Increased Batch Size** ✅
**Old**: `batch_size=256, rollout_length=128`  
**New**: `batch_size=512, rollout_length=256`

**Why this helps:**
- Larger batches = more stable gradient estimates
- Reduces variance from individual episode noise
- Better use of GPU memory
- More robust convergence with high-variance multi-task learning

---

## Hyperparameter Alignment

### Configuration Recommendations by Setup

#### **Setup A: Default (Recommended for your case)**
```python
--num_steps 3000000          # 3M steps is good starting point
--train_batch 512             # Increased for stability
--rollout_frag 256           # Double the rollout (PPO n_steps)
--lr 0.0001                  # LOWER learning rate (was 3e-4)
--gamma 0.99                 # Keep unchanged
--checkpoint_freq 20000      # Frequent checkpoints to monitor
--normalize_rewards          # ON (default)
```

**Why these values work together:**
- Larger batch + rollout → larger policy gradient updates → need lower LR to avoid instability
- Lower LR → more stable learning with heterogeneous rewards
- 256 rollout allows PPO to see 256/512 = 0.5 epoch per update (good coverage)
- Normalization + lower LR = smooth, interpretable convergence

#### **Setup B: Memory-Constrained (if GPU OOM)**
```python
--train_batch 384            # Compromise between 256 and 512
--rollout_frag 192
--lr 0.00015                 # Slightly higher than Setup A
--num_workers 1              # Reduce parallelism if needed
```

#### **Setup C: Aggressive Learning (for fast iteration)**
```python
--train_batch 512            # Keep large batch
--rollout_frag 256
--lr 0.00005                 # LOWER than Setup A (safer)
--gamma 0.995                # Slightly higher discount (more foresight)
--checkpoint_freq 10000      # More frequent monitoring
```

---

## Key Hyperparameter Relationships

### Learning Rate Guidance

| Batch Size | Recommended LR | Rationale |
|-----------|----------------|-----------|
| 256       | 3e-4 (old)     | Baseline; works with smaller updates |
| 512       | 1e-4           | **Recommended** for your setup |
| 512       | 5e-5           | Conservative; very stable, slower convergence |
| 512       | 2e-4           | Aggressive; may cause instability |

### Rollout Fragment Length
- **Rule of thumb**: `rollout_frag ≈ batch_size / 2` to `batch_size`
- **512 batch → rollout 256-512 recommended**
- Larger rollout = more trajectory diversity before policy update
- PPO internally does `n_epochs=10` passes over collected data
- Total experiences per update: `batch_size × n_epochs / (buffer_size / rollout_frag)`

### Number of Epochs
- **Current**: `n_epochs=10`
- **Assessment**: This is good! Don't change.
- With batch_size=512, this allows enough policy refinement without overfitting

---

## What Changed in Code

### 1. New Wrapper: `env_reward_normalizer.py`
```python
env = RandomCaseEnv(cases_to_train=[1, 6, 21], ...)
env = RewardNormalizerByShipCount(env, normalize_rewards=True)
env = Monitor(env)
env = EpisodeReturnTracker(env)
```

### 2. Updated Arguments in `train_generalized_policy_sb3.py`
- `--train_batch`: 256 → **512** (default)
- `--rollout_frag`: 128 → **256** (default)
- `--normalize_rewards`: NEW flag (default: True)
- `--no_normalize_rewards`: NEW flag to disable for ablations

### 3. Updated Config JSON
Config now includes:
```json
{
  "train_batch_size": 512,
  "rollout_fragment_length": 256,
  "normalize_rewards": true,
  ...
}
```

---

## Expected Training Behavior

### Before (Current):
- Wild swings in episode return plot (±5000 range)
- Hard to tell if policy is converging
- Occasional crashes from extreme rewards
- Per-ship breakdown shows 4-ship cases dragging down overall stats

### After (With Normalization + New Batch Size):
- Smoother episode return curves (±1000-2000 range)
- Clear convergence trend visible
- Per-ship-count returns converge toward 0 (normalized)
- Training is more stable and GPU-friendly
- Individual case curves (2-ship, 3-ship, 4-ship) should all show positive trends

---

## Monitoring During Training

Watch for these signs of good training:

✅ **Good Signs:**
- Episode returns trending upward (despite noise)
- Per-scenario (2/3/4-ship) curves improve independently
- Model saves frequently without errors
- GPU utilization ~80-90%

⚠️ **Warning Signs:**
- Episode returns suddenly tank (may need lower LR)
- Batch size consistently causes OOM (reduce batch/rollout)
- Per-ship curves diverge (some improving, others degrading)

---

## Recommended First Run

```bash
python -m maritime_rl_pkg.train_generalized_policy_sb3 \
    --num_steps 3000000 \
    --checkpoint_freq 20000 \
    --train_batch 512 \
    --rollout_frag 256 \
    --lr 0.0001 \
    --gamma 0.99 \
    --cases 1 6 21 \
    --master_seed 42
```

**Expected runtime:**
- GPU (V100): ~8-10 hours
- GPU (A100): ~4-6 hours
- CPU: ~20-30 hours (not recommended)

---

## Future Improvements (Optional)

1. **Curriculum Learning** (highest impact):
   - Phase 1 (0-1M steps): 2-ship cases only
   - Phase 2 (1-2M steps): Add 3-ship cases
   - Phase 3 (2-3M steps): Add 4-ship cases
   - Expected: Cleaner convergence curves, ~10-15% better final performance

2. **Learning Rate Scheduling**:
   ```python
   # Linear decay from 0.0001 to 0.00001 over training
   --learning_rate_schedule "linear_decay"
   ```

3. **Higher n_epochs for multi-task**:
   - Current: n_epochs=10
   - Consider: n_epochs=15-20 (allows more exploration of each batch)

4. **Action Space Smoothing**:
   - Add small Gaussian noise to actions during early training
   - Prevents premature policy lock-in on easy cases

---

## Troubleshooting

### Q: "Out of memory" with batch_size=512
A: Try Setup B (batch=384, rollout=192) or reduce `num_workers`

### Q: "Returns still too noisy even with normalization"
A: 
1. Lower LR to 0.00005
2. Increase `n_epochs` to 15-20
3. Increase `train_batch` to 768 if memory allows

### Q: "One scenario type converges but others don't"
A: Curriculum learning is needed. See "Future Improvements" section.

### Q: "Training is very slow"
A: Consider increasing `num_workers` to 4 or 8 for more parallel environments.

---

## Questions?

For issues or unexpected behavior, check:
1. `logger_diagnostics.txt` in output directory
2. Raw reward statistics in training config JSON
3. Plot the normalized vs raw rewards to compare
