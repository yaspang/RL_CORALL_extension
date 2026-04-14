# Checkpoint Selection Guide

## Problem

Previously:
- RLlib only kept the **latest** checkpoint → overwrote previous ones
- After training, only 1 checkpoint was available
- Couldn't select the "best" checkpoint by eval metrics
- Lost intermediate checkpoints that may have been better

## Solution

### 1. Training Script Now Tracks Best Checkpoint

**Changes to `multi_agent_train_ppo.py`:**
- Added checkpoint configuration:
  ```python
  .checkpointing(
      keep_checkpoints_num=5,              # Keep 5 most recent checkpoints
      checkpoint_score_attr="eval_return_mean",  # Sort by eval return
      checkpoint_frequency=args.ckpt_every,
  )
  ```
- Tracks best eval return during training
- Saves `best_checkpoint_info.json` with metadata

**Training Output:**
```
Iter 5/30 | train_return=... | eval_return=45.2 | ...
  ✓ New best checkpoint at iteration 5: eval_return=45.2
Iter 10/30 | train_return=... | eval_return=42.8 | ...
Iter 15/30 | train_return=... | eval_return=48.1 | ...
  ✓ New best checkpoint at iteration 15: eval_return=48.1
...
[INFO] Best checkpoint was at iteration 15
       with eval_return_mean = 48.1
       Metadata saved to: MARL_ppo_case2_XXXXXX/seed_0/best_checkpoint_info.json

[INFO] All saved checkpoints in MARL_ppo_case2_XXXXXX/seed_0/checkpoints/:
  - checkpoint_000005
  - checkpoint_000010
  - checkpoint_000015
  - checkpoint_000020
  - checkpoint_000025
```

### 2. Find Best Checkpoint After Training

Use the new helper script:

```bash
python -m maritime_rl_pkg.maritime_rl.select_best_checkpoint \
    --run_dir "MARL_ppo_case2_20260412-091424/seed_0"
```

**Output:**
```
✓ Best checkpoint found (from training metadata)
  Iteration: 15
  Eval Return: 48.1
  Eval Ep Length: 2145

Checkpoint directory: C:\path\to\checkpoints\
```

### 3. Use Best Checkpoint for Evaluation

#### Option A: Manual (specify full checkpoint path)

```bash
python -m maritime_rl_pkg.maritime_rl.eval_trained_policy \
    --checkpoint "C:\path\to\checkpoints\checkpoint_000015" \
    --case 2 --episodes 40 --seed 0
```

#### Option B: List All Checkpoints First

Find the iteration number with best eval return:

```bash
python -m maritime_rl_pkg.maritime_rl.select_best_checkpoint \
    --run_dir "MARL_ppo_case2_20260412-091424/seed_0" \
    --list
```

**Output:**
```
All checkpoints in seed_0:
  1. checkpoint_000005
  2. checkpoint_000010
  3. checkpoint_000015  ← BEST (eval_return=48.1)
  4. checkpoint_000020
  5. checkpoint_000025
```

Then use that checkpoint:
```bash
python -m maritime_rl_pkg.maritime_rl.eval_trained_policy \
    --checkpoint "MARL_ppo_case2_20260412-091424/seed_0/checkpoints/checkpoint_000015" \
    --case 2 --episodes 40 --seed 0
```

## Files Modified

### `multi_agent_train_ppo.py`
- **Line ~185**: Added `.checkpointing()` config block
- **Line ~240**: Added `best_eval_return` and `best_checkpoint_data` tracking
- **Line ~275**: Added logic to update best checkpoint when eval improves
- **Line ~310**: Added final output listing all saved checkpoints and best checkpoint metadata

### New Files

- **`select_best_checkpoint.py`**: Utility to identify and describe best checkpoint

### Output Files (created per training run)

- **`best_checkpoint_info.json`**: Metadata about best checkpoint
  ```json
  {
    "iteration": 15,
    "eval_return_mean": 48.1,
    "eval_ep_length_mean": 2145
  }
  ```
- **`training_metrics.csv`**: Row per iteration with all metrics (fallback for best selection)
- **`checkpoints/`**: Directory with up to 5 most recent checkpoint folders

## Key Insights

### Why Keep 5 Checkpoints?
- **Too many**: Disk space, slower cleanup
- **Too few**: Might miss best checkpoint or have only 1-2 options
- **5 is reasonable**: Balances availability with storage (~1-2 GB per checkpoint)

### Why Sort by `eval_return_mean`?
- Training return can be noisy due to exploration
- Eval return (with `explore=False`) is more stable
- Best eval return = best policy for actual deployment

### What If Eval Doesn't Improve?
- Script will still mark final checkpoint
- All checkpoints are saved in chronological order
- You can manually pick any checkpoint to try

## Typical Workflow

1. **Run training:**
   ```bash
   python -m maritime_rl_pkg.maritime_rl.multi_agent_train_ppo \
       --case 2 --iters 30 --num_workers 2 --seed 0
   ```

2. **Training completes:**
   - See which iteration had best eval return
   - Checkpoints 5, 10, 15, 20, 25 are saved
   - `best_checkpoint_info.json` created

3. **Evaluate best checkpoint:**
   ```bash
   # Option 1: Use the reported best iteration
   python -m maritime_rl_pkg.maritime_rl.eval_trained_policy \
       --checkpoint "MARL_ppo_case2_XXXXXX/seed_0/checkpoints/checkpoint_000015" \
       --case 2 --episodes 40
   
   # Option 2: Use helper to find it
   python -m maritime_rl_pkg.maritime_rl.select_best_checkpoint \
       --run_dir "MARL_ppo_case2_XXXXXX/seed_0"
   ```

4. **Compare vs baseline:**
   ```bash
   python -m maritime_rl_pkg.maritime_rl.eval_baseline --case 2 --episodes 40
   ```

## Troubleshooting

### "ERROR: No checkpoints found"
- Training may have crashed or been interrupted
- Check that `--ckpt_every` fits within `--iters`
  - If `--ckpt_every 25` but `--iters 20`, no checkpoints saved
  - Use `--ckpt_every 5` or `--iters 50` instead

### "⚠ Using latest checkpoint (best info not found)"
- Likely no evaluation was run during training
- Check `--eval_every` was less than total iters
- Solution: Re-run with `--eval_every 5` (evaluate every 5 iterations)

### "Best checkpoint was at iteration X but checkpoint_X not found"
- Checkpoints might have been rotated out (only 5 kept)
- Re-run training with larger `keep_checkpoints_num` in config
- Or accept the latest available checkpoint

### Multiple Seeds

If running ablation with multiple seeds, repeat the workflow for each:
```bash
# After training seed 0, 1, 2
for seed in 0 1 2; do
  python -m maritime_rl_pkg.maritime_rl.select_best_checkpoint \
      --run_dir "MARL_ppo_case2_XXXXXX/seed_$seed"
  echo "---"
done
```
