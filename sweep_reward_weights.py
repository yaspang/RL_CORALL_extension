"""
Sweep reward weights to find optimal balance between goal-reaching and collision avoidance.
Tests: waypoint multiplier, risk penalty, and success bonus combinations.
"""
import json
import os
import subprocess
from datetime import datetime

# Configuration
CASE = 6
ITERS = 20  # Short training run for testing
EVAL_EVERY = 20
EVAL_EPISODES = 10
SEEDS = [0]

# Reward weight combinations to test
WEIGHT_COMBINATIONS = [
    # Format: (w_along_mult, w_risk, w_success, label)
    (2.0, -1.0, 15.0, "ORIGINAL"),
    (5.0, -0.5, 20.0, "MILD_INCREASE"),
    (10.0, -0.3, 25.0, "MODERATE_INCREASE"),
    (15.0, -0.2, 25.0, "CURRENT"),
    (8.0, -0.4, 20.0, "BALANCED_1"),
    (6.0, -0.6, 18.0, "BALANCED_2"),
    (12.0, -0.25, 22.0, "AGGRESSIVE_SAFE"),
    (3.0, -1.5, 20.0, "RISK_AVERSE"),
]

def read_reward_weights(env_file):
    """Extract current reward weights from env file."""
    with open(env_file, 'r') as f:
        content = f.read()
    return content

def set_reward_weights(env_file, w_along_mult, w_risk, w_success):
    """Update reward weights in env file."""
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Find and replace each weight
    # w_along *= X (around line 856)
    import re
    content = re.sub(
        r'w_along \*= [\d.]+',
        f'w_along *= {w_along_mult}',
        content
    )
    # w_risk = X (around line 883)
    content = re.sub(
        r'w_risk = -[\d.]+',
        f'w_risk = {w_risk}',
        content
    )
    # w_success = X (around line 995)
    content = re.sub(
        r'w_success = [\d.]+',
        f'w_success = {w_success}',
        content
    )
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print(f"Updated weights: w_along*={w_along_mult}, w_risk={w_risk}, w_success={w_success}")

def run_training(case, seed, w_along_mult, w_risk, w_success, label):
    """Run training with specific reward weights."""
    output_dir = f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = os.path.join(output_dir, f"{label}_seed{seed}.json")
    
    cmd = [
        "python", "-m", "maritime_rl_pkg.train_multi_agent_ppo",
        "--case", str(case),
        "--iters", str(ITERS),
        "--num_workers", "2",
        "--rollout_frag", "500",
        "--train_batch", "2000",
        "--lr", "3e-4",
        "--gamma", "0.99",
        "--seed", str(seed),
        "--route_len_nmi", "2.0",
        "--sim_time", "490.0",
        "--eval_every", str(EVAL_EVERY),
        "--n_eval_episodes", str(EVAL_EPISODES),
        "--ckpt_every", str(ITERS)
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"w_along*={w_along_mult}, w_risk={w_risk}, w_success={w_success}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Extract final eval_return from output
        lines = result.stdout.split('\n')
        final_eval = None
        for line in reversed(lines):
            if 'eval_return=' in line:
                parts = line.split('eval_return=')
                if len(parts) > 1:
                    val_str = parts[1].split()[0]
                    try:
                        final_eval = float(val_str)
                        break
                    except:
                        pass
        
        results = {
            "label": label,
            "w_along_mult": w_along_mult,
            "w_risk": w_risk,
            "w_success": w_success,
            "final_eval_return": final_eval,
            "success": result.returncode == 0,
            "output_sample": '\n'.join(lines[-20:])
        }
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Completed. Final eval_return: {final_eval}")
        return results
    
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 3600s")
        return {"label": label, "error": "timeout"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"label": label, "error": str(e)}

def main():
    env_file = "maritime_rl_pkg/env_multi_agent_ppo.py"
    original_content = read_reward_weights(env_file)
    
    all_results = []
    
    try:
        for w_along, w_risk, w_success, label in WEIGHT_COMBINATIONS:
            for seed in SEEDS:
                # Update weights
                set_reward_weights(env_file, w_along, w_risk, w_success)
                
                # Run training
                result = run_training(CASE, seed, w_along, w_risk, w_success, label)
                all_results.append(result)
    
    finally:
        # Restore original
        with open(env_file, 'w') as f:
            f.write(original_content)
        print("\n✓ Restored original reward weights")
    
    # Summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}\n")
    for r in all_results:
        if "error" not in r:
            print(f"{r['label']:20} | w_along={r['w_along_mult']:5.1f} | w_risk={r['w_risk']:6.2f} | w_success={r['w_success']:5.1f} | eval_return={r.get('final_eval_return', 'N/A')}")

if __name__ == "__main__":
    main()
