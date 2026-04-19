"""
RL Lib PPO training script for MultiShipParallelEnv

Notes: 
- adjust imports depending on RL version
- supports PettingZoo ParallelEnv via PettingZooEnv wrapper

Example usage:
python -m maritime_rl_pkg.multi_agent_train_ppo --case 2 --iters 50 --num_workers 1 --rollout_frag 1000 --train_batch 4000 --seed 0

Example with different MLP sizes:
python -m maritime_rl_pkg.multi_agent_train_ppo --case 6 --iters 40 --mlp_hiddens 64 64 --seed 0
python -m maritime_rl_pkg.multi_agent_train_ppo --case 6 --iters 40 --mlp_hiddens 128 128 --seed 0
python -m maritime_rl_pkg.multi_agent_train_ppo --case 6 --iters 40 --mlp_hiddens 256 256 --seed 0

"""

import argparse
import time
from pathlib import Path
import csv
import shutil
import json

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plotting in headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch

def parse_args():
    """Make it easier to make case numbers and training outputs clear after trained
    
    outer loop parameters:
    --case: which CORALL Imazu case to train on 
    --iters: number of "train iteration" training library should run 
    --num_workers: how many rollout worker processes collect experience in parallel
    --rollout_frag: how many env steps each worker collects per 'fragment' before sending to the learner for training. Larger means less communication overhead but more stale data.
    --train_batch: how many total env steps aggregated into training batch per iteration
    --lr: learning rate for policy optimization
    --gamma: discount factor for future rewards
    --seed: random seed for reproducibility
    
    """
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, default=2, help="CORALL Imazu case_number")
    p.add_argument("--iters", type=int, default=200, help="Training iterations")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--rollout_frag", type=int, default=2000)
    p.add_argument("--train_batch", type=int, default=8000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--route_len_nmi", type=float, default=2.0, help="Route length in nautical miles (scaling)")
    p.add_argument("--sim_time", type=float, default=300.0, help="Max simulation time in seconds (scaling)")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--n_eval_episodes", type=int, default=3, help="Number of evaluation episodes to run at each evaluation checkpoint")
    p.add_argument("--ckpt_every", type=int, default=25, help="How often (in training iterations) to save checkpoints")
    p.add_argument("--mlp_hiddens", type=int, nargs="+", default=[64, 64], help="MLP hidden layer sizes (default 64,64 to reduce overfitting; try 128,128 for more capacity)")
    return p.parse_args()

def run_policy_evaluation(algo, env_creator, n_eval_episodes=5, use_stochastic=False):
    """
    Run evaluation rollouts with current PPO policy during training.
    If use_stochastic=False (default), uses deterministic argmax actions.
    If use_stochastic=True, samples from action distribution for better generalization testing.
    Returns average episode-level metrics 
    """
    eval_returns = []
    eval_lengths = []

    for ep in range(n_eval_episodes):
        # Use different seeds for eval episodes to test generalization
        eval_seed = 10000 + ep  # Deterministic but different from training seed (0)
        env = env_creator({"seed": eval_seed})
        obs, infos = env.reset()

        ep_return_by_agent = {agent: 0.0 for agent in env.agents}
        ep_len = 0
        
        # Get action space info from environment (PettingZoo)
        first_agent = env.agents[0]
        action_space = env.action_space(first_agent)
        action_space_shape = None
        if hasattr(action_space, 'nvec'):  # MultiDiscrete
            action_space_shape = list(action_space.nvec)
        elif hasattr(action_space, 'n'):  # Discrete
            action_space_shape = [action_space.n]
        elif hasattr(action_space, 'shape'):  # Box (continuous)
            action_space_shape = list(action_space.shape)

        while True: 
            actions = {}

            # Use new API stack: get RLModule and call forward_inference
            rl_module = algo.get_module("shared_policy")
            
            for agent_id, agent_obs in obs.items():
                # Convert observation to tensor batch format (add batch dimension)
                obs_tensor = torch.from_numpy(np.array([agent_obs])).float()
                
                # Inference with the new API - forward_inference returns action distribution inputs
                with torch.no_grad():
                    output = rl_module.forward_inference(batch={"obs": obs_tensor})
                    
                    # Extract action from RLModule output
                    # For PPO new API, forward_inference returns action_dist_inputs (logits)
                    if isinstance(output, dict) and "action_dist_inputs" in output:
                        # Extract logits from batch dimension: shape [1, num_logits]
                        logits = output["action_dist_inputs"][0].cpu().numpy()  # shape [num_logits]
                        
                        # For MultiDiscrete([7, 5]), action_dist_inputs concatenates logits for each component
                        if action_space_shape and len(action_space_shape) > 1:
                            # Multi-discrete action space
                            action = []
                            offset = 0
                            for num_categories in action_space_shape:
                                component_logits = logits[offset:offset + num_categories]
                                if use_stochastic:
                                    # Sample from distribution (tests generalization better)
                                    probs = np.exp(component_logits) / np.sum(np.exp(component_logits))
                                    action_component = np.random.choice(num_categories, p=probs)
                                else:
                                    # Deterministic argmax (best current policy)
                                    action_component = int(np.argmax(component_logits))
                                action.append(action_component)
                                offset += num_categories
                            action = np.array(action, dtype=np.int64)
                        else:
                            # Single discrete action
                            if use_stochastic:
                                probs = np.exp(logits) / np.sum(np.exp(logits))
                                action = np.array([np.random.choice(len(logits), p=probs)], dtype=np.int64)
                            else:
                                action = np.array([int(np.argmax(logits))], dtype=np.int64)
                    elif isinstance(output, dict):
                        print(f"[DEBUG] Unexpected output format. Keys: {list(output.keys())}")
                        raise KeyError(f"Cannot extract action from output keys: {list(output.keys())}")
                    else:
                        raise TypeError(f"Expected dict output, got {type(output)}")
                
                actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            ep_len += 1
            for agent_id, reward in rewards.items():
                ep_return_by_agent[agent_id] += float(reward)

            # continue until time limit (all agents uniformly truncated)
            if all(truncations.values()):
                break

        # mean return over agents for this eval episode
        eval_returns.append(float(np.mean(list(ep_return_by_agent.values()))))
        eval_lengths.append(ep_len)
        
        if hasattr(env, "close"):
            env.close()

    return {
        "eval_return_mean": float(np.mean(eval_returns)) if eval_returns else float("nan"),
        "eval_ep_length_mean": float(np.mean(eval_lengths)) if eval_lengths else float("nan"),
    }

def moving_average(values, window=10):
    """
    Compute moving average of a list of values with specified window size, ignoring NaNs. 
    
    Returns array of same length with NaN for positions where moving average is not defined.
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0: 
        return vals
    out = np.full_like(vals, np.nan, dtype=float)
    for i in range(len(vals)):
        lo = max(0, i-window+1)
        chunk = vals[lo:i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) > 0:
            out[i] = np.mean(chunk)
    return out 

def main():
    args = parse_args()
    print(f"[DEBUG] Parsed args: {args}")

    # Import time at function start to avoid scoping issues
    import time
    
    # Create an output folder for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"MARL_ppo_case{args.case}_{timestamp}") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logging] writing outputs to: {output_dir.resolve()}")

    # local imports so file can be imported without Ray installed
    import ray
    from ray import tune
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo import PPO

    from maritime_rl_pkg.env_multi_agent_ppo import MultiShipParallelEnv

    # explicitly initialize Ray to avoid hangs on Windows
    if not ray.is_initialized():
        try:
            ray.init(
                ignore_reinit_error=True, 
                _temp_dir=None, 
                include_dashboard=False,
            )
        except Exception as e:
            print(f"[ERROR] Ray initialization failed: {e}")
            print("[INFO] Attempting to shutdown any existing Ray instances and retry...")
            try:
                ray.shutdown()
            except:
                pass
            time.sleep(2)
            try:
                ray.init(
                    ignore_reinit_error=True, 
                    _temp_dir=None, 
                    include_dashboard=False,
                )
                print("[INFO] Ray re-initialized successfully")
            except Exception as e2:
                print(f"[FATAL] Ray initialization failed again: {e2}")
                raise

    def env_creator(config):
        case_number = config.get("case_number", args.case)
        env = MultiShipParallelEnv(
            case_number=case_number, 
            dt=config.get("dt", 0.5),
            sim_time=config.get("sim_time", args.sim_time),
            route_len_nmi=config.get("route_len_nmi", args.route_len_nmi),
            seed=config.get("seed", args.seed)
        )
        return env

    # register PettingZoo parallel env with RLlib 
    tune.register_env("corall_mappo_env", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))


    # create temporary env instance to read spaces and agent ids 
    # --> probe env once to get spaces + agent_ids (so RLlib can build policy model with correct input/output sizes)
    tmp_env = env_creator({"case_number": args.case, "seed": args.seed})
    obs_space = tmp_env.observation_space(tmp_env.agents[0])
    act_space = tmp_env.action_space(tmp_env.agents[0])
    # agent_ids = tmp_env.agents
    tmp_env.close() if hasattr(tmp_env, "close") else None

    def policy_mapping_fn(agent_id, *args, **kwargs):
        """
        Tell RLlib that all agents share one policy (indicate PPO type policy)
        """
        return "shared_policy"
    
    # Build PPO algorithm configuration using RLlib's config API based on docs
    # MLP Architecture Configuration
    mlp_hiddens = args.mlp_hiddens  # e.g., [128, 128] or [64, 64]
    mlp_activation = "tanh"
    
    print(f"[Model Config] MLP Architecture: {mlp_hiddens} with {mlp_activation} activation")
    print(f"[Model Config] Separate actor/critic networks (vf_share_layers=False)")
    print(f"[Model Config] Total parameters estimate: ~{np.prod(mlp_hiddens) * 2 // 1000}K weights per network")
    
    config = (
        PPOConfig()
        .environment(
            env="corall_mappo_env",
            env_config={
                "case_number": args.case,
                "seed": args.seed,
                "sim_time": args.sim_time,
                "route_len_nmi": args.route_len_nmi,
            }, 
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length=args.rollout_frag
        )
        .training(
            lr=args.lr,
            gamma=args.gamma,
            train_batch_size=args.train_batch,
            clip_param=0.2, 
            vf_clip_param=10.0,
            entropy_coeff=0.005,  # INCREASED from 0.0: Prevent collapse to deterministic policy, improve generalization
            lambda_=0.95, 
            num_epochs=10,
            minibatch_size=128,
        )
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  # policy class (None = use default)
                    obs_space,
                    act_space,
                    {  # Model config dict
                        "fcnet_hiddens": mlp_hiddens,
                        "fcnet_activation": mlp_activation,
                        "vf_share_layers": False,
                        "use_lstm": False,
                    }
                )
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"]
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
    )

    # instantiate PPO algorithm with config
    algo = PPO(config=config)

    # create subdirectory for checkpoints
    checkpoint_dir = (output_dir / "checkpoints").resolve()  # Use absolute path for new API stack
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_metrics.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "iteration", 
            "episode_return_mean", 
            "episode_len_mean",
            "eval_return_mean",
            "eval_ep_length_mean",
            "train_wall_time_s",
            "eval_wall_time_s",
            "approx_train_steps_per_sec",
            "num_workers"
        ])
    
    xs_train, ys_train = [], []
    xs_eval, ys_eval = [], []
    
    # Tracking for ablation study
    timing_metrics = []
    
    # Tracking for best checkpoint selection
    best_eval_return = float("-inf")
    best_checkpoint_data = None
    best_checkpoint_folder = output_dir / "best_checkpoint"

    for i in range(args.iters):
        # Time training step separately from evaluation
        train_t0 = time.perf_counter()
        result = algo.train()
        train_wall_time_s = time.perf_counter() - train_t0
        eval_wall_time_s = 0.0

        runner_stats = result.get("env_runners", {})
        rew = runner_stats.get("episode_return_mean", float("nan"))
        ep_len = runner_stats.get("episode_len_mean", float("nan"))
        approx_train_steps_per_sec = (args.train_batch / train_wall_time_s if train_wall_time_s > 0 else float("nan"))

        xs_train.append(i+1)
        ys_train.append(rew)

        # run evaluation periodically
        eval_return_mean = float("nan")
        eval_ep_length_mean = float("nan")
        
        if (i+1) % args.eval_every == 0:
            try:
                eval_t0 = time.perf_counter()
                # Run deterministic eval (best current policy)
                eval_results = run_policy_evaluation(
                    algo, 
                    env_creator, 
                    n_eval_episodes=max(10, args.n_eval_episodes),  # At least 10 episodes for better stats
                    use_stochastic=False
                )
                eval_wall_time_s = time.perf_counter() - eval_t0
                eval_return_mean = eval_results["eval_return_mean"]
                eval_ep_length_mean = eval_results["eval_ep_length_mean"]
                xs_eval.append(i+1)
                ys_eval.append(eval_return_mean)
                
                # Track best checkpoint by eval return
                if eval_return_mean > best_eval_return:
                    best_eval_return = eval_return_mean
                    best_checkpoint_data = {
                        "iteration": i+1,
                        "eval_return_mean": eval_return_mean,
                        "eval_ep_length_mean": eval_ep_length_mean,
                    }
                    print(f"  ✓ New best checkpoint at iteration {i+1}: eval_return={eval_return_mean:.2f}")
                    
                    # Save the CURRENT checkpoint first (not the stale one in checkpoint_dir)
                    try:
                        ckpt_path = algo.save(str(checkpoint_dir))
                        print(f"    → Checkpoint saved: {ckpt_path}")
                        
                        # Now copy the fresh checkpoint to best_checkpoint folder
                        if checkpoint_dir.exists():
                            # Remove old best_checkpoint if it exists
                            if best_checkpoint_folder.exists():
                                shutil.rmtree(best_checkpoint_folder)
                            # Copy the entire checkpoint directory to best_checkpoint
                            shutil.copytree(checkpoint_dir, best_checkpoint_folder)
                            print(f"    → Copied to best_checkpoint: {best_checkpoint_folder}")
                    except Exception as copy_err:
                        print(f"    [WARNING] Failed to save/copy best checkpoint: {copy_err}")
            except Exception as e:
                print(f"[WARNING] Evaluation failed at iteration {i+1}: {e}")

        # Store timing metrics for ablation analysis
        timing_metrics.append({
            "iteration": i+1,
            "train_wall_time_s": train_wall_time_s,
            "eval_wall_time_s": eval_wall_time_s,
            "approx_train_steps_per_sec": approx_train_steps_per_sec,
        })
        
        print(
            f"Iter {i+1}/{args.iters} | "
            f"train_return={rew:.2f} | "
            f"ep_len={ep_len:.2f} | "
            f"eval_return={eval_return_mean:.2f} | "
            f"train_wall_time={train_wall_time_s:.3f}s | "
            f"steps/sec={approx_train_steps_per_sec:.1f}"
        )

        with open(csv_path, mode="a", newline="") as csv_file:
            csv.writer(csv_file).writerow([
                i+1, 
                rew, 
                ep_len,
                eval_return_mean,
                eval_ep_length_mean,
                train_wall_time_s,
                eval_wall_time_s,
                approx_train_steps_per_sec,
                args.num_workers
            ])
        
        if (i+1) % args.ckpt_every == 0:
            ckpt_path = algo.save(str(checkpoint_dir))
            print(f"Checkpoint saved at iteration {i+1}: {ckpt_path}")
    
    final_ckpt_path = algo.save(str(checkpoint_dir))
    print(f"Training complete. Final checkpoint saved at: {final_ckpt_path}")

    # ========== Save best checkpoint info ==========
    best_ckpt_info_path = output_dir / "best_checkpoint_info.json"
    if best_checkpoint_data:
        # Save best checkpoint metadata
        with open(best_ckpt_info_path, "w") as f:
            json.dump(best_checkpoint_data, f, indent=2)
        print(f"\n[INFO] Best checkpoint was at iteration {best_checkpoint_data['iteration']}")
        print(f"       with eval_return_mean = {best_checkpoint_data['eval_return_mean']:.2f}")
        print(f"\n[INFO] Ready to evaluate:")
        print(f"       python -m maritime_rl_pkg.eval_trained_policy \\")
        print(f"           --checkpoint \"{best_checkpoint_folder.resolve()}\" \\")
        print(f"           --case {args.case} --episodes 40 --seed {args.seed}")
    else:
        print("\n[WARNING] No evaluation was performed; cannot identify best checkpoint")
        print("          Use --eval_every < --iters to enable evaluation during training")

    # ========== Parallelization / num_workers study ==========
    # Calculate aggregated timing statistics for ablation analysis
    # Initialize at module scope so it's available for plotting even if timing_metrics is empty
    mean_steps_per_sec = float("nan")
    mean_train_wall_time_s = float("nan")
    mean_eval_wall_time_s = float("nan")
    total_run_time_s = float("nan")
    
    if timing_metrics:
        train_times = [m["train_wall_time_s"] for m in timing_metrics]
        eval_times = [m["eval_wall_time_s"] for m in timing_metrics if m["eval_wall_time_s"] > 0]
        steps_per_sec = [m["approx_train_steps_per_sec"] for m in timing_metrics if np.isfinite(m["approx_train_steps_per_sec"])]
        
        mean_train_wall_time_s = float(np.mean(train_times)) if train_times else float("nan")
        mean_eval_wall_time_s = float(np.mean(eval_times)) if eval_times else 0.0
        mean_steps_per_sec = float(np.mean(steps_per_sec)) if steps_per_sec else float("nan")
        total_run_time_s = float(np.sum(train_times) + np.sum(eval_times))
        
        # Create ablation summary for this run
        ablation_summary = {
            "case": args.case,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "iters": args.iters,
            "train_batch_size": args.train_batch,
            "rollout_frag_len": args.rollout_frag,
            "mean_train_wall_time_s": mean_train_wall_time_s,
            "mean_eval_wall_time_s": mean_eval_wall_time_s,
            "mean_approx_train_steps_per_sec": mean_steps_per_sec,
            "total_run_time_s": total_run_time_s,
        }
        
        # Save ablation summary as JSON
        ablation_json_path = output_dir / "ablation_summary.json"
        with open(ablation_json_path, "w") as f:
            json.dump(ablation_summary, f, indent=2)
        
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        print(f"num_workers: {args.num_workers}")
        print(f"train_batch_size: {args.train_batch}")
        print(f"rollout_frag_len: {args.rollout_frag}")
        print(f"mean_train_wall_time_s: {mean_train_wall_time_s:.3f}")
        print(f"mean_eval_wall_time_s: {mean_eval_wall_time_s:.3f}")
        print(f"mean_steps_per_sec: {mean_steps_per_sec:.1f}")
        print(f"total_run_time_s: {total_run_time_s:.1f}")
        print("="*60 + "\n")
        print(f"Ablation summary saved to: {ablation_json_path}")

    # Final training plot
    try:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(xs_train, ys_train, label="train_return", color="blue", alpha=0.5)
        ax1.plot(xs_train, moving_average(ys_train, window=10), label="train_return (moving avg)", color="darkblue", linestyle="--")
        
        if len(xs_eval) > 0:
            ax1.plot(xs_eval, ys_eval, marker="o", label="eval_return", color="orange", linestyle="-", markersize=6)
        
        ax1.set_title(f"PPO training vs evaluation return | case {args.case} | seed {args.seed} | workers {args.num_workers}")
        ax1.set_xlabel("Training Iteration")
        ax1.set_ylabel("Mean Episode Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(output_dir / "training_return.png", dpi=300, format='png')
        fig1.savefig(output_dir / "training_return.pdf", format='pdf')
        plt.close(fig1)
        print(f"✓ Training return plot saved to: {output_dir / 'training_return.png'} and .pdf")
    except Exception as e:
        print(f"[WARNING] Failed to save training_return plot: {e}")
    
    # Plot timing trajectory for this run
    try:
        iterations = [m["iteration"] for m in timing_metrics]
        steps_per_sec_trajectory = [m["approx_train_steps_per_sec"] for m in timing_metrics]
        
        if len(iterations) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(iterations, steps_per_sec_trajectory, marker="o", label="approximate steps/sec", color="green", alpha=0.7)
           
            ax2.set_title(f"Training throughput over iterations | num_workers={args.num_workers}")
            ax2.set_xlabel("Training Iteration")
            ax2.set_ylabel("Approximate Steps/Second")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(output_dir / "throughput_trajectory.png", dpi=300, format='png')
            fig2.savefig(output_dir / "throughput_trajectory.pdf", format='pdf')
            plt.close(fig2)
            print(f"✓ Throughput plot saved to: {output_dir / 'throughput_trajectory.png'} and .pdf")
        else:
            print(f"[WARNING] No timing metrics available; skipping throughput plot")
    except Exception as e:
        print(f"[WARNING] Failed to save throughput plot: {e}")

    # Clean up Ray resources
    import ray
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    main()