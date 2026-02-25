"""
Minimal RL Lib MAPPO training script for MultiShipParallelEnv

Notes: 
- adjust imports depending on RL version
- supports PettingZoo ParallelEnv via PettingZooEnv wrapper

"""

import argparse

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
    return p.parse_args()


def main():
    args = parse_args()

    # local imports so file can be imported without Ray installed
    from ray import tune
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.algorithms.callbacks import DefaultCallbacks

    from callbacks import CORALL_Colav_TrainingCallbacks
    from multi_agent_env_mappo import MultiShipParallelEnv
    

    def env_creator(config):
        case_number = config.get("case_number", args.case)
        return MultiShipParallelEnv(
            case_number=case_number, 
            dt=config.get("dt", 0.2),
            sim_time=config.get("sim_time", 300.0),
            seed=config.get("seed", args.seed)
        )

    # register PettingZoo parallel env with RLlib 
    tune.register_env("corall_mappo_env", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))


    # create temporary env instance to read spaces and agent ids 
    # --> probe env once to get spaces + agent_ids (so RLlib can build policy model with correct input/output sizes)
    tmp_env = env_creator({"case_number": args.case, "seed": args.seed})
    obs_space = tmp_env.observation_space(tmp_env.agents[0])
    act_space = tmp_env.action_space(tmp_env.agents[0])
    agent_ids = tmp_env.agents
    tmp_env.close() if hasattr(tmp_env, "close") else None

    def policy_mapping_fn(agent_id, *args, **kwargs):
        """
        Tell RLlib that all agents share one policy (indicate MAPPO type policy)
        """
       
        return "shared_policy"
    
    # Build MAPPO algorithm configuration using RLlib's config API based on docs available
    print("Creating PPOConfig...")
    base_config = PPOConfig()
    print(f"PPOConfig instance created: {type(base_config)}")
    
    # config = (
        # base_config
        #.environment(env="corall_mappo_env", env_config={"case_number": args.case, "seed": args.seed})
        #.framework("torch")
        #.rollouts(num_rollout_workers=args.num_workers, rollout_fragment_length=args.rollout_frag)
        #.training(
            #lr=args.lr, 
            #gamma=args.gamma,
            #train_batch_size=args.train_batch,

            # tune hyperparameters here as needed
            #clip_param=0.2, 
            #vf_clip_param=10.0, 
            #entropy_coeff=0.0, 
            #lambda_=0.95, 
            #num_sgd_iter=10,
            #sgd_minibatch_size=2048,
        #)
        #.multi_agent(
            #policies={"shared_policy": (None, obs_space, act_space, {})},
            #policy_mapping_fn=policy_mapping_fn, 
            #policies_to_train=["shared_policy"],
        #)
        #.debugging(seed=args.seed)
        #.callbacks(CORALL_Colav_TrainingCallbacks)
    #)

    config = PPO.get_default_config()

    # set environment
    config["env"] = "corall_mappo_env"
    config["env_config"] = {"case_number": args.case, "seed": args.seed}

    # set framework
    config["framework"] = "torch"

    # set rollout workers
    config["num_rollout_workers"] = args.num_workers
    config["rollout_fragment_length"] = args.rollout_frag

    # set training parameters
    config["lr"] = args.lr
    config["gamma"] = args.gamma
    config["train_batch_size"] = args.train_batch
    # optional PPO hyperparameters (tune as needed)
    config["clip_param"] = 0.2
    config["vf_clip_param"] = 10.0
    config["entropy_coeff"] = 0.0
    config["lambda"] = 0.95
    config["num_sgd_iter"] = 10
    config["sgd_minibatch_size"] = 2048 

    # multi-agent setup
    config["multiagent"] = {
        "policies": {"shared_policy": (None, obs_space, act_space, {})},
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["shared_policy"],
    }

    # callbacks
    config["callbacks"] = CORALL_Colav_TrainingCallbacks

    # seed
    config["seed"] = args.seed

    print(f"Config: {config}")

    algo = config.build()

    # training loop (
    ## collect rollouts and print desired metrics at each checkpoint)
    for i in range(args.iters):
        result = algo.train()

        if i % 5 == 0:
            print(f"Iter{i}:"
                  f"episode_reward_mean={result.get('episode_reward_mean')}", 
                  f"episode_len_mean={result.get('episode_len_mean')}" )
        # later also log, collision rate, success rate, avg max risk / min DCPA per episode
        
    ckpt = algo.save()
    print("Saved checkpoint:", ckpt)

if __name__ == "__main__":
    main()