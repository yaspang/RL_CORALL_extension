import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
import numpy as np

from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy

# ✅ import your actual env
from multi_agent_env_mappo import MultiShipParallelEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(case_number: int, dt: float, sim_time: float, seed: int):
    # AsyncPettingZooVecEnv expects a zero-arg callable
    def _thunk():
        return MultiShipParallelEnv(
            case_number=case_number,
            dt=dt,
            sim_time=sim_time,
            seed=seed,
        )
    return _thunk


def eval_trained_agent(trained_pop, case_number: int, n_eval_episodes: int = 20, dt: float = 0.2, sim_time: float = 60.0):
    """Evaluate the trained agent over multiple episodes and return success/collision rates."""
    eval_env = MultiShipParallelEnv(case_number=case_number, dt=dt, sim_time=sim_time, seed=123)

    agent = trained_pop[0]

    success_count = 0
    collision_count = 0
    steps_list = []
    returns_list = []

    for ep in range(n_eval_episodes):
        obs, infos = eval_env.reset()
        steps = 0
        ep_return = 0.0

        ep_success = False
        ep_collision = False

        while True:
            actions = {}
            for a in eval_env.agents:
                o = np.asarray(obs[a], dtype=np.float32)

                # AgileRL supports routing by agent_id; keep your try/fallback pattern
                try:
                    act = agent.get_action(o, agent_id=a)
                except TypeError:
                    act = agent.get_action(o)

                # Ensure MultiDiscrete action shape (2,) int64
                act = np.asarray(act).squeeze().astype(np.int64)
                if act.shape == ():  # if algo ever returns scalar, fail loudly
                    raise RuntimeError(f"Got scalar action for {a}; expected shape (2,) for MultiDiscrete.")
                actions[a] = act

            obs, rewards, terminations, truncations, infos = eval_env.step(actions)

            steps += 1
            ep_return += float(sum(rewards.values()))  # team return

            for a in eval_env.agents:
                if infos.get(a, {}).get("success", False):
                    ep_success = True
                if infos.get(a, {}).get("collision", False):
                    ep_collision = True

            if any(terminations.values()) or any(truncations.values()):
                break

        steps_list.append(steps)
        returns_list.append(ep_return)
        success_count += int(ep_success)
        collision_count += int(ep_collision)

    print("\n=== Eval Results ===")
    print(f"Episodes:        {n_eval_episodes}")
    print(f"Success rate:    {success_count/n_eval_episodes:.2%}")
    print(f"Collision rate:  {collision_count/n_eval_episodes:.2%}")
    print(f"Avg steps/ep:    {sum(steps_list)/len(steps_list):.1f}")
    print(f"Avg return/ep:   {sum(returns_list)/len(returns_list):.2f}")  # ✅ fixed typo


def train_ippo_corall(
    case_number: int = 1,
    num_envs: int = 4,
    dt: float = 0.2,
    sim_time: float = 60.0,
    max_steps: int = 300_000,
):
    env = AsyncPettingZooVecEnv([
        make_env(case_number=case_number, dt=dt, sim_time=sim_time, seed=100 + i)
        for i in range(num_envs)
    ])

    print("1) Resetting vec env...", flush=True)
    env.reset()
    print("2) Vec env reset OK.", flush=True)

    # quick sanity step like your original script
    print("SANITY: stepping vec env 10 times with random actions...", flush=True)
    for _ in range(10):
        actions = {
            a: [env.single_action_space(a).sample() for _ in range(num_envs)]
            for a in env.agents
        }
        env.step(actions)
    print("SANITY: vec env step OK", flush=True)

    agent_ids = list(env.agents)

    obs_spaces = [env.single_observation_space(a) for a in agent_ids]
    act_spaces = [env.single_action_space(a) for a in agent_ids]

    print("agents:", agent_ids, flush=True)
    print("obs space example:", obs_spaces[0], flush=True)
    print("act space example:", act_spaces[0], flush=True)

    BASE_NET = {
        "encoder_config": {"hidden_size": [64, 64], "activation": "ReLU"},
        "head_config": {"hidden_size": [64], "activation": "ReLU"},
    }
    NET_CONFIG = {aid: BASE_NET for aid in agent_ids}

    INIT_HP = {
        "AGENT_IDS": agent_ids,

        # IPPO/PPO hyperparams (keep close to your originals)
        "lr": 3e-4,
        "batch_size": 256,
        "learn_step": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,

        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 4,
    }

    pop = create_population(
        algo="IPPO",
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        observation_space=obs_spaces,
        action_space=act_spaces,
        hp_config=None,
        population_size=1,
        num_envs=num_envs,
        device=str(device),
    )

    trained_pop, pop_fitnesses = train_multi_agent_on_policy(
        env,
        env_name="MultiShipParallelEnv_CORALL",
        algo="IPPO",
        pop=pop,
        sum_scores=True,
        INIT_HP=INIT_HP,
        MUT_P=None,
        max_steps=max_steps,
        evo_steps=1_000_000,
        eval_steps=None,
        eval_loop=1,
        target=None,
        tournament=None,
        mutation=None,
        wb=False,
        accelerator=None,
    )

    eval_trained_agent(trained_pop, case_number=case_number, n_eval_episodes=20, dt=dt, sim_time=sim_time)


if __name__ == "__main__":
    train_ippo_corall(
        case_number=1,   # pick an Imazu case id that exists in CORALL
        num_envs=4,
        dt=0.2,
        sim_time=60.0,
        max_steps=300_000,
    )