import argparse
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from env import CORALLEnv, ImazuEvalEnv

MAX_STEPS = 5000
STEP_SIZE = 60
STEPS_PER_FRAME = 4


def run(
    model_path: str,
    vec_normalize_path: str,
    render_mode: str | None,
    case: int | None,
    render_every: int = 1,
    env_type: str = "imazu",
    n_episodes: int = 1,
    round_robin: bool = False,
):
    def make_env():
        if env_type == "imazu":
            return Monitor(TimeLimit(
                ImazuEvalEnv(case=case, render_mode=render_mode, step_size=STEP_SIZE, fps=30, render_every=render_every, max_intruders=20, round_robin=round_robin),
                max_episode_steps=MAX_STEPS,
            ))
        else:
            return Monitor(TimeLimit(
                CORALLEnv(render_mode=render_mode, step_size=STEP_SIZE, fps=30, render_every=render_every, max_intruders=20, round_robin=round_robin),
                max_episode_steps=MAX_STEPS,
            ))

    venv = DummyVecEnv([make_env])
    if vec_normalize_path:
        env = VecNormalize.load(vec_normalize_path, venv)
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, training=False)
    env.training = False
    env.norm_reward = False

    model = SAC.load(model_path, env=env)

    if env_type == "imazu":
        n_episodes = 1 if case is not None else ImazuEvalEnv.N_CASES

    results = []
    inner_env = env.envs[0].env.env

    def episode_iter():
        if env_type == "random":
            ep = 0
            while True:
                yield ep
                ep += 1
        else:
            yield from range(n_episodes)

    try:
        for ep in episode_iter():
            obs = env.reset()
            label = getattr(inner_env, "current_case", ep + 1)
            total_reward = 0.0

            while True:
                if render_mode and inner_env._renderer is not None:
                    inner_env._renderer.events = []

                terminated, truncated = False, False

                for _ in range(STEPS_PER_FRAME if render_mode else 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += float(reward[0])
                    terminated = done[0] and not info[0].get("TimeLimit.truncated", False)
                    truncated = done[0] and info[0].get("TimeLimit.truncated", False)
                    if done[0]:
                        break
                    if render_mode and inner_env._renderer is not None and inner_env._renderer.interrupted:
                        break

                if render_mode and inner_env._renderer is not None:
                    events = inner_env._renderer.events
                    if any(e.type == pygame.QUIT for e in events):
                        break
                    if any(e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE for e in events):
                        terminated = True

                if terminated or truncated:
                    results.append((label, total_reward))
                    break
    except KeyboardInterrupt:
        pass

    env.close()

    if not results:
        print("No completed episodes.")
        return

    print(f"\n{'Episode':>8}  {'Reward':>10}")
    print("-" * 22)
    for label, reward in results:
        print(f"{label:>8}  {reward:>10.1f}")
    print("-" * 22)
    print(f"{'Mean':>8}  {np.mean([r for _, r in results]):>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to saved model (e.g. models/corall_final)")
    parser.add_argument("--vec-normalize", default=None, help="Path to VecNormalize stats")
    parser.add_argument("--render-mode", choices=["human", "plot"], default=None)
    parser.add_argument("--env", choices=["imazu", "random"], default="imazu", help="Environment type (default: imazu)")
    parser.add_argument("--case", type=int, default=None, help="Imazu: run a single case (1-23)")
    parser.add_argument("--episodes", type=int, default=1, help="Random env: number of episodes to run (default: 1)")
    parser.add_argument("--render-every", type=int, default=1, help="Render every Nth physics sub-step (default: 1)")
    parser.add_argument("--round-robin", action="store_true", help="Imazu: agent controls all vessels in round-robin")
    args = parser.parse_args()

    run(
        args.model,
        args.vec_normalize,
        render_mode=args.render_mode,
        case=args.case,
        render_every=args.render_every,
        env_type=args.env,
        n_episodes=args.episodes,
        round_robin=args.round_robin,
    )
