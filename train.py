import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import TimeLimit

from env import CORALLEnv
from policy import TargetAttentionExtractor, MLPExtractor


def make_train_env(step_size, max_steps, max_intruders, round_robin):
    return Monitor(TimeLimit(
        CORALLEnv(step_size=step_size, max_intruders=max_intruders, round_robin=round_robin),
        max_episode_steps=max_steps,
    ))


def make_eval_env(step_size, max_steps, max_intruders, round_robin):
    return Monitor(TimeLimit(
        CORALLEnv(step_size=step_size, max_intruders=max_intruders, round_robin=round_robin),
        max_episode_steps=max_steps,
    ))


def main(args):
    train_env = VecNormalize(
        make_vec_env(
            lambda: make_train_env(args.step_size, args.max_steps, args.max_intruders, args.round_robin),
            n_envs=args.n_envs,
        ),
        norm_obs=True,
        norm_reward=True,
    )

    eval_env = VecNormalize(
        DummyVecEnv([lambda: make_eval_env(args.step_size, args.max_steps, args.max_intruders, args.round_robin)]),
        norm_obs=True,
        norm_reward=True,
        training=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.save_dir}/best",
        log_path="./logs/eval",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=f"{args.save_dir}/checkpoints",
        name_prefix=args.run_name,
    )

    if args.policy == "mlp":
        extractor_class = MLPExtractor
        extractor_kwargs = dict(
            hidden_sizes=args.mlp_hidden,
            features_dim=args.features_dim,
        )
    else:
        extractor_class = TargetAttentionExtractor
        extractor_kwargs = dict(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            features_dim=args.features_dim,
        )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        buffer_size=2000000,
        tensorboard_log="./logs/tb",
        learning_starts=args.learning_starts,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        policy_kwargs=dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=extractor_kwargs,
            net_arch=args.net_arch,
        ),
        target_entropy=args.target_entropy,
        device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([eval_callback, checkpoint_callback]),
    )
    model.save(f"{args.save_dir}/{args.run_name}_final")
    train_env.save(f"{args.save_dir}/{args.run_name}_vec_normalize.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--step-size",    type=int,   default=60)
    parser.add_argument("--max-steps",    type=int,   default=200)
    parser.add_argument("--max-intruders", type=int,   default=20)
    parser.add_argument("--round-robin",  action="store_true", default=False)
    parser.add_argument("--n-envs",       type=int,   default=16)

    # Training
    parser.add_argument("--total-timesteps", type=int,   default=16_000_000)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--batch-size",      type=int,   default=512)
    parser.add_argument("--learning-starts", type=int,   default=10_000)
    parser.add_argument("--eval-freq",       type=int,   default=10_000)
    parser.add_argument("--eval-episodes",   type=int,   default=100)

    # Policy
    parser.add_argument("--policy",       type=str,   default="mlp", choices=["attn", "mlp"])
    parser.add_argument("--embed-dim",    type=int,   default=128)
    parser.add_argument("--num-heads",    type=int,   default=4)
    parser.add_argument("--features-dim", type=int,   default=256)
    parser.add_argument("--mlp-hidden",   type=int,   nargs="+", default=[256])
    parser.add_argument("--net-arch",     type=int,   nargs="+", default=[256])
    parser.add_argument("--target-entropy", type=lambda x: float(x) if x != "auto" else x, default="auto")
    parser.add_argument("--device",        type=str,   default="cpu")

    # Output
    parser.add_argument("--run-name",  default="attn")
    parser.add_argument("--save-dir",  default="./models")

    main(parser.parse_args())
