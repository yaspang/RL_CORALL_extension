
# import libraries
from __future__ import annotations
from datetime import datetime

import numpy as np
import cv2
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList


from .single_agent_ppo_env import CORALL_ReactiveAvoidanceGymEnv

import os
import io

from .path_setup import ensure_paths
ensure_paths()

from visualization.animate import animate_step_dense
from visualization.save_animation import create_video


def capture_frame_rgba(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4).copy()


def rollout_fixed_action(env, fixed_action: float, max_steps: int = 1000):
    """
    Roll out one episode using fixed scalar action (in [-1, 1])
    Returns: total_reward, steps, terminated, truncated, last_info
    """

    obs, info = env.reset(seed=0)
    total_r = 0
    terminated = False
    truncated = False
    last_info = {}

    for t in range(max_steps):
        action = np.array([fixed_action], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        last_info = info
        if terminated or truncated:
            return total_r, t+1, terminated, truncated, last_info
        
    return total_r, max_steps, terminated, truncated, last_info


def rollout_policy_make_video_fixed_camera(
    model,
    env,
    video_path=None,
    max_steps=None,
    deterministic=True,
    record_every=2, 
    dpi=120,
    figsize=(7, 7),
    pad_x=5, 
    pad_y=10,
):
    """
        
    Roll out a trained policy and save a video without storing frames in RAM. 
    All plotting units in nautical miles (nmi) to match CORALL interface. 
    env.X state is in meters for x, y

    - if max_steps is None, uses env.max_steps (if defined) or 3000 as default
    - record_every controls frame downsampling
    - dpi controls the resolution of the output video
    """

    obs, info = env.reset(seed=42)

    # autonomatically generate video name if not provided
    if video_path is None: 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        video_path = f"ppo_rollout_case{getattr(env, 'case_number', 'X')}_{timestamp}.avi"

    # determine rollout horizon
    if max_steps is None:
        max_steps = int(getattr(env, "max_steps", 3000) or 3000)

    ep_ret = 0.0

    # compute fixed bounds based on initial geometry (nmi units for plotting)
    x_own0 = float(env.X[0]) / 1852.0
    y_own0 = float(env.X[1]) / 1852.0
    x_obs0 = np.asarray(env.Xob, dtype=float) / 1852.0
    y_obs0 = np.asarray(env.Yob, dtype=float) / 1852.0
    x_wpt = np.asarray(env.Xwpt, dtype=float)
    y_wpt = np.asarray(env.Ywpt, dtype=float)

    # initial ownship position
    xs0 = np.concatenate([np.array([x_own0]), x_obs0, x_wpt])
    ys0 = np.concatenate([np.array([y_own0]), y_obs0, y_wpt])
    
    XMIN, XMAX = xs0.min() - pad_x, xs0.max() + pad_x
    YMIN, YMAX = ys0.min() - pad_y, ys0.max() + pad_y

    # output fps derived from env time step 
    fps_out = int(round(1.0 / float(env.dt)))
    writer = None 

    x_hist, y_hist = [], []

    ## guided b/c had trouble with rendering 
     # codec fallback: try MJPG->XVID (avi), then MP4V (mp4)
     # note: MP4V may not be available in some OpenCV builds, and MJPG may produce larger files but is more widely supported.
     # we default to .avi extension for better compatibility, but allow .mp4 if specified in video_path
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".mp4":
        fourcc_list = [cv2.VideoWriter_fourcc(*"mp4v")]
    else:
        fourcc_list = [cv2.VideoWriter_fourcc(*"MJPG"), cv2.VideoWriter_fourcc(*"XVID")]


    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    try:
        t = 0  # initialize t before loop
        terminated = False
        truncated = False
        info = {}
        
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)

            # update ownship position history (nmi)
            x_hist.append(float(env.X[0]) / 1852)
            y_hist.append(float(env.X[1]) / 1852)

            # only render / capture every N steps (saves time + file size)
            if (t % record_every) == 0:
                ax.clear()
                ax.set_title(f"PPO rollout (t={t})  return={ep_ret:.2f}")

                # draw ships / obstacles first 
                ## ownship sizes 
                LOA_own = 0.03   # ~55 m if 0.03 nmi
                BOL_own = 0.006  # nmi
                CPA_own = 0.02   # nmi
                
                # all obstacles (nmi)
                Xob = np.asarray(env.Xob, dtype=float) / 1852
                Yob = np.asarray(env.Yob, dtype=float) / 1852
                psiob = np.asarray(env.psiob, dtype=float)
                Vob = np.asarray(env.Vob, dtype=float)
                LOA_ob = np.ones(len(env.Xob)) * 0.03  # nmi
                BOL_ob = np.ones(len(env.Xob)) * 0.006 # nmi 
                CPA_ob = np.ones(len(env.Xob)) * 0.15  # nmi 

                Risk = info.get("risk", np.zeros(len(env.Xob), dtype=float))

                # --- Draw one step (this may internally call plt.axis('equal') and autoscale)
                animate_step_dense(
                    x=float(env.X[0]) / 1852.0,
                    y=float(env.X[1]) / 1852.0,
                    psi=float(env.X[2]),
                    LOA_own=LOA_own,
                    BOL_own=BOL_own,
                    CPA_own=CPA_own,
                    Xob=Xob,
                    Yob=Yob,
                    psiob=psiob,
                    LOA_ob=LOA_ob,
                    BOL_ob=BOL_ob,
                    CPA_ob=CPA_ob,
                    Risk=Risk,
                    Vob=Vob,
                    step=t,
                    ax=ax,
                )

                # fixed camera rendering 
                ax.set_autoscale_on(False)
                ax.set_xlim(XMIN, XMAX)
                ax.set_ylim(YMIN, YMAX)
                ax.set_aspect("equal", adjustable="box")

                # final overlays (waypoints already in nmi)
                ax.plot(env.Xwpt, env.Ywpt, "k--", linewidth=2, zorder=50)
                ax.scatter(env.Xwpt, env.Ywpt, marker='o')
                ax.text(env.Xwpt[0], env.Ywpt[0], 'start')
                ax.text(env.Xwpt[-1], env.Ywpt[-1], 'goal')
                ax.plot(x_hist, y_hist, linewidth=1.5, zorder=50)
                # ax.plot([x_hist[-1]], [y_hist[-1]], marker="o", markersize=4, zorder=51)  # current point

                # captue rendered frame to array 
                #fig.canvas.draw()
                # w, h = fig.canvas.get_width_height()
                # buf = np.asarray(fig.canvas.tostring_rgba(), dtype=np.uint8).reshape(h, w, 4)
                # frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR

                # Capture rendered frame to array
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi)
                buf.seek(0)
                
                # Read PNG buffer and decode
                png_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                frame_bgr = cv2.imdecode(png_data, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise RuntimeError("Failed to decode PNG frame buffer")


                # capture rendered frame
                # frame_rgba = capture_frame_rgba(fig)
                # frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGB2BGR)

                # init writer once the framesize is known
                if writer is None:
                    H, W = frame_bgr.shape[:2]

                    for fourcc in fourcc_list:
                        writer = cv2.VideoWriter(video_path, fourcc, fps_out, (W, H))
                        if writer.isOpened():
                            break
                        writer.release()
                        writer = None
                    if writer is None:
                            raise RuntimeError(f"VideoWriter failed to open with codecs {fourcc_list} for path: {video_path}")
        
                writer.write(frame_bgr)

                if t % 200 == 0:
                    x, y, psi, r, b, u = env.X
                    print(f"[rollout] step={t:4d}  action={action}  x={x:8.2f} y={y:8.2f} psi={psi:7.3f}  u={u:6.3f} r={r:7.3f}")

                if terminated or truncated:
                    break
    
    finally:
        if writer is not None: 
            writer.release()
        plt.close(fig)

    return ep_ret, (t + 1), info

class RewardLoggingCallback(BaseCallback):
    """
    Logs training episodic returns from Monitor during learning. 

    Saves: 
        - raw episodic returns
        - timestep at episode end
        - rolling mean episodic return
    """

    def __init__(self, rolling_window: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.rolling_window = rolling_window
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_end_steps = []
        self.rolling_mean_returns = []
    
    def on_step(self) -> bool:
        # in SB3, Monitor injects "episode" into info when an episode ends 
        # check if episode ended and log return
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            
            if ep is not None:
                ep_ret = float(ep["r"])
                ep_len = int(ep["l"])
                
                self.episode_returns.append(ep_ret)
                self.episode_lengths.append(ep_len)
                self.episode_end_steps.append(self.num_timesteps)

                start = max(0, len(self.episode_returns) - self.rolling_window)
                rolling_mean = float(np.mean(self.episode_returns[start:]))
                self.rolling_mean_returns.append(rolling_mean)

                # log to terminal / tensorboard
                self.logger.record("custom/episode_return", ep_ret)
                self.logger.record("custom/episode_length", ep_len)
                self.logger.record("custom/rolling_mean_return", rolling_mean)

                if self.verbose > 0: 
                    print(
                        f"  [train] step={self.num_timesteps} "
                        f"ep_ret={ep_ret:.2f} ep_len={ep_len} "
                        f"mean_return={rolling_mean:.2f}"
                    )
                    )
                    
        return True

def moving_average(x, window: int):
    if len(x) == 0:
        return np.array([])
    window = max(1, int(window))
    if len(x) < window:
        return np.array([np.mean(x)] * len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def save_training_plots(log_dir: str, reward_callback: RewardLoggingCallback):
    log_dir = Path(log_dir)

    # --- Plot 1: raw episodic return + rolling mean from callback
    x = np.array(reward_callback.episode_end_steps, dtype=float)
    y = np.array(reward_callback.episode_returns, dtype=float)
    y_roll = np.array(reward_callback.rolling_mean_returns, dtype=float)

    if len(x) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, alpha=0.35, label="episode return")
        plt.plot(x, y_roll, linewidth=2, label=f"rolling mean ({reward_callback.rolling_window})")
        plt.xlabel("training timesteps")
        plt.ylabel("episodic return")
        plt.title("Training episodic return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "training_episode_return.png", dpi=300, format='png')
        plt.close()

    # --- Plot 2: Monitor-based reward history
    try:
        results = load_results(str(log_dir))
        x_m, y_m = ts2xy(results, "timesteps")

        plt.figure(figsize=(10, 5))
        plt.plot(x_m, y_m, alpha=0.35, label="episode return (Monitor)")
        if len(y_m) >= 10:
            y_m_smooth = moving_average(y_m, window=10)
            x_m_smooth = x_m[len(x_m) - len(y_m_smooth):]
            plt.plot(x_m_smooth, y_m_smooth, linewidth=2, label="moving avg (10)")
        plt.xlabel("training timesteps")
        plt.ylabel("episodic return")
        plt.title("Monitor reward history")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_dir / "monitor_episode_return.png", dpi=300, format='png')
        plt.close()
    except Exception as e:
        print(f"[warn] could not load monitor results for plotting: {e}")

    # --- Save numeric arrays too
    np.savez(
        log_dir / "training_reward_data.npz",
        episode_end_steps=np.array(reward_callback.episode_end_steps),
        episode_returns=np.array(reward_callback.episode_returns),
        episode_lengths=np.array(reward_callback.episode_lengths),
        rolling_mean_returns=np.array(reward_callback.rolling_mean_returns),
    )

def main():

    # --- 1) determinstic "does action matter?" test

    env = CORALL_ReactiveAvoidanceGymEnv(case_number=2, dt=0.2, K_obstacles=1, max_steps_cap=20000)
    ret, steps, term, trunc, info = rollout_fixed_action(env, fixed_action=0, max_steps=5000)
        
    print(
            f"[Case 2] return={ret:.2f}, steps={steps}, trunc={trunc}, "
            f"risk_max={info.get('risk_max', float('nan')):.3f}, "
            f"collision={info.get('collision')}, goal={info.get('reached_goal')}"
    )


    # --- 2) PPO training with logging + periodic evaluation
    # use DummyVecEnv for single-process vectorized environment

    run_name = datetime.now().strftime("ppo_corall_baby_case2_%Y%m%d_%H%M%S")
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        env = CORALL_ReactiveAvoidanceGymEnv(
                    case_number=2, 
                    dt=0.2, 
                    K_obstacles=1, 
                    max_steps_cap=20000
        )
        return Monitor(env, filename=str(log_dir / "monitor.csv"))
    
    vec_env = DummyVecEnv([make_env])

    # separate eval environment
    eval_env = DummyVecEnv([
        lambda: Monitor(
            CORALL_ReactiveAvoidanceGymEnv(
                case_number=2, 
                dt=0.2,
                K_obstacles=1,
                max_steps_cap=20000

            ), 
            filename=str(log_dir / "eval_monitor.csv")
        )
    ])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1, 
        n_steps=1024, 
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        tensorboard_log='./tb_corall_babyppo', 
        device='auto'
    )

    reward_callback = RewardLoggingCallback(rolling_window=20, verbose=1)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_logs"),
        eval_freq = 10_000, # adjust as desired
        n_eval_episodes=5,
        deterministic=True,
        render=False, 
        verbose=1
    )

    callback = CallbackList([reward_callback, eval_callback])

    # train over timesteps
    model.learn(total_timesteps=1_000_000, callback=callback, progress_bar=True)

    # save plots + numeric reward arrays
    save_training_plots(log_dir, reward_callback)

 
    # --- 3) evaluate trained policy
    final_eval_env = CORALL_ReactiveAvoidanceGymEnv(
                    case_number=2, 
                    dt=0.2, 
                    K_obstacles=1,  # Use fixed value (matches training environment)
                    max_steps_cap=20000
            )
    
    mean_reward, std_reward = evaluate_policy(
                    model, 
                    final_eval_env, 
                    n_eval_episodes=10, 
                    deterministic=True
                )
    print(f"\n[PPO eval] mean_return={mean_reward:.2f} +/- {std_reward:.2f} over 10 episodes")

    # 4) rollout / visualize trained policy

    # Run a rollout that records video/frames using the trained policy
    # fps = int(round(1.0 / final_eval_env.dt))
    ret, steps, info = rollout_policy_make_video_fixed_camera(
                    model, 
                    final_eval_env, 
                    video_path=None, 
                    max_steps=None, 
                    deterministic=True,
                    record_every=5, 
                    dpi=150, 
                    figsize=(7, 7))

    print(f"[PPO rollout] return={ret:.2f}, steps={steps}, done={info.get('collision') or info.get('reached_goal')}")
            
    # save model 
    model.save(str(log_dir / "ppo_corall_baby_case2.zip"))
    print(f"Saved model and logs under: {log_dir}")

if __name__ == "__main__":
    main()