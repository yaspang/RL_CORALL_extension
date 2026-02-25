
# import libraries
from __future__ import annotations
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import cv2

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.evaluation import evaluate_policy

from .test_env import CORALL_ReactiveAvoidanceGymEnv

import sys
import os

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
    record_every=1, 
    dpi=150,
    figsize=(7, 7),
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
        video_path = f"ppo_rollout_case{env.case_number}.avi"

    # determine rollout horizon
    if max_steps is None:
        max_steps = int(getattr(env, "max_steps", 3000) or 3000)

    ep_ret = 0.0

    # compute fixed bounds based on initial geometry (in nmi units)
    x_own0 = float(env.X[0]) / 1852.0
    y_own0 = float(env.X[1]) / 1852.0

    x_obs0 = np.asarray(env.Xob, dtype=float) / 1852
    y_obs0 = np.asarray(env.Yob, dtype=float) / 1852

    x_wpt = np.asarray(env.Xwpt, dtype=float)
    y_wpt = np.asarray(env.Ywpt, dtype=float)

    xs0 = np.concatenate([np.array([x_own0]), x_obs0, x_wpt])
    ys0 = np.concatenate([np.array([y_own0]), y_obs0, y_wpt])
    
    pad_x = 2 
    pad_y = 5

    XMIN, XMAX = xs0.min() - pad_x, xs0.max() + pad_x
    YMIN, YMAX = ys0.min() - pad_y, ys0.max() + pad_y

    # output fps derived from env time step 
    fps_out = int(round(1.0 / float(env.dt)))

    # grab one frame to get size (after first draw)
    writer = None 
    steps_recorded = 0
    x_hist, y_hist = [], []

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    try:
        t = 0 # initialize t before loop
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
                # ---- sizes (use smaller values if ship looks huge) ----
                LOA_own = 0.03   # ~55 m if 0.03 nmi
                BOL_own = 0.006  # nmi
                CPA_own = 0.02   # nmi

                LOA_ob = np.ones(len(env.Xob)) * 0.03  # nmi
                BOL_ob = np.ones(len(env.Xob)) * 0.006 # nmi 
                CPA_ob = np.ones(len(env.Xob)) * 0.15  # nmi 

                Risk = info.get("risk", np.zeros(len(env.Xob), dtype=float))
               
                # single head-on vessel for case 1 imazu 
                Xob = np.asarray(env.Xob, dtype=float) / 1852
                Yob = np.asarray(env.Yob, dtype=float) / 1852
                psiob = np.asarray(env.psiob, dtype=float)
                Vob = np.asarray(env.Vob, dtype=float)
               
                #LOA_ob = np.asarray(LOA_ob, dtype=float)
                #BOL_ob = np.asarray(BOL_ob, dtype=float)
                #CPA_ob = np.asarray(CPA_ob, dtype=float)
                #Risk = np.asarray(Risk, dtype=float) if len(Risk) else Risk 

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

                # force fixed camera rendering 
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
                ax.plot([x_hist[-1]], [y_hist[-1]], marker="o", markersize=4, zorder=51)  # current point

                # capture rendered frame
                frame_rgba = capture_frame_rgba(fig)
                frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGB2BGR)

                # init writer once the framesize is known
                if writer is None:
                    h, w = frame_bgr.shape[:2]

                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # type: ignore 
                    except (AttributeError, cv2.error):
                        print("Warning: VideoWriter_fourcc not available, using -1 for default codec")
                        fourcc = -1
                        
                    writer = cv2.VideoWriter(video_path, fourcc, fps_out, (w, h))

                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed to open: {video_path}")
        
                writer.write(frame_bgr)

                if t % 25 == 0:
                    x, y, psi, r, b, u = env.X
                    print(f"[rollout] step={t:4d}  action={action}  x={x:8.2f} y={y:8.2f} psi={psi:7.3f}  u={u:6.3f} r={r:7.3f}")

                if terminated or truncated:
                    break
    
    finally:
        if writer is not None: 
            writer.release()
        plt.close(fig)

    return ep_ret, (t + 1), info

def main():

    # --- 1) determinstic "does action matter?" test

    env = CORALL_ReactiveAvoidanceGymEnv(case_number=2, dt=0.2, K_obstacles=1, max_steps_cap=2000)
    ret, steps, term, trunc, info = rollout_fixed_action(env, fixed_action=0, max_steps=5000)
        
    print(
            f"[Case 1] return={ret:.2f}, steps={steps}, trunc={trunc}, "
            f"risk_max={info.get('risk_max', float('nan')):.3f}, "
            f"collision={info.get('collision')}, goal={info.get('reached_goal')}"
    )


    # --- 2) PPO training (tiny sanity run)
    # use DummyVecEnv for single-process vectorized environment
    def make_env():
        env = CORALL_ReactiveAvoidanceGymEnv(case_number=2, dt=0.2, K_obstacles=1, max_steps_cap=2000)
        return Monitor(env, filename='monitor.csv')
    
    vec_env = DummyVecEnv([make_env])

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

    # train for a small number of timesteps to see learning steps
    model.learn(total_timesteps=1_000_000)

    # load monitor results and print reward history
    results = load_results('.')
    x, y = ts2xy(results, 'timesteps')

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('PPO Training Reward History over Training  Time Steps')
    plt.grid()
    plt.savefig("ppo_training_reward_history.png", dpi=150)
    plt.close()

    print("saved reward curve to ppo_training_reward_history.png")
 
    # --- 3) evaluate trained policy
    eval_env = CORALL_ReactiveAvoidanceGymEnv(case_number=2, dt=0.2, K_obstacles=1)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\n[PPO eval] mean_return={mean_reward:.2f} +/- {std_reward:.2f} over 10 episodes")

    # 4) rollout / visualize trained policy

    # Run a rollout that records video/frames using the trained policy
    fps = int(round(1.0 / env.dt))
    ret, steps, info = rollout_policy_make_video_fixed_camera(model, eval_env, video_path=None, max_steps=None, deterministic=True, record_every=5)

    print(f"[PPO rollout] return={ret:.2f}, steps={steps}, done={info.get('collision') or info.get('reached_goal')}")
            
    # save model 
    model.save("ppo_corall_baby_case2")
    print("Saved model to ppo_corall_baby_case2.zip")

if __name__ == "__main__":
    main()