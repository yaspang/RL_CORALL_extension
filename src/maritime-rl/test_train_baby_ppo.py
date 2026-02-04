
# import libraries
from __future__ import annotations

import numpy as np

import cv2

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from test_env import CORALL_ReactiveAvoidanceGymEnv

import sys
import os

def _add_corall_to_syspath():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..', '..'))
    corall_root = os.path.join(repo_root, 'third_party', 'CORALL')
    if corall_root not in sys.path:
        sys.path.append(corall_root)

_add_corall_to_syspath()

from src.visualization.animate import animate_step_dense
from src.visualization.save_animation import create_video


def capture_frame_rgba(fig) -> np.ndarray:
    """
    Render `fig` using an Agg canvas and return an (H, W, 4) uint8 RGBA frame.
    Works regardless of the currently active matplotlib backend/canvas.
    """
    canvas = FigureCanvasAgg(fig)   # force an Agg canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)  # (H, W, 4)
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
    video_path="ppo_rl_case1.avi",
    max_steps=300,
    fps=10,
    deterministic=True
):
    """
        
    :param model: Description
    :param env: Description
    :param video_path: Description
    :param max_steps: Description
    :param fps: Description
    :param XMIN: Description
    :param XMAX: Description
    :param YMIN: Description
    :param YMAX: Description
    :param deterministic: Description

    all units in nmi 
    """

    obs, info = env.reset(seed=42)
    ep_ret = 0.0

    # compute fixed bounds 
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

    # XMIN, XMAX = xs0.min() - pad_x, xs0.max() + pad_x
    # YMIN, YMAX = ys0.min() - pad_y, ys0.max() + pad_y

    # case 1 over 40 nmi
    XMIN, XMAX =- 1, 41
    YMIN, YMAX = -5, 5


    frames = []
    x_hist, y_hist = [], []

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)

        # ---- CLEAR FIRST ----
        ax.clear()
        ax.set_title(f"PPO rollout (t={t})  return={ep_ret:.2f}")

        ax.set_autoscale_on(False)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_aspect("equal", adjustable="box")

        ax.plot(env.Xwpt, env.Ywpt, "k--", linewidth=2)
        ax.scatter(env.Xwpt, env.Ywpt, marker='o')
        ax.text(env.Xwpt[0], env.Ywpt[0], 'start')
        ax.text(env.Xwpt[-1], env.Ywpt[-1], 'goal')

        ax.plot(x_hist, y_hist, linewidth=1.5)

        # ---- sizes (use smaller values if ship looks huge) ----
        LOA_own = 0.03   # ~55 m if 0.03 nmi
        BOL_own = 0.006  # nmi
        CPA_own = 0.02   # nmi

        LOA_ob = np.ones(len(env.Xob)) * 0.03  # nmi
        BOL_ob = np.ones(len(env.Xob)) * 0.006 # nmi 
        CPA_ob = np.ones(len(env.Xob)) * 0.15  # nmi 

        Risk = info.get("risk", np.zeros(len(env.Xob), dtype=float))
        Vob = env.Vob


        # if t == 0:
            # print("DEBUG: len(Xob)=", len(env.Xob))
        #if len(env.Xob) > 0:
            #print("DEBUG: first obstacle (m):", env.Xob[0], env.Yob[0], "Vob[0]=", float(env.Vob[0]) if len(env.Vob)>0 else None)
            #print("DEBUG: camera nmi bounds:", XMIN, XMAX, YMIN, YMAX)
            #print("DEBUG: first obstacle (nmi):", float(env.Xob[0])/1852.0, float(env.Yob[0])/1852.0)

        #if t == 0:
            #print("DEBUG info keys:", list(info.keys()))


        # plot waypoint line (planned route)

        if t == 0 and len(env.Xob):
            x_obs = np.asarray(env.Xob)/1852.0
            y_obs = np.asarray(env.Yob)/1852.0
            print("obs nmi bounds:", x_obs.min(), x_obs.max(), y_obs.min(), y_obs.max())

        Xob = np.asarray(env.Xob, dtype=float)[:1] / 1852
        Yob = np.asarray(env.Yob, dtype=float)[:1] / 1852
        psiob = np.asarray(env.psiob, dtype=float)[:1]
        Vob = np.asarray(env.Vob, dtype=float)[:1]
        LOA_ob = np.asarray(LOA_ob, dtype=float)[:1]
        BOL_ob = np.asarray(BOL_ob, dtype=float)[:1]
        CPA_ob = np.asarray(CPA_ob, dtype=float)[:1]
        Risk = np.asarray(Risk, dtype=float)[:1] if len(Risk) else Risk 

        # --- Draw one step (this may internally call plt.axis('equal') and autoscale)
        animate_step_dense(
            x=float(env.X[0]) / 1852.0,
            y=float(env.X[1]) / 1852.0,
            psi=float(env.X[2]),
            LOA_own=LOA_own,
            BOL_own=BOL_own,
            CPA_own=CPA_own,
            Xob=np.asarray(env.Xob, dtype=float) / 1852.0,
            Yob=np.asarray(env.Yob, dtype=float) / 1852.0,
            psiob=np.asarray(env.psiob, dtype=float),
            LOA_ob=LOA_ob,
            BOL_ob=BOL_ob,
            CPA_ob=CPA_ob,
            Risk=Risk,
            Vob=Vob,
            step=t,
            ax=ax,
        )

        x_hist.append(float(env.X[0]) / 1852)
        y_hist.append(float(env.X[1]) / 1852)

        # ---- CAPTURE ----
        frame = capture_frame_rgba(fig)
        frames.append(frame)

        if t % 25 == 0:
            x, y, psi, r, b, u = env.X
            print(f"[rollout] step={t:4d}  action={action}  x={x:8.2f} y={y:8.2f} psi={psi:7.3f}  u={u:6.3f} r={r:7.3f}")

        if terminated or truncated:
            break

    create_video(frames, output_filename=video_path, fps=fps)
    print(f"[rollout end] steps={len(frames)}, return={ep_ret:.2f}, risk_max={info.get('risk_max', 0.0):.3f}")
    plt.close(fig)
    return ep_ret, len(frames), info

def main():

    # --- 1) determinstic "does action matter?" test

    env = CORALL_ReactiveAvoidanceGymEnv(case_number=1, dt=0.2, sim_time=300, K_obstacles=1)
    ret, steps, term, trunc, info = rollout_fixed_action(env, fixed_action=0, max_steps=5000)
        
    print(
            f"[Case 1] return={ret:.2f}, steps={steps}, trunc={trunc}, "
            f"risk_max={info.get('risk_max', float('nan')):.3f}, "
            f"collision={info.get('collision')}, goal={info.get('reached_goal')}"
    )


    # --- 2) PPO training (tiny sanity run)
    # use DummyVecEnv for single-process vectorized environment
    
    # case_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    vec_env = DummyVecEnv([lambda: CORALL_ReactiveAvoidanceGymEnv(case_number=1, dt=0.2, sim_time=300, K_obstacles=1)])

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
    model.learn(total_timesteps=50_000)

    # --- 3) evaluate trained policy
    eval_env = CORALL_ReactiveAvoidanceGymEnv(case_number=1, dt=0.2, sim_time=300, K_obstacles=1)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\n[PPO eval] mean_return={mean_reward:.2f} +/- {std_reward:.2f} over 10 episodes")

    # 4) rollout / visualize trained policy

    # Run a rollout that records video/frames using the trained policy
    ret, steps, info = rollout_policy_make_video_fixed_camera(model, eval_env, max_steps=1500, fps=10)

    print(f"[PPO rollout] return={ret:.2f}, steps={steps}, done={info.get('collision') or info.get('reached_goal')}")
            
    # save model 
    model.save("ppo_corall_baby")
    print("Saved model to ppo_corall_baby.zip")

if __name__ == "__main__":
    main()