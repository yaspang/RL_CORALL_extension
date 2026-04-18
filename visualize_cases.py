import pygame
from env import ImazuEvalEnv

STEPS_PER_FRAME = 4

env = ImazuEvalEnv(render_mode="human", step_size=60, fps=30)

for _ in range(ImazuEvalEnv.N_CASES):
    obs, _ = env.reset()

    while True:
        if env._renderer is not None:
            env._renderer.events = []

        terminated, truncated = False, False
        for _ in range(STEPS_PER_FRAME):
            obs, _, terminated, truncated, _ = env.step([0.0, 0.0])
            if terminated or truncated:
                break
            if env._renderer is not None and env._renderer.interrupted:
                break

        env.render()

        events = env._renderer.events
        if any(e.type == pygame.QUIT for e in events):
            env.close()
            raise SystemExit
        if terminated or truncated or any(e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE for e in events):
            break

env.close()
