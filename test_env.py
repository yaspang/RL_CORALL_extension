import pygame
from env import CORALLEnv

STEPS_PER_FRAME = 10

env = CORALLEnv(render_mode="human", step_size=60, max_intruders=20, render_every=30)
obs, _ = env.reset()

while True:
    if env._renderer is not None:
        env._renderer.events = []

    terminated, truncated = False, False
    for _ in range(STEPS_PER_FRAME):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
        if env._renderer is not None and env._renderer.interrupted:
            break

    env.render()

    if any(e.type == pygame.QUIT for e in env._renderer.events):
        break
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
