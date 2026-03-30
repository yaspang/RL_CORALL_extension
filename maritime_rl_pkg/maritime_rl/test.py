def smoke_test_env():
    from maritime_rl_pkg.maritime_rl.multi_agent_env_ppo import MultiShipParallelEnv
    env = MultiShipParallelEnv(case_number=2, dt=0.2, sim_time=20.0, seed=0)
    obs, infos = env.reset(seed=0)
    print("reset ok, agents:", env.agents)

    for t in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"step {t+1} ok")
        if any(terminations.values()) or any(truncations.values()):
            print("episode ended early")
            break

    if hasattr(env, "close"):
        env.close()
    
if __name__ == "__main__":
    smoke_test_env()