from envs.bomber_env import BomberEnv

env = BomberEnv()
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # hoặc policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(mode="human")
    done = terminated or truncated
