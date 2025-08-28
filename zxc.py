import time

from envs.bomber_env import BomberEnv, Bomb

env = BomberEnv()
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # hoặc policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
