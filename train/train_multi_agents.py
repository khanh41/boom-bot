from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.multi_bomber_env import SingleAgentWrapper, MultiBomberEnv


def make_env():
    return SingleAgentWrapper(MultiBomberEnv())


env = make_vec_env(make_env, n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_bomberman")
