import numpy as np
import torch as th
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.multi_bomber_env import MultiBomberEnv, ACTIONS


# Wrapper to make PettingZoo compatible with Stable-Baselines3
class SB3Wrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=env.obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTIONS)
        self.current_agent = None

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.current_agent = list(self.agents)[0]
        return obs[self.current_agent], infos[self.current_agent]

    def step(self, action):
        actions = {agent: 0 for agent in self.agents}  # Default action (stay)
        actions[self.current_agent] = action
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        reward = rewards[self.current_agent]
        terminated = terminations[self.current_agent]
        truncated = truncations[self.current_agent]
        # Rotate to next agent to simulate parallel execution
        agents = list(self.agents)
        current_idx = agents.index(self.current_agent)
        self.current_agent = agents[(current_idx + 1) % len(agents)]
        return obs[self.current_agent], reward, terminated, truncated, infos[self.current_agent]

    def observation_space(self, agent):
        return self.observation_space

    def action_space(self, agent):
        return self.action_space


def make_env(seed=None):
    def _init():
        env = MultiBomberEnv(grid_w=28, grid_h=18, max_steps=3000, seed=seed)
        return SB3Wrapper(env)

    return _init


def evaluate_policy(model, env, n_eval_episodes=10):
    episode_rewards = {agent: [] for agent in env.get_attr("agents")[0]}
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = {agent: False for agent in env.get_attr("agents")[0]}
        ep_rewards = {agent: 0.0 for agent in env.get_attr("agents")[0]}
        while not all(done.values()):
            actions = {}
            for agent in env.get_attr("agents")[0]:
                if not done[agent]:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = action
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for agent in rewards:
                ep_rewards[agent] += rewards[agent]
                done[agent] = terminations[agent] or truncations[agent]
        for agent in ep_rewards:
            episode_rewards[agent].append(ep_rewards[agent])

    # Print evaluation results
    for agent in episode_rewards:
        mean_reward = np.mean(episode_rewards[agent])
        std_reward = np.std(episode_rewards[agent])
        print(f"Agent {agent}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")


def main():
    # Hyperparameters
    n_envs = 8  # Number of parallel environments
    total_timesteps = 1_000_000  # Total training steps
    learning_rate = 3e-4
    batch_size = 64
    n_steps = 2048
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.01

    # Create vectorized environment
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv
    )

    # Define policy architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Two hidden layers for actor and critic
        activation_fn=th.nn.ReLU,
    )

    # Initialize PPO model
    model = PPO(
        policy="CnnPolicy",  # Use CNN for grid-based observations
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )

    # Save the model
    model.save("ppo_multi_bomber")

    # Evaluate the model (optional)
    evaluate_policy(model, env, n_eval_episodes=10)


if __name__ == "__main__":
    main()
