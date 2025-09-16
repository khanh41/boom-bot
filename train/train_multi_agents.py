import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.multi_bomber_env import MultiBomberEnv


# === Self-Play Wrapper ===
class SelfPlayWrapper(gymnasium.Env):
    def __init__(self, env: MultiBomberEnv):
        super().__init__()
        self.env = env
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        self.current_agent_idx = 0

        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])

        self.last_obs = None

    def reset(self, **kwargs):
        obs_dict, infos = self.env.reset(**kwargs)
        self.current_agent_idx = 0
        agent = self.agents[self.current_agent_idx]
        self.last_obs = obs_dict[agent]
        return self.last_obs, infos[agent]

    def step(self, action):
        agent = self.agents[self.current_agent_idx]
        action_dict = {a: 0 for a in self.agents}  # mặc định stay
        action_dict[agent] = int(action)

        obs_dict, rewards, terminations, truncations, infos = self.env.step(action_dict)

        reward = rewards[agent]
        done = terminations[agent] or truncations[agent]
        info = infos[agent]

        self.current_agent_idx = (self.current_agent_idx + 1) % self.num_agents
        next_agent = self.agents[self.current_agent_idx]
        self.last_obs = obs_dict[next_agent]

        if all(terminations.values()) or all(truncations.values()):
            print("🔴 Episode ended at step", self.env.steps, " - terminations:", terminations, " truncations:",
                  truncations)
            obs_dict, infos = self.env.reset()
            self.current_agent_idx = 0
            next_agent = self.agents[self.current_agent_idx]
            self.last_obs = obs_dict[next_agent]
            done = True

        return self.last_obs, reward, done, False, info

    def render(self, mode="human"):
        s = self.env.render()
        if mode == "human":
            print(s)
        return s

    def close(self):
        self.env.close()


# === Utility để tạo envs ===
def make_selfplay_env(seed=0):
    def _init():
        base = MultiBomberEnv(seed=seed, max_steps=10000)
        return SelfPlayWrapper(base)

    return _init


# === Custom Logging Callback ===
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]
        self.current_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)

            if self.verbose > 0:
                mean_r = np.mean(self.episode_rewards[-100:])
                mean_l = np.mean(self.episode_lengths[-100:])
                print(
                    f"[Episode {len(self.episode_rewards)}] "
                    f"Env Steps: {self.num_timesteps} "
                    f"Reward: {self.current_rewards:.2f} "
                    f"Len: {self.current_length} "
                    f"(Avg100 R={mean_r:.2f}, L={mean_l:.1f})"
                )

            self.current_rewards = 0
            self.current_length = 0

        return True


class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]  # 6

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute output dim
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float().permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))


# === MAIN ===
if __name__ == "__main__":
    N_ENVS = 8
    TOTAL_TIMESTEPS = 2_000_000  # số bước train thêm (fine-tuning)

    env_fns = [make_selfplay_env(seed=2000 + i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns) if N_ENVS > 1 else DummyVecEnv(env_fns)

    # Nếu đã có model trước đó
    try:
        print("🔄 Loading existing model for fine-tuning...")
        model = PPO.load(
            "multi_bomber_selfplay_ppo.zip",
            env=vec_env,
            device="cuda"
        )
    except Exception:
        print("⚠️ No previous model found, training from scratch.")
        policy_kwargs = dict(
            features_extractor_class=SmallCNN,
            features_extractor_kwargs=dict(features_dim=256)
        )

        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=256,
            n_epochs=10,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./multi_bomber_selfplay_tensorboard/",
            device="cuda"
        )

    callback = RewardLoggingCallback(verbose=1)

    # Tiếp tục train thêm
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save("multi_bomber_selfplay_ppo_finetuned")

    vec_env.close()
    print("✅ Fine-tuning done. Model saved to multi_bomber_selfplay_ppo_finetuned.zip")
