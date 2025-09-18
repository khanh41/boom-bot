import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from envs.multi_bomber_env import MultiBomberEnv


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = {}

    def _on_step(self) -> bool:
        # infos là list (mỗi env 1 dict)
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            if "final_info" in info:  # Gymnasium style
                final_info = info["final_info"]
                if "episode" in final_info:
                    ep_rew = final_info["episode"]["r"]
                    self.logger.record("rollout/ep_rew_mean", ep_rew)

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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ResidualBlock(out_ch),
                ResidualBlock(out_ch),
            )

        self.cnn = nn.Sequential(
            conv_block(n_input_channels, 16),
            conv_block(16, 32),
            conv_block(32, 32),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float().permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs.permute(0, 3, 1, 2)))


# === MAIN ===
if __name__ == "__main__":
    N_ENVS = 8
    TOTAL_TIMESTEPS = 10_000_000

    env = MultiBomberEnv(max_steps=10000)
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=4, base_class="stable_baselines3")

    try:
        print("🔄 Loading existing model for fine-tuning...")
        model = PPO.load(
            "multi_bomber_selfplay_ppo_finetuned.zip",
            env=vec_env,
            device="cuda"
        )
    except Exception:
        print("⚠️ No previous model found, training from scratch.")
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(features_dim=512)
        )

        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./multi_bomber_tensorboard/",
            device="cuda"
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("multi_bomber_selfplay_ppo_finetuned")

    vec_env.close()
    print("✅ Fine-tuning done. Model saved to multi_bomber_selfplay_ppo_finetuned.zip")
