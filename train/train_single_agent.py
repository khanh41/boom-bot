import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.bomber_env import BomberEnv


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
    TOTAL_TIMESTEPS = 500_000

    env = make_vec_env(lambda: BomberEnv(max_steps=10000), n_envs=N_ENVS)

    try:
        print("🔄 Loading existing model for fine-tuning...")
        model = PPO.load(
            "multi_bomber_selfplay_ppo_finetuned.zip",
            env=env,
            device="cuda"
        )
    except Exception:
        print("⚠️ No previous model found, training from scratch.")
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(features_dim=512)
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            clip_range=0.2,
            ent_coef=0.05,
            # policy_kwargs=policy_kwargs,
            tensorboard_log="./multi_bomber_tensorboard/",
            device="cuda"
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("multi_bomber_selfplay_ppo_finetuned")

    env.close()
    print("✅ Fine-tuning done. Model saved to multi_bomber_selfplay_ppo_finetuned.zip")
