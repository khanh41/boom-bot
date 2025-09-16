import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

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
        base = MultiBomberEnv(seed=seed)
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
                    f"Reward: {self.current_rewards:.2f} "
                    f"Len: {self.current_length} "
                    f"(Avg100 R={mean_r:.2f}, L={mean_l:.1f})"
                )

            self.current_rewards = 0
            self.current_length = 0

        return True


# === MAIN ===
if __name__ == "__main__":
    N_ENVS = 8
    TOTAL_TIMESTEPS = 5_000_000

    env_fns = [make_selfplay_env(seed=1000 + i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns) if N_ENVS > 1 else DummyVecEnv(env_fns)

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

    model = PPO(
        "MlpPolicy",
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

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save("multi_bomber_selfplay_ppo")

    vec_env.close()
    print("✅ Training done. Model saved to multi_bomber_selfplay_ppo.zip")
