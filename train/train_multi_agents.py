import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.multi_bomber_env import MultiBomberEnv

# === HYPERPARAMS ===
NUM_AGENTS = 4
ACTIONS = 7  # per-agent action space size (0..6)
JOINT_ACTION_SIZE = ACTIONS ** NUM_AGENTS  # 7^4 = 2401
ENV_PER_CPU = 1
N_ENVS = 8  # number of parallel envs for data collection (adjust to CPU)
TOTAL_TIMESTEPS = 2_000_000
SEED_BASE = 1000


# === WRAPPER: Convert PettingZoo ParallelEnv -> single-agent Gym Env (centralized) ===
class PettingZooParallelToGymWrapper(gymnasium.Env):
    """
    Wraps a PettingZoo ParallelEnv into a single-agent gym.Env:
    - obs: concatenation of per-agent observations (flattened)
    - action: Discrete(ACTIONS ** NUM_AGENTS) where each joint-action maps to per-agent actions
    - reward: sum of per-agent rewards (you can change to avg or custom)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env: MultiBomberEnv, reward_mode: str = "sum"):
        super().__init__()
        self.env = env
        self.agents = list(self.env.agents)
        self.num_agents = len(self.agents)
        assert self.num_agents == NUM_AGENTS, "NUM_AGENTS mismatch with env.agents"
        self.obs_shape = self.env.obs_shape  # (C, H, W)
        self.per_agent_obs_size = int(np.prod(self.obs_shape))
        # observation is concatenated flattened vector of all agents
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_agents * self.per_agent_obs_size,),
            dtype=np.float32
        )
        # joint action is Discrete(ACTIONS ** num_agents)
        self.action_space = spaces.Discrete(ACTIONS ** self.num_agents)
        self.reward_mode = reward_mode
        self.last_obs = None
        self.episode_reward = 0

    def _obs_dict_to_flat(self, obs_dict):
        arrs = []
        for a in self.agents:
            ob = obs_dict[a]  # shape (C, H, W)
            arrs.append(np.asarray(ob, dtype=np.float32).ravel())
        return np.concatenate(arrs).astype(np.float32)

    def reset(self, **kwargs):
        obs_dict, infos = self.env.reset(**kwargs)
        flat = self._obs_dict_to_flat(obs_dict)
        self.last_obs = flat
        self.episode_reward = 0
        return flat, infos

    def step(self, action):
        idx = int(np.asarray(action).item())
        multi = np.unravel_index(idx, tuple([ACTIONS] * self.num_agents))
        action_dict = {agent: int(act) for agent, act in zip(self.agents, multi)}

        obs_dict, rewards_dict, terminations, truncations, infos = self.env.step(action_dict)

        # gộp reward
        if self.reward_mode == "sum":
            reward = float(sum(rewards_dict.values()))
        elif self.reward_mode == "mean":
            reward = float(np.mean(list(rewards_dict.values())))
        else:
            reward = float(sum(rewards_dict.values()))

        # cộng dồn reward cho episode
        self.episode_reward += reward

        terminated = any(
            all(self.env.players[a].status == "dead" for a in [p for p, t in self.env.teams.items() if t == team])
            for team in ["A", "B"]
        )
        truncated = bool(any(truncations.values()))

        obs_flat = self._obs_dict_to_flat(obs_dict)
        self.last_obs = obs_flat

        done = terminated or truncated

        info = {
            "per_agent_rewards": rewards_dict,
            "terminations": terminations,
            "truncations": truncations,
            **(infos.get(self.agents[0], {}) if isinstance(infos, dict) else {})
        }

        # Nếu kết thúc episode thì thêm log cho SB3
        if done:
            info["episode"] = {
                "r": self.episode_reward,  # tổng reward cả episode
                "l": self.env.steps,
            }

        return obs_flat, reward, terminated, truncated, info

    def render(self, mode='human'):
        # delegate to underlying env.render (which returns ascii string)
        try:
            s = self.env.render()
            if mode == 'human':
                print(s)
            return s
        except Exception:
            return None

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# === Utility to create env instances for SubprocVecEnv ===
def make_wrapped_env(seed: int = 0):
    def _init():
        base = MultiBomberEnv(grid_w=28, grid_h=18, max_steps=1000, seed=seed)
        wrapped = PettingZooParallelToGymWrapper(base)
        return wrapped

    return _init


if __name__ == "__main__":
    # Create n parallel envs
    env_fns = []
    for i in range(N_ENVS):
        env_fns.append(make_wrapped_env(seed=SEED_BASE + i))

    if N_ENVS == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    # Policy network size needs to handle large obs dim; tune net_arch
    obs_dim = vec_env.observation_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim (Discrete): {JOINT_ACTION_SIZE}")

    policy_kwargs = dict(net_arch=[dict(pi=[512, 256], vf=[512, 256])])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./multi_bomber_tensorboard/",
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("multi_bomber_ppo_centralized")

    # Close
    vec_env.close()
    print("Training complete. Model saved to multi_bomber_ppo_centralized.zip")
