import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack

from envs.multi_bomber_env import MultiBomberEnv

env = MultiBomberEnv()

# Apply SuperSuit wrappers for preprocessing
env = ss.dtype_v0(env, 'float32')  # Ensure float32 dtype
env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize observations

# Convert to vectorized env for SB3
vec_env = ss.pettingzoo_env_to_vec_env_v1(env)

# Vectorize multiple instances for faster training
vec_env = ss.concat_vec_envs_v1(vec_env, 8, num_cpus=4, base_class='stable_baselines3')

# Frame stacking for temporal information
vec_env = VecFrameStack(vec_env, n_stack=4)

# Create PPO model with CNN policy
model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    n_steps=256,
    batch_size=256,
    learning_rate=0.0001,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
    n_epochs=10
)

# Train the model
model.learn(total_timesteps=2000000)

# Save the trained model
model.save("ppo_multi_bomber")

# Optional: Test the trained model
obs = vec_env.reset()
while True:
    actions, _ = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(actions)
    print(vec_env.render())  # Render the environment
    if any(dones):
        break
