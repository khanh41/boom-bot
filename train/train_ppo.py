import argparse, os
import numpy as np
import torch
from collections import defaultdict
from envs.bomber_env import BomberEnv, ACTIONS
from agents.ppo import PPOAgent
from utils.common import device, ensure_dir

def collect_rollout(env, agent, horizon=4096, last_obs=None):
    traj = defaultdict(list)
    obs = last_obs if last_obs is not None else env.reset()[0]

    for _ in range(horizon):
        a, logp, v = agent.get_action(obs)
        obs2, r, done, trunc, _ = env.step(a)
        traj['obs'].append(obs)
        traj['act'].append(a)
        traj['logp'].append(logp)
        traj['val'].append(v)
        traj['rew'].append(r)
        traj['done'].append(done)
        obs = obs2
        if done or trunc:
            obs, _ = env.reset()
    return traj, obs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-path', type=str, default='checkpoints/ppo_actor.pt')
    parser.add_argument('--save-critic', type=str, default='checkpoints/ppo_critic.pt')
    parser.add_argument('--grid-w', type=int, default=28)
    parser.add_argument('--grid-h', type=int, default=18)
    parser.add_argument('--horizon', type=int, default=8192)
    parser.add_argument('--resume', action='store_true', help="Resume from existing checkpoint")
    parser.add_argument('--patience', type=int, default=100)
    args = parser.parse_args()

    env = BomberEnv(grid_w=args.grid_w, grid_h=args.grid_h)
    agent = PPOAgent(in_channels=env.obs_shape[0], n_actions=ACTIONS)
    ensure_dir(args.save_path)
    ensure_dir(args.save_critic)

    # fine-tuning
    if args.resume and os.path.exists(args.save_path) and os.path.exists(args.save_critic):
        agent.load(args.save_path, args.save_critic)
        print(f"Loaded checkpoints → continuing training")

    steps = 0
    reward_history = []
    last_obs, _ = env.reset()
    best_reward = -np.inf

    while steps < args.timesteps:
        traj, last_obs = collect_rollout(env, agent, horizon=args.horizon, last_obs=last_obs)
        steps += args.horizon

        # GAE
        adv, ret = agent.compute_gae(traj['rew'], traj['val'], traj['done'], gamma=agent.gamma, lam=agent.lam)
        batch = {
            'obs': np.array(traj['obs'], dtype=np.float32),
            'act': np.array(traj['act'], dtype=np.int64),
            'logp': np.array(traj['logp'], dtype=np.float32),
            'adv': np.array(adv, dtype=np.float32),
            'ret': np.array(ret, dtype=np.float32),
        }
        stats = agent.update(batch)

        total_reward = sum(traj['rew'])
        reward_history.append(total_reward)

        # Check reward improvement
        if len(reward_history) > args.patience:
            recent = reward_history[-args.patience:]
            if max(recent) <= best_reward:
                last_obs, _ = env.reset()
                print(f"Step {steps}: reward không cải thiện {args.patience} rollout → reset env")
            else:
                best_reward = max(best_reward, max(recent))

        if steps % (args.horizon * 4) == 0:
            print(f"Steps {steps}: {stats}, recent reward: {total_reward:.2f}")
            agent.save(args.save_path, args.save_critic)

    agent.save(args.save_path, args.save_critic)
    print('Training finished → saved to', args.save_path, 'and', args.save_critic)

if __name__ == "__main__":
    main()
