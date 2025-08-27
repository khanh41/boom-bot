import argparse, os
import numpy as np
import torch
from collections import defaultdict
from envs.bomber_env import BomberEnv, ACTIONS
from agents.ppo import PPOAgent
from utils.common import device, ensure_dir

def collect_rollout(env, agent, horizon=4096):
    traj = defaultdict(list)
    obs, _ = env.reset()
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
    return traj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-path', type=str, default='checkpoints/ppo_actor.pt')
    parser.add_argument('--save-critic', type=str, default='checkpoints/ppo_critic.pt')
    parser.add_argument('--grid', type=int, default=11)
    parser.add_argument('--horizon', type=int, default=8192)
    args = parser.parse_args()

    env = BomberEnv(grid_size=args.grid)
    agent = PPOAgent(in_channels=env.obs_shape[0], n_actions=ACTIONS)
    ensure_dir(args.save_path); ensure_dir(args.save_critic)

    steps = 0
    while steps < args.timesteps:
        traj = collect_rollout(env, agent, horizon=args.horizon)
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
        if steps % (args.horizon * 4) == 0:
            print(f"Steps {steps}: {stats}")
            agent.save(args.save_path, args.save_critic)

    agent.save(args.save_path, args.save_critic)
    print('Saved to', args.save_path, 'and', args.save_critic)

if __name__ == "__main__":
    main()
