import argparse, os
import numpy as np
import torch
from envs.bomber_env import BomberEnv, ACTIONS
from agents.dqn import DQNAgent
from utils.common import device, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--save-path', type=str, default='checkpoints/dqn.pt')
    parser.add_argument('--grid', type=int, default=11)
    args = parser.parse_args()

    env = BomberEnv(grid_size=args.grid)
    agent = DQNAgent(in_channels=env.obs_shape[0], n_actions=ACTIONS)
    ensure_dir(args.save_path)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        total_r = 0.0
        while not (done or trunc):
            action = agent.act(obs)
            obs2, r, done, trunc, info = env.step(action)
            agent.push(obs, action, r, obs2, done, trunc)
            agent.learn()
            obs = obs2
            total_r += r
        if (ep+1) % 50 == 0:
            print(f'Episode {ep+1}: R={total_r:.2f}, eps={agent.epsilon():.3f}')
        if (ep+1) % 200 == 0:
            agent.save(args.save_path)
    agent.save(args.save_path)
    print('Saved to', args.save_path)

if __name__ == "__main__":
    main()
