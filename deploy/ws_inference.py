import asyncio
import json
import argparse
import numpy as np
import torch
import websockets
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from envs.bomber_env import ACTIONS
from utils.common import device

def decode_server_state(msg: dict) -> np.ndarray:
    """Map the tournament server JSON into the model observation.
    TODO: Replace this stub with real mapping.
    Expecting channels-first float32 array of shape [C,H,W].
    """
    # Example: simply pass a dummy obs to show the wiring works
    C, H, W = 6, 11, 11
    return np.zeros((C,H,W), dtype=np.float32)

def encode_action_to_server(a: int) -> dict:
    """Map discrete action -> server protocol JSON."""
    table = {0:'stay',1:'up',2:'down',3:'left',4:'right',5:'place_bomb'}
    return {'action': table[a]}

async def run(uri: str, policy_path: str, algo: str):
    if algo.lower() == 'dqn':
        # only need the Q network; build a dummy env shape to construct
        agent = DQNAgent(in_channels=6, n_actions=ACTIONS)
        agent.load(policy_path)
        select_action = lambda obs: agent.act(obs, greedy=True)
    else:
        agent = PPOAgent(in_channels=6, n_actions=ACTIONS)
        agent.load(policy_path, policy_path.replace('actor','critic'))
        def select_action(obs):
            a, _, _ = agent.get_action(obs)
            return a

    async with websockets.connect(uri) as ws:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            obs = decode_server_state(msg)
            a = select_action(obs)
            out = encode_action_to_server(a)
            await ws.send(json.dumps(out))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--algo', type=str, choices=['dqn','ppo'], default='dqn')
    args = parser.parse_args()
    asyncio.run(run(args.uri, args.policy, args.algo))

if __name__ == "__main__":
    main()
