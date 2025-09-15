import argparse
import asyncio
import json
import time
import uuid

import numpy as np
import websockets

from stable_baselines3 import PPO
from agents.dqn import DQNAgent
from envs.bomber_env import ACTIONS


# -------------------
# Helpers
# -------------------
def gen_uuid():
    return str(uuid.uuid4())


def pos_to_cell(p):
    try:
        x = p.get('x') if isinstance(p, dict) else None
        y = p.get('y') if isinstance(p, dict) else None
        if x is None or y is None:
            return None
        if abs(x) <= 100 and abs(y) <= 100:
            cx = int(round(x))
            cy = int(round(y))
        else:
            cx = int(round(x / 100.0))
            cy = int(round(y / 100.0))
        return cx, cy
    except Exception:
        return None


def is_game_event_type(tag: str) -> bool:
    game_events = [
        'tick', 'initial_state', 'tick_delta',
        'bomb_placed', 'bomb_exploding_soon',
        'player_died', 'game_over'
    ]
    return tag in game_events


# -------------------
# Observation mapping
# -------------------
def decode_server_state(msg: dict, default_H=18, default_W=28) -> np.ndarray:
    if 'map' not in msg or not msg.get('map'):
        return np.zeros((6, default_H, default_W), dtype=np.float32)

    tiles = msg['map'].get('tiles', None)
    if tiles is None:
        return np.zeros((6, default_H, default_W), dtype=np.float32)

    H = len(tiles)
    W = len(tiles[0]) if H > 0 else 0
    if H == 0 or W == 0:
        return np.zeros((6, default_H, default_W), dtype=np.float32)

    obs = np.zeros((6, H, W), dtype=np.float32)

    # channel 0: static
    for r in range(H):
        for c in range(W):
            t = tiles[r][c]
            if t == 1:
                obs[0, r, c] = 1.0
            elif t == 2:
                obs[0, r, c] = 0.5
            else:
                obs[0, r, c] = 0.0

    # bombs
    for b in msg.get('bombs', []):
        try:
            cell = pos_to_cell(b.get('p', b.get('position', {})))
            if not cell:
                continue
            cx, cy = cell
            fuse = b.get('c', b.get('countdownTicks', 180))
            obs[1, cy, cx] = max(min(fuse / 180.0, 1.0), 0.0)
        except Exception:
            continue

    # items
    for it in msg.get('items', []):
        try:
            cell = pos_to_cell(it.get('p', it.get('position', {})))
            if not cell:
                continue
            cx, cy = cell
            obs[3, cy, cx] = 1.0
        except Exception:
            continue

    # players -> mark enemies (channel 5). 'self' channel 4 will be set later
    for p in msg.get('players', []):
        try:
            cell = pos_to_cell(p.get('p', p.get('position', {})))
            if not cell:
                continue
            cx, cy = cell
            obs[5, cy, cx] = 1.0
        except Exception:
            continue

    return obs


def encode_action_to_server(a: int) -> dict:
    table = {0: 'stay', 1: 'u', 2: 'd', 3: 'l', 4: 'r', 5: 'b'}
    action_char = table.get(int(a), 'stay')
    return {'type': 'control', 'data': {'action': action_char}}


def encode_control_ghost(x: int, y: int) -> dict:
    return {'type': 'control_ghost', 'data': {'action': {'x': int(x), 'y': int(y)}}}


# -------------------
# Stable-Baselines3 PPO Wrapper
# -------------------
class SB3PPOWrapper:
    def __init__(self, path, device="cpu"):
        self.model = PPO.load(path, device=device)
        # lấy shape mà model expect
        self.obs_shape = self.model.observation_space.shape[0]

    def act(self, obs: np.ndarray):
        # flatten obs
        obs_flat = obs.flatten().astype(np.float32)
        # cắt hoặc reshape đúng chiều mà PPO expect
        obs_proc = obs_flat[:self.obs_shape]
        obs_batch = np.expand_dims(obs_proc, axis=0)  # (1,858)
        action, _ = self.model.predict(obs_batch, deterministic=True)
        return int(action)


# -------------------
# Main run loop
# -------------------
async def run(uri: str, policy_path: str, algo: str,
              player_id: str, game_id: str,
              team_id: str, player_name: str, team_name: str):

    if algo.lower() == 'dqn':
        agent = DQNAgent(in_channels=6, n_actions=ACTIONS, env_h=18, env_w=28)
        agent.load(policy_path)
        select_action = lambda obs: agent.act(obs, greedy=True)
    else:  # PPO via Stable-Baselines3
        agent = SB3PPOWrapper(policy_path)
        select_action = lambda obs: agent.act(obs)

    reconnect_delay = 3
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print(f"[ws_inference] Connected to {uri}")
                join_msg = {
                    "type": "join_game",
                    "data": {
                        "gameId": game_id,
                        "teamId": team_id,
                        "playerId": player_id,
                        "playerName": player_name,
                        "teamName": team_name,
                        "role": "player"
                    }
                }
                await ws.send(json.dumps(join_msg))
                print("[ws_inference] Sent join_game")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    if msg.get('type') == 'join_success':
                        print("[ws_inference] join_success:", msg.get('data', {}))
                        continue
                    if msg.get('type') == 'join_error':
                        print("[ws_inference] join_error:", msg.get('data', {}))
                        continue

                    if is_game_event_type(msg.get('tag') or msg.get('type') or ''):
                        obs = decode_server_state(msg)

                        # mark self in channel 4
                        if 'players' in msg and isinstance(msg['players'], list):
                            for p in msg['players']:
                                try:
                                    if p.get('id') == player_id:
                                        cell = pos_to_cell(p.get('p', p.get('position', {})))
                                        if cell:
                                            cx, cy = cell
                                            if 0 <= cy < obs.shape[1] and 0 <= cx < obs.shape[2]:
                                                obs[4, cy, cx] = 1.0
                                                obs[5, cy, cx] = 0.0
                                except Exception:
                                    continue

                        # check alive
                        am_dead = False
                        if 'players' in msg and isinstance(msg['players'], list):
                            for p in msg['players']:
                                if p.get('id') == player_id:
                                    if (p.get('s') or p.get('status')) == 'dead':
                                        am_dead = True
                                    break

                        if am_dead:
                            ghost_target = (obs.shape[2] // 2, obs.shape[1] // 2)
                            out = encode_control_ghost(*ghost_target)
                            await ws.send(json.dumps(out))
                        else:
                            now = time.time()
                            a = select_action(obs)
                            out = encode_action_to_server(a)
                            print(f"[ws_inference] action {a} sent in {time.time()-now:.3f}s")
                            await ws.send(json.dumps(out))

                    elif msg.get('type') == 'game_over' or msg.get('tag') == 'game_over':
                        print("[ws_inference] game_over:", msg.get('data', {}))
                    else:
                        pass

        except (websockets.ConnectionClosedError, ConnectionRefusedError, OSError) as e:
            print(f"[ws_inference] connection error: {e}. reconnect in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            continue
        except Exception as e:
            print(f"[ws_inference] unexpected error: {e}. reconnecting in {reconnect_delay}s...")
            await asyncio.sleep(reconnect_delay)
            continue


# -------------------
# CLI
# -------------------
def main():
    parser = argparse.ArgumentParser(description="WebSocket inference bridge for Bomberman RL agent")
    parser.add_argument('--uri', type=str, default="ws://localhost:5001")
    parser.add_argument('--policy', type=str, required=True, help="Path to SB3 PPO model .zip or DQN .pt")
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo'], default='ppo')
    parser.add_argument('--player-id', type=str, default=None)
    parser.add_argument('--player-name', type=str, default="PyBot")
    parser.add_argument('--team-id', type=str, default=None)
    parser.add_argument('--team-name', type=str, default="PyTeam")
    parser.add_argument('--game-id', type=str, default=None)
    args = parser.parse_args()

    player_id = args.player_id or gen_uuid()
    print(f"[ws_inference] using player_id={player_id}")

    asyncio.run(
        run(args.uri, args.policy, args.algo, player_id, args.game_id,
            args.team_id, args.player_name, args.team_name)
    )


if __name__ == '__main__':
    main()
