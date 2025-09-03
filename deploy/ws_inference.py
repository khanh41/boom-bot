import argparse
import asyncio
import json
import time
import uuid

import numpy as np
import websockets

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from envs.bomber_env import ACTIONS


# -------------------
# Helpers
# -------------------
def gen_uuid():
    return str(uuid.uuid4())


def pos_to_cell(p):
    """Convert quantized position (x,y) where server uses *100 to integer cell coords.
    Accepts dict-like or object with ['x','y'] keys.
    Returns (cx, cy) or None on failure.
    """
    try:
        # p may be {'x': 500, 'y': 300} or {'x': 5, 'y':3}
        x = p.get('x') if isinstance(p, dict) else None
        y = p.get('y') if isinstance(p, dict) else None
        if x is None or y is None:
            return None
        # if positions are already small (<100) we assume they're cell coords; otherwise quantized*100
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
    game_events = ['tick', 'initial_state', 'tick_delta', 'bomb_placed', 'bomb_exploding_soon', 'player_died',
                   'game_over']
    return tag in game_events


# -------------------
# Observation mapping
# -------------------
def decode_server_state(msg: dict, default_H=18, default_W=28) -> np.ndarray:
    """Build an obs [C,H,W] as env expects. If msg lacks map, return zeros.
    Channels:
      0: static (wall=1.0, brick=0.5)
      1: bombs normalized fuse
      2: flames (not provided by server -> left 0)
      3: items
      4: self
      5: enemies
    """
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
            # normalize by typical fuse (config uses 180 ticks)
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

    # players -> mark enemies (channel 5). 'self' channel 4 will be set by run() when matching player_id.
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
    # map discrete action index to protocol char; use 'stay' if unknown
    table = {0: 'stay', 1: 'u', 2: 'd', 3: 'l', 4: 'r', 5: 'b'}
    action_char = table.get(int(a), 'stay')
    return {'type': 'control', 'data': {'action': action_char}}


def encode_control_ghost(x: int, y: int) -> dict:
    return {'type': 'control_ghost', 'data': {'action': {'x': int(x), 'y': int(y)}}}


# -------------------
# Main run loop
# -------------------
async def run(uri: str, policy_path: str, algo: str, player_id: str, game_id: str, team_id: str, player_name: str,
              team_name: str):
    # prepare agent
    if algo.lower() == 'dqn':
        agent = DQNAgent(in_channels=6, n_actions=ACTIONS, env_h=18, env_w=28)
        agent.load(policy_path)
        select_action = lambda obs: agent.act(obs, greedy=True)
    else:
        agent = PPOAgent(in_channels=6, n_actions=ACTIONS, H=18, W=28)
        # assume policy_path is actor path; critic path expected next to it with 'critic' in name
        try:
            agent.load(policy_path, policy_path.replace('actor', 'critic'))
        except Exception:
            # if load fails, try load actor only (user may supply different naming)
            try:
                agent.load(policy_path, policy_path)
            except Exception:
                pass

        def select_action(obs):
            a, _, _ = agent.get_action(obs)
            return a

    # connect and join
    reconnect_delay = 3
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print(f"[ws_inference] Connected to {uri}")
                # send join_game
                join_msg = {"type": "join_game",
                    "data": {"gameId": game_id, "teamId": team_id, "playerId": player_id, "playerName": player_name,
                        "teamName": team_name, "role": "player"}}
                await ws.send(json.dumps(join_msg))
                print("[ws_inference] Sent join_game")

                # listen loop
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    # handle join success / errors from server
                    if msg.get('type') == 'join_success':
                        print("[ws_inference] join_success:", msg.get('data', {}))
                        continue
                    if msg.get('type') == 'join_error':
                        print("[ws_inference] join_error:", msg.get('data', {}))
                        # do not immediately bail — maybe credentials wrong
                        continue

                    # If message is a game event, act
                    if is_game_event_type(msg.get('tag') or msg.get('type') or ''):
                        # Build observation
                        obs = decode_server_state(msg)
                        # mark self in channel 4 (if players present)
                        if 'players' in msg and isinstance(msg['players'], list):
                            for p in msg['players']:
                                try:
                                    if p.get('id') == player_id:
                                        cell = pos_to_cell(p.get('p', p.get('position', {})))
                                        if cell:
                                            cx, cy = cell
                                            # guard bounds
                                            H = obs.shape[1]
                                            W = obs.shape[2]
                                            if 0 <= cy < H and 0 <= cx < W:
                                                obs[4, cy, cx] = 1.0
                                                obs[5, cy, cx] = 0.0
                                except Exception:
                                    continue

                        # determine whether we're alive or dead (if players data contains status)
                        am_dead = False
                        if 'players' in msg and isinstance(msg['players'], list):
                            for p in msg['players']:
                                if p.get('id') == player_id:
                                    status = p.get('s') or p.get('status') or ''
                                    if status == 'dead':
                                        am_dead = True
                                    break

                        if am_dead:
                            # ghost action: simple heuristic — move to (1,1) or try rescue behavior
                            ghost_target = None
                            # try to rescue teammate: find nearest dying teammate in payload
                            if 'players' in msg:
                                best = None
                                for p in msg['players']:
                                    st = p.get('s') or p.get('status')
                                    if st == 'dying' and p.get('id') != player_id:
                                        cell = pos_to_cell(p.get('p', p.get('position', {})))
                                        if cell:
                                            best = cell
                                            break
                                if best:
                                    ghost_target = best
                            if not ghost_target:
                                # default roam near center
                                ghost_target = (max(1, obs.shape[2] // 2), max(1, obs.shape[1] // 2))  # (x,y)
                            out = encode_control_ghost(int(ghost_target[0]), int(ghost_target[1]))
                            await ws.send(json.dumps(out))
                        else:
                            # alive -> normal control
                            now = time.time()
                            a = select_action(obs)
                            out = encode_action_to_server(a)
                            print(f"[ws_inference] action {a} sent in {time.time()-now:.3f}s")
                            await ws.send(json.dumps(out))

                    # handle other message types if needed (game_over...)
                    elif msg.get('type') == 'game_over' or msg.get('tag') == 'game_over':
                        print("[ws_inference] game_over:", msg.get('data', {}))
                    else:
                        # ignore or print for debug
                        # print("[ws_inference] other msg:", msg.get('type') or msg.get('tag'))
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
    parser.add_argument('--uri', type=str, default="ws://171.251.51.213:5001", help="WebSocket server URI")
    parser.add_argument('--policy', type=str, required=True, help="Path to saved policy (DQN .pt or PPO actor .pt)")
    parser.add_argument('--algo', type=str, choices=['dqn', 'ppo'], default='dqn')
    parser.add_argument('--player-id', type=str, default=None, help="Player ID (uuid). If omitted, auto-generated.")
    parser.add_argument('--player-name', type=str, default="PyBot", help="Player display name")
    parser.add_argument('--team-id', type=str, default=None, help="Team ID")
    parser.add_argument('--team-name', type=str, default="PyTeam", help="Team name")
    parser.add_argument('--game-id', type=str, default=None, help="Game ID (if required)")
    args = parser.parse_args()

    player_id = args.player_id or gen_uuid()
    print(f"[ws_inference] using player_id={player_id}")

    asyncio.run(
        run(args.uri, args.policy, args.algo, player_id, args.game_id, args.team_id, args.player_name, args.team_name))


if __name__ == '__main__':
    main()
