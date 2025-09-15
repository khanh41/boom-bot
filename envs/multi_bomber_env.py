from dataclasses import dataclass
from typing import Dict, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


# Tile types
class TileType:
    EMPTY = 0
    WALL = 1
    BRICK = 2


# Item types
class ItemType:
    BOMB_UP = 1
    POWER_UP = 2
    SPEED_UP = 3


# Discrete actions
# 0: stay, 1: up, 2: down, 3: left, 4: right, 5: place_bomb
ACTIONS = 6

# Directions
ACTION_TO_DIR = {
    0: (0, 0),  # stay
    1: (0, -1),  # up
    2: (0, 1),  # down
    3: (-1, 0),  # left
    4: (1, 0),  # right
    5: (0, 0)  # bomb
}

EXPLOSION_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


@dataclass
class Bomb:
    x: int
    y: int
    timer: int
    power: int
    owner: str


# ===== Multi-Agent Env (2 team) =====
class MultiBomberEnv(ParallelEnv):
    metadata = {"render_modes": ["ansi"], "name": "multi_bomber_team_v0"}

    def __init__(self, grid_w: int = 28, grid_h: int = 18, max_steps: int = 3000, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.max_steps = max_steps

        self.obs_shape = (6, self.grid_h, self.grid_w)

        self.agents = [f"player_{i}" for i in range(4)]
        self.possible_agents = self.agents[:]
        self.teams = {
            "player_0": "A",
            "player_1": "A",
            "player_2": "B",
            "player_3": "B"
        }

        # States
        self.pos: Dict[str, Tuple[int, int]] = {}
        self.alive: Dict[str, bool] = {}
        self.bombs: List[Bomb] = []
        self.flames = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.items = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.map = None
        self.steps = 0

    @property
    def observation_spaces(self):
        return {
            a: spaces.Box(
                low=0,
                high=1,
                shape=self.obs_shape,
                dtype=np.float32,
            )
            for a in self.agents
        }

    @property
    def action_spaces(self):
        return {a: spaces.Discrete(ACTIONS) for a in self.agents}

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.bombs.clear()
        self.flames.fill(0)
        self.items.fill(0)
        self.map = self._gen_static_map()
        self.pos = {
            "player_0": (1, 1),
            "player_1": (self.grid_w - 2, self.grid_h - 2),
            "player_2": (1, self.grid_h - 2),
            "player_3": (self.grid_w - 2, 1),
        }
        self.alive = {a: True for a in self.agents}
        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        self.steps += 1
        rewards = {a: 0.0 for a in self.agents}

        # Movement + bomb placing
        for a, act in actions.items():
            if not self.alive[a]:
                continue

            x, y = self.pos[a]

            if act == 5:  # place bomb
                if not any(b.x == x and b.y == y for b in self.bombs):
                    self.bombs.append(Bomb(x, y, 30, 3, a))
            else:
                dx, dy = ACTION_TO_DIR[act]
                nx, ny = x + dx, y + dy
                if self._is_free(nx, ny):
                    self.pos[a] = (nx, ny)

        # Bomb ticking
        new_bombs = []
        for b in self.bombs:
            b.timer -= 1
            if b.timer <= 0:
                self._explode(b)
            else:
                new_bombs.append(b)
        self.bombs = new_bombs

        # Flames decay
        self.flames[self.flames > 0] -= 1

        # Kills and rewards
        dead_this_step = []
        for a in self.agents:
            if self.alive[a]:
                x, y = self.pos[a]
                if self.flames[y, x] > 0:
                    self.alive[a] = False
                    dead_this_step.append(a)

        # Team rewards
        for a in self.agents:
            if self.alive[a]:
                rewards[a] += 0.01  # survival bonus

        for d in dead_this_step:
            team = self.teams[d]
            opp_team = "A" if team == "B" else "B"
            # penalty for team of dead agent
            for a in self.agents:
                if self.teams[a] == team:
                    rewards[a] -= 30.0
                if self.teams[a] == opp_team and self.alive[a]:
                    rewards[a] += 30.0

        terminations = {a: not self.alive[a] for a in self.agents}
        truncations = {a: self.steps >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}
        obs = {a: self._encode_obs(a) for a in self.agents}
        return obs, rewards, terminations, truncations, infos

    def _encode_obs(self, agent_id):
        H = np.zeros(self.obs_shape, dtype=np.float32)

        # Map: walls/bricks
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self.map[y, x] == TileType.WALL:
                    H[0, y, x] = 1.0
                elif self.map[y, x] == TileType.BRICK:
                    H[0, y, x] = 0.5

        # Bombs
        for b in self.bombs:
            H[1, b.y, b.x] = b.timer / 30.0

        # Flames
        H[2] = np.clip(self.flames / 5.0, 0, 1)

        # Items
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self.items[y, x] != 0:
                    H[3, y, x] = 1.0

        # Self
        if self.alive[agent_id]:
            x, y = self.pos[agent_id]
            H[4, y, x] = 1.0

        # Others
        for a in self.agents:
            if a != agent_id and self.alive[a]:
                x, y = self.pos[a]
                H[5, y, x] = 1.0

        return H.flatten()

    def _gen_static_map(self):
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if x == 0 or y == 0 or x == self.grid_w - 1 or y == self.grid_h - 1:
                    grid[y, x] = TileType.WALL
                elif x % 2 == 0 and y % 2 == 0:
                    grid[y, x] = TileType.WALL
                elif self.rng.random() < 0.2:
                    grid[y, x] = TileType.BRICK
        for (x, y) in [(1, 1), (self.grid_w - 2, self.grid_h - 2), (1, self.grid_h - 2), (self.grid_w - 2, 1)]:
            grid[y, x] = TileType.EMPTY
        return grid

    def _is_free(self, x, y):
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return False
        if self.map[y, x] in (TileType.WALL, TileType.BRICK):
            return False
        if any(b.x == x and b.y == y for b in self.bombs):
            return False
        return True

    def _explode(self, bomb: Bomb):
        x, y, p = bomb.x, bomb.y, bomb.power
        self.flames[y, x] = 3
        for dx, dy in EXPLOSION_DIRS:
            for i in range(1, p + 1):
                nx, ny = x + dx * i, y + dy * i
                if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
                    break
                if self.map[ny, nx] == TileType.WALL:
                    break
                self.flames[ny, nx] = 3
                if self.map[ny, nx] == TileType.BRICK:
                    self.map[ny, nx] = TileType.EMPTY
                    break


# ---------- Wrapper for SB3 ----------
class SingleAgentWrapper(gym.Env):
    def __init__(self, env: MultiBomberEnv, learning_agent="player_0"):
        super().__init__()
        self.env = env
        self.learning_agent = learning_agent
        self.observation_space = self.env.observation_spaces[self.learning_agent]
        self.action_space = self.env.action_spaces[self.learning_agent]

    def reset(self, **kwargs):
        obs, infos = self.env.reset()
        return obs[self.learning_agent], infos[self.learning_agent]

    def step(self, action):
        actions = {a: self.env.action_spaces[a].sample() for a in self.env.agents}
        actions[self.learning_agent] = action
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return (
            obs[self.learning_agent],
            rewards[self.learning_agent],
            terms[self.learning_agent],
            truncs[self.learning_agent],
            infos[self.learning_agent],
        )
