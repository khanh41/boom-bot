from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
    'u': (-1, 0),
    'd': (1, 0),
    'l': (0, -1),
    'r': (0, 1),
    'b': (0, 0),  # place bomb
    'k': (0, 0),  # kick bomb (future)
}

EXPLOSION_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

@dataclass
class Bomb:
    x: int
    y: int
    fuse: int   # countdown ticks
    owner: int  # player id
    is_exploding_soon: bool = False

class BomberEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_w: int = 28, grid_h: int = 18, max_bombs: int = 1, max_steps: int = 3000, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.max_bombs = max_bombs
        self.max_steps = max_steps

        # Player config
        self.player_speed = 0.5
        self.invincible_ticks = 0
        self.dying_ticks = 0
        self.is_stunned = False
        self.stun_ticks = 0
        self.bomb_range = 1

        # Map
        self.solid = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self._gen_static_map()

        # Observation space: layers [0:tiles,1:bombs,2:flames,3:items,4:self,5:enemies]
        self.obs_shape = (6, self.grid_h, self.grid_w)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_TO_DIR))

        self.reset()

    def _gen_static_map(self):
        self.solid.fill(TileType.EMPTY)
        # outer walls
        self.solid[0,:] = TileType.WALL
        self.solid[-1,:] = TileType.WALL
        self.solid[:,0] = TileType.WALL
        self.solid[:,-1] = TileType.WALL

        # internal pillars
        for i in range(2, self.grid_h-1, 2):
            for j in range(2, self.grid_w-1, 2):
                self.solid[i,j] = TileType.WALL

        # crates
        self.crates = np.zeros_like(self.solid)
        for i in range(1, self.grid_h-1):
            for j in range(1, self.grid_w-1):
                if self.solid[i,j] == TileType.EMPTY and self.rng.random() < 0.35:
                    self.crates[i,j] = 1

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.player_pos = self._spawn_safe()
        self.enemies = [self._spawn_safe(exclude=[self.player_pos])]
        self.bombs: List[Bomb] = []
        self.flames = np.zeros_like(self.solid, dtype=np.int32)
        self.items = np.zeros_like(self.solid, dtype=np.int32)
        self.player_speed = 0.5
        self.invincible_ticks = 0
        self.dying_ticks = 0
        self.is_stunned = False
        self.stun_ticks = 0
        self.bomb_range = 1
        self.alive = True
        obs = self._encode_obs()
        return obs, {}

    def _spawn_safe(self, exclude: List[Tuple[int,int]] | None = None) -> Tuple[int,int]:
        exclude = exclude or []
        candidates = []
        for i in range(1, self.grid_h-1):
            for j in range(1, self.grid_w-1):
                if self.solid[i,j] == TileType.EMPTY and self.crates[i,j]==0 and (i,j) not in exclude:
                    candidates.append((i,j))
        if not candidates:
            return (1,1)
        return candidates[self.rng.integers(0, len(candidates))]

    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        action_key = list(ACTION_TO_DIR.keys())[action]
        dx, dy = ACTION_TO_DIR[action_key]
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        # move
        if self._is_free(nx, ny):
            self.player_pos = (nx, ny)

        # place bomb
        if action_key == 'b' and len(self.bombs) < self.max_bombs:
            if not any(b.x==x and b.y==y for b in self.bombs):
                self.bombs.append(Bomb(x=x, y=y, fuse=180, owner=0))
                reward += 0.2  # reward for placing bomb

        # enemy simple move
        self._enemy_act()

        # tick bombs
        reward += self._tick_bombs()  # updated to return reward

        # item pickup
        px, py = self.player_pos
        item = self.items[px, py]
        if item != 0:
            if item == ItemType.BOMB_UP:
                self.max_bombs = min(self.max_bombs + 1, 8)
            elif item == ItemType.POWER_UP:
                self.bomb_range = min(self.bomb_range + 1, 8)
            elif item == ItemType.SPEED_UP:
                self.player_speed = min(self.player_speed + 0.25, 3.5)
            self.items[px, py] = 0
            reward += 5.0

        # survive bonus
        reward += 0.01
        if not self.alive:
            terminated = True
            reward -= 50.0

        obs = self._encode_obs()
        return obs, reward, terminated, truncated, {}

    def _is_free(self, i, j):
        if i < 0 or j < 0 or i >= self.grid_h or j >= self.grid_w:
            return False
        tile = self.solid[i,j]
        if tile == TileType.WALL:
            return False
        if tile == TileType.BRICK and self.crates[i,j]==1:
            return False
        if self.flames[i,j] > 0:
            return False
        return True

    def _enemy_act(self):
        ex, ey = self.enemies[0]
        moves = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
        self.rng.shuffle(moves)
        for dx,dy in moves:
            nx, ny = ex+dx, ey+dy
            if self._is_free(nx, ny):
                self.enemies[0] = (nx, ny)
                break

    def _tick_bombs(self):
        reward = 0.0
        for b in self.bombs:
            b.fuse -= 1
            b.is_exploding_soon = b.fuse <= 60
        new_bombs = []
        for b in self.bombs:
            if b.fuse <= 0:
                reward += self._explode(b.x, b.y, self.bomb_range)
            else:
                new_bombs.append(b)
        self.bombs = new_bombs

        # decay flames
        self.flames = np.maximum(self.flames - 1, 0)

        # decrease invincibility/stun/dying ticks
        if self.invincible_ticks > 0: self.invincible_ticks -= 1
        if self.stun_ticks > 0: self.stun_ticks -= 1
        if self.dying_ticks > 0: self.dying_ticks -= 1
        if self.dying_ticks > 0: self.alive = False

        return reward

    def _explode(self, x, y, rng):
        reward = 0.0
        self.flames[x, y] = 30

        # center damage (check enemy)
        for idx, (ex, ey) in enumerate(self.enemies):
            if (ex, ey) == (x, y) or self.flames[ex, ey] > 0:
                reward += 20.0  # enemy killed
                self.enemies[idx] = self._spawn_safe()  # respawn enemy

        # bricks and explosion
        for dx, dy in EXPLOSION_DIRS:
            for k in range(1, rng + 1):
                nx, ny = x + dx * k, y + dy * k
                if nx < 0 or ny < 0 or nx >= self.grid_h or ny >= self.grid_w:
                    break
                if self.solid[nx, ny] == TileType.WALL:
                    break
                self.flames[nx, ny] = 30
                if self.crates[nx, ny] == 1:
                    self.crates[nx, ny] = 0
                    reward += 1.0  # reward for destroying brick
                    r = self.rng.random()
                    if r < 0.15:
                        self.items[nx, ny] = ItemType.BOMB_UP
                    elif r < 0.3:
                        self.items[nx, ny] = ItemType.POWER_UP
                    elif r < 0.4:
                        self.items[nx, ny] = ItemType.SPEED_UP
                    break

        # damage player
        px, py = self.player_pos
        if (px, py) == (x, y) or self.flames[px, py] > 0:
            self.alive = False

        return reward

    def _encode_obs(self):
        H = np.zeros(self.obs_shape, dtype=np.float32)
        # tiles
        H[0] = (self.solid==TileType.WALL).astype(np.float32) + 0.5*(self.crates==1).astype(np.float32)
        # bombs
        bomb_map = np.zeros_like(self.solid, dtype=np.float32)
        for b in self.bombs:
            bomb_map[b.x,b.y] = max(b.fuse,0)/180.0
        H[1] = bomb_map
        # flames
        H[2] = self.flames.astype(np.float32)/30.0
        # items
        H[3] = (self.items>0).astype(np.float32)
        # self
        self_map = np.zeros_like(self.solid, dtype=np.float32)
        self_map[self.player_pos] = 1.0
        H[4] = self_map
        # enemies
        e_map = np.zeros_like(self.solid, dtype=np.float32)
        for ex,ey in self.enemies:
            e_map[ex,ey] = 1.0
        H[5] = e_map
        return H
