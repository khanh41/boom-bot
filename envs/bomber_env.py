import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Discrete actions
# 0: stay, 1: up, 2: down, 3: left, 4: right, 5: place_bomb
ACTIONS = 6

@dataclass
class Bomb:
    x: int
    y: int
    fuse: int   # steps to explode
    owner: int  # player id

class BomberEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, grid_size: int = 11, max_bombs: int = 3, max_steps: int = 300, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.grid_size = grid_size
        self.max_bombs = max_bombs
        self.max_steps = max_steps

        # Build observation space
        # Layers: 0 empty/solid/crate, 1 bombs fuse, 2 flames (ttl), 3 items, 4 self, 5 enemies
        self.obs_shape = (6, grid_size, grid_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(ACTIONS)

        # Static map: 0 empty, 1 solid wall, 2 crate
        self.solid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self._gen_static_map()
        self.reset()

    def _gen_static_map(self):
        # outer walls solid
        self.solid[0,:] = 1
        self.solid[-1,:] = 1
        self.solid[:,0] = 1
        self.solid[:,-1] = 1
        # add internal unbreakable pillars (like classic Bomberman)
        for i in range(2, self.grid_size-1, 2):
            for j in range(2, self.grid_size-1, 2):
                self.solid[i,j] = 1
        # crates random
        self.crates = np.zeros_like(self.solid)
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                if self.solid[i,j] == 0 and self.rng.random() < 0.35:
                    self.crates[i,j] = 1

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.player_pos = self._spawn_safe()
        self.enemies = [self._spawn_safe(exclude=[self.player_pos])]
        self.bombs: list[Bomb] = []
        self.flames = np.zeros_like(self.solid, dtype=np.int32)  # flame ttl
        self.items = np.zeros_like(self.solid, dtype=np.int32)   # 1: range+, 2: speed (placeholder)
        self.bomb_range = 3
        self.alive = True
        obs = self._encode_obs()
        info = {}
        return obs, info

    def _spawn_safe(self, exclude: list[Tuple[int,int]] | None = None) -> Tuple[int,int]:
        exclude = exclude or []
        candidates = []
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                if self.solid[i,j] == 0 and self.crates[i,j] == 0 and (i,j) not in exclude:
                    candidates.append((i,j))
        return candidates[self.rng.integers(0, len(candidates))]

    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.max_steps

        # move
        x, y = self.player_pos
        dx, dy = {0:(0,0),1:(-1,0),2:(1,0),3:(0,-1),4:(0,1),5:(0,0)}[int(action)]
        nx, ny = x + dx, y + dy
        if self._is_free(nx, ny):
            self.player_pos = (nx, ny)

        # place bomb
        if action == 5 and len(self.bombs) < self.max_bombs:
            if not any(b.x == x and b.y == y for b in self.bombs):
                self.bombs.append(Bomb(x=x, y=y, fuse=8, owner=0))

        # enemy simple policy: random safe move
        self._enemy_act()

        # tick bombs and apply explosions
        self._tick_bombs()

        # items pickup
        px, py = self.player_pos
        if self.items[px,py] == 1:
            self.bomb_range = min(self.bomb_range+1, 6)
            self.items[px,py] = 0
            reward += 2.0

        # reward shaping
        reward += 0.01  # survive bonus
        if not self.alive:
            terminated = True
            reward -= 5.0

        obs = self._encode_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def _is_free(self, i, j):
        if i < 0 or j < 0 or i >= self.grid_size or j >= self.grid_size:
            return False
        if self.solid[i,j] == 1:
            return False
        if self.crates[i,j] == 1:
            return False
        if self.flames[i,j] > 0:
            return False
        return True

    def _enemy_act(self):
        ex, ey = self.enemies[0]
        # random safe move
        moves = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
        self.rng.shuffle(moves)
        for dx,dy in moves:
            nx, ny = ex+dx, ey+dy
            if self._is_free(nx, ny):
                self.enemies[0] = (nx, ny)
                break

    def _tick_bombs(self):
        # decrement fuse
        for b in self.bombs:
            b.fuse -= 1
        # explode bombs
        new_bombs = []
        for b in self.bombs:
            if b.fuse <= 0:
                self._explode(b.x, b.y, self.bomb_range)
            else:
                new_bombs.append(b)
        self.bombs = new_bombs
        # decay flames
        self.flames = np.maximum(self.flames-1, 0)

    def _explode(self, x, y, rng):
        self.flames[x,y] = 3
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            for k in range(1, rng+1):
                nx, ny = x + dx*k, y + dy*k
                if nx < 0 or ny < 0 or nx >= self.grid_size or ny >= self.grid_size:
                    break
                if self.solid[nx,ny] == 1:
                    break
                self.flames[nx,ny] = 3
                if self.crates[nx,ny] == 1:
                    self.crates[nx,ny] = 0
                    # chance to spawn item
                    if np.random.random() < 0.2:
                        self.items[nx,ny] = 1
                    break

        # check damage
        if tuple(self.player_pos) == (x,y) or self.flames[self.player_pos] > 0:
            self.alive = False

    def _encode_obs(self):
        H = np.zeros(self.obs_shape, dtype=np.float32)
        # 0: static (solid=1, crate=0.5, empty=0)
        H[0] = (self.solid == 1).astype(np.float32) + 0.5*(self.crates==1).astype(np.float32)
        # 1: bombs (normalized fuse)
        bomb_map = np.zeros_like(self.solid, dtype=np.float32)
        for b in self.bombs:
            bomb_map[b.x,b.y] = max(b.fuse, 0)/8.0
        H[1] = bomb_map
        # 2: flames (ttl normalized)
        H[2] = self.flames.astype(np.float32)/3.0
        # 3: items
        H[3] = (self.items>0).astype(np.float32)
        # 4: self
        self_map = np.zeros_like(self.solid, dtype=np.float32)
        self_map[self.player_pos] = 1.0
        H[4] = self_map
        # 5: enemies
        e_map = np.zeros_like(self.solid, dtype=np.float32)
        for (ex,ey) in self.enemies:
            e_map[ex,ey]=1.0
        H[5] = e_map
        return H
