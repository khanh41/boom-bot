import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import gymnasium
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


# Discrete actions: 0: stay, 1: up, 2: down, 3: left, 4: right, 5: place_bomb, 6: kick_bomb
ACTIONS = 7

# Directions
ACTION_TO_DIR = {
    0: (0, 0),  # stay
    1: (0, -1),  # up
    2: (0, 1),  # down
    3: (-1, 0),  # left
    4: (1, 0),  # right
    5: (0, 0),  # place_bomb
    6: (0, 0),  # kick_bomb
}

EXPLOSION_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


@dataclass
class Bomb:
    x: int
    y: int
    timer: int
    power: int
    owner: str
    is_exploding_soon: bool = False
    is_moving: bool = False
    move_direction: Tuple[int, int] = (0, 0)
    move_distance_left: float = 0.0


@dataclass
class PlayerState:
    position: Tuple[float, float]
    alive: bool
    status: str  # 'alive', 'dying', 'dead'
    speed: float
    bomb_limit: int
    bombs_placed: int
    bomb_power: int
    dying_ticks: int
    invincibility_ticks: int
    is_stunned: bool
    stun_ticks: int
    score: int
    stuck_counter: int = 0  # counter to detect if stuck


class BomberEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi"], "name": "single_agent_bomber_env_v0"}

    def __init__(self, grid_w: int = 28, grid_h: int = 18, max_steps: int = 3000, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        # self.player_id = random.randint(0, 3)
        self.player_id = 0
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.max_steps = max_steps

        # --- Tick & timing ---
        self.tick_rate = 16  # ms per tick (~60 ticks per second)
        self.player_tick_rate = 200 // self.tick_rate  # ~200 ms per action

        # Time-based constants (converted to ticks)
        self.dying_time = 5000 // self.tick_rate  # ~5s
        self.invincibility_ticks = 2000 // self.tick_rate  # ~2s
        self.ghost_stun_duration = 2000 // self.tick_rate  # ~2s
        self.bomb_fuse_ticks = 3000 // self.tick_rate  # ~3s
        self.bomb_exploding_soon_ticks = 1000 // self.tick_rate  # ~1s warning
        self.explosion_lifetime = 500 // self.tick_rate  # ~0.5s

        self.bomb_slide_speed = 3.0  # still per tick distance

        self.item_spawn_chance = {
            ItemType.BOMB_UP: 0.15,
            ItemType.POWER_UP: 0.15,
            ItemType.SPEED_UP: 0.1
        }

        # Observation shape (H, W, C)
        self.obs_shape = (self.grid_h, self.grid_w, 6)

        # Agents, teams
        self.agents = [f"player_{i}" for i in range(4)]
        self.possible_agents = self.agents[:]
        self.teams = {
            "player_0": "A",
            "player_1": "A",
            "player_2": "B",
            "player_3": "B"
        }
        self.agent_timers = {a: 0 for a in self.agents}

        self.render_mode = "ansi"

        # States
        self.players: Dict[str, PlayerState] = {}
        self.bombs: List[Bomb] = []
        self.flames = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.items = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.map = None
        self.steps = 0

        # Spaces
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=self.obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTIONS)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.bombs.clear()
        self.flames.fill(0)
        self.items.fill(0)
        self.map = self._gen_random_map()
        self.players = {
            "player_0": PlayerState((1, 1), True, "alive", 0.5, 1, 0, 1, 0, 0, False, 0, 0),
            "player_1": PlayerState((self.grid_w - 2, self.grid_h - 2), True, "alive", 0.5, 1, 0, 1, 0, 0, False, 0, 0),
            "player_2": PlayerState((1, self.grid_h - 2), True, "alive", 0.5, 1, 0, 1, 0, 0, False, 0, 0),
            "player_3": PlayerState((self.grid_w - 2, 1), True, "alive", 0.5, 1, 0, 1, 0, 0, False, 0, 0)
        }
        self.agent_timers = {a: 0 for a in self.agents}
        obs_dict = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        player_name = f"player_{self.player_id}"
        return obs_dict[player_name], infos[player_name]

    def step(self, action: int):
        self.steps += 1

        rewards = {a: 0.0 for a in self.agents}

        player_name = f"player_{self.player_id}"
        rewards[player_name] += self._process_player_action(action, player_name)

        # Process other players with random move actions (for simplicity)
        for i in range(4):
            if i == self.player_id:
                continue

            other_player_name = f"player_{i}"
            if self.players[other_player_name].status == "alive":
                other_action = random.randint(0, ACTIONS - 3)
                rewards[other_player_name] += self._process_player_action(other_action, other_player_name)

        # --- Bomb updates ---
        new_bombs = []
        for b in self.bombs:
            b.timer -= 1
            b.is_exploding_soon = b.timer <= self.bomb_exploding_soon_ticks
            if b.timer <= 0:
                rewards = self._explode(b, rewards)
                self.players[b.owner].bombs_placed -= 1
            else:
                new_bombs.append(b)
        self.bombs = new_bombs

        # Decay flames
        self.flames = np.maximum(self.flames - 1, 0)

        # Process item pickups
        for a, player in self.players.items():
            if player.status != "alive":
                continue
            x, y = player.position
            if self.items[y, x] != 0:
                item_type = self.items[y, x]
                if item_type == ItemType.BOMB_UP:
                    player.bomb_limit = min(player.bomb_limit + 1, 8)
                elif item_type == ItemType.POWER_UP:
                    player.bomb_power = min(player.bomb_power + 1, 8)
                elif item_type == ItemType.SPEED_UP:
                    player.speed = min(player.speed + 0.25, 3.5)
                self.items[y, x] = 0
                print(f"🎁 {a} picked up item {item_type} at {(x, y)}")
                rewards[a] += 1.0

        # --- Check flames / dying ---
        dead_this_step = []
        for a, player in self.players.items():
            if player.status == "alive" and self.flames[player.position[1], player.position[0]] > 0:
                player.status = "dying"
                player.dying_ticks = self.dying_time // self.tick_rate
                dead_this_step.append(a)

        # Update timers
        for a, player in self.players.items():
            if player.dying_ticks > 0:
                player.dying_ticks -= 1
                if player.dying_ticks <= 0:
                    print(f"☠️ {a} is dead!")
                    player.status = "dead"
                    player.alive = False
            if player.invincibility_ticks > 0:
                player.invincibility_ticks -= 1
            if player.stun_ticks > 0:
                player.stun_ticks -= 1
                player.is_stunned = player.stun_ticks > 0

        # Survival bonus
        for a, player in self.players.items():
            if player.status == "alive":
                rewards[a] += 0.0001

        # Terminations & truncations
        terminations = {a: self.players[a].status == "dead" for a in self.agents}
        truncations = {a: self.steps >= self.max_steps for a in self.agents}
        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {"score": self.players[a].score} for a in self.agents}

        # Check if only one team remains
        alive_teams = {self.teams[a] for a in self.agents if self.players[a].status == "alive"}
        if len(alive_teams) == 1:
            winning_team = alive_teams.pop()
            for a in self.agents:
                if self.teams[a] == winning_team:
                    rewards[a] += 200.0

            terminations = {a: True for a in self.agents}
            print(f"🏆 Team {winning_team} wins!")

        # Aggregate rewards, terminations, truncations for main player
        ob = obs[player_name]
        reward = rewards[player_name]
        termination = terminations[player_name]
        truncation = truncations[player_name]
        info = infos[player_name]

        return ob, reward, termination, truncation, info

    def _process_player_action(self, action: int, player_name: str):
        player = self.players[player_name]
        reward = 0

        # Increase agent timer
        self.agent_timers[player_name] += 1

        if self.agent_timers[player_name] >= self.player_tick_rate:
            # Agent can act
            x, y = player.position

            if action == 5:  # Place bomb
                x, y = int(x), int(y)
                if player.bombs_placed < player.bomb_limit and not any(b.x == x and b.y == y for b in self.bombs):
                    self.bombs.append(Bomb(x, y, self.bomb_fuse_ticks, player.bomb_power, player_name))
                    player.bombs_placed += 1
                    reward += 0.05
            elif action == 0:
                # Penalty for idling 10 seconds
                player.stuck_counter += 1
                if player.stuck_counter >= (10000 // self.tick_rate):
                    reward -= 0.5
            else:  # Move
                dx, dy = ACTION_TO_DIR[action]
                nx, ny = x + dx, y + dy
                if self._is_free(nx, ny):
                    player.position = (nx, ny)
                    reward += 0.05
                    self.agent_timers[player_name] = 0

        return reward

    def _encode_obs(self, agent_id):
        H = np.zeros((6, self.grid_h, self.grid_w), dtype=np.float32)

        # Map: walls (1.0), bricks (0.5), empty (0.0)
        H[0] = (self.map == TileType.WALL).astype(np.float32) + 0.5 * (self.map == TileType.BRICK).astype(np.float32)

        # Bombs: b.y and b.x are the coordinates of the bomb
        # b.timer / self.bomb_fuse_ticks gives a value between 0-1, 1 means just placed, 0 means about to explode
        for b in self.bombs:
            H[1, b.y, b.x] = b.timer / self.bomb_fuse_ticks

        # Flames
        H[2] = np.clip(self.flames / self.explosion_lifetime, 0, 1)

        # Items
        H[3] = (self.items > 0).astype(np.float32)

        # Self
        if self.players[agent_id].status != "dead":
            x, y = self.players[agent_id].position
            H[4, y, x] = 1.0

        # Others
        for a in self.agents:
            if a != agent_id and self.players[a].status != "dead":
                x, y = self.players[a].position
                H[5, y, x] = 1.0 if self.teams[a] != self.teams[agent_id] else 0.5

        # Chuyển sang (H, W, C)
        obs = np.transpose(H, (1, 2, 0))
        return obs.astype(np.float32)

    def _gen_random_map(self):
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

        # Wall and Bricks random
        for y in range(1, self.grid_h - 1):
            for x in range(1, self.grid_w - 1):
                # 60% empty, 30% brick, 10% wall
                r = self.rng.random()
                if r < 0.6:
                    grid[y, x] = TileType.EMPTY
                elif r < 0.9:
                    grid[y, x] = TileType.BRICK
                else:
                    grid[y, x] = TileType.WALL

        # Outer walls
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = TileType.WALL

        # Clear spawn points
        spawn_points = [(1, 1), (self.grid_w - 2, self.grid_h - 2), (1, self.grid_h - 2), (self.grid_w - 2, 1)]
        for x, y in spawn_points:
            grid[y, x] = TileType.EMPTY
            # Clear surrounding tiles to ensure agents are not trapped
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 < ny < self.grid_h - 1 and 0 < nx < self.grid_w - 1:
                        grid[ny, nx] = TileType.EMPTY

        return grid

    def _is_free(self, x, y):
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return False
        if self.map[y, x] in (TileType.WALL, TileType.BRICK):
            return False
        if any(b.x == x and b.y == y for b in self.bombs):
            return False
        return True

    def _is_danger(self, x, y):
        # Nếu trong flame thì chắc chắn nguy hiểm
        if self.flames[y, x] > 0:
            return True

        # Nếu trong vùng bom sắp nổ
        for b in self.bombs:
            if not b.is_exploding_soon:
                continue
            # bom chính
            if b.x == x and b.y == y:
                return True
            # vùng ảnh hưởng theo power
            for dx, dy in EXPLOSION_DIRS:
                for i in range(1, b.power + 1):
                    nx, ny = b.x + dx * i, b.y + dy * i
                    if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
                        break
                    if self.map[ny, nx] == TileType.WALL:
                        break
                    if nx == x and ny == y:
                        return True
                    if self.map[ny, nx] == TileType.BRICK:
                        break
        return False

    def handle_cell(self, nx, ny, bomb: Bomb, rewards: Dict[str, float]):
        # Nếu là item -> phá hủy
        if self.items[ny, nx] != 0:
            print(f"💥 Item at {(nx, ny)} destroyed by bomb {bomb.owner}")
            self.items[ny, nx] = 0
            rewards[bomb.owner] -= 0.05  # phạt khi phá item

        # Nếu có bom khác -> chain reaction
        for other_b in list(self.bombs):  # copy để tránh modify khi iterate
            if other_b.x == nx and other_b.y == ny:
                print(f"💣 Chain reaction triggered at {(nx, ny)} by {bomb.owner}")
                self.bombs.remove(other_b)
                self.players[other_b.owner].bombs_placed -= 1
                rewards = self._explode(other_b, rewards)
                break
        return rewards

    def _explode(self, bomb: Bomb, rewards: Dict[str, float]):
        x, y, p = bomb.x, bomb.y, bomb.power
        self.flames[y, x] = self.explosion_lifetime
        for dx, dy in EXPLOSION_DIRS:
            brick_destroyed = 0
            for i in range(1, p + 1):
                nx, ny = x + dx * i, y + dy * i
                if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
                    break
                if self.map[ny, nx] == TileType.WALL:
                    rewards[bomb.owner] -= 0.05
                    break

                self.flames[ny, nx] = self.explosion_lifetime
                rewards = self.handle_cell(nx, ny, bomb, rewards)

                if self.map[ny, nx] == TileType.BRICK:
                    brick_destroyed += 1
                    print(f"🔥 Bomb by {bomb.owner} destroyed brick at {(nx, ny)}")
                    self.map[ny, nx] = TileType.EMPTY
                    r = self.rng.random()
                    if r < self.item_spawn_chance[ItemType.BOMB_UP]:
                        self.items[ny, nx] = ItemType.BOMB_UP
                    elif r < self.item_spawn_chance[ItemType.BOMB_UP] + self.item_spawn_chance[ItemType.POWER_UP]:
                        self.items[ny, nx] = ItemType.POWER_UP
                    elif r < sum(self.item_spawn_chance.values()):
                        self.items[ny, nx] = ItemType.SPEED_UP
                    break

                for other_a, other_p in self.players.items():
                    if other_p.status == "alive" and (other_p.position == (nx, ny) or (other_p.position == (x, y))):
                        if other_a != bomb.owner:
                            rewards[bomb.owner] += 10.0
                        else:
                            rewards[bomb.owner] -= 10.0

            rewards[bomb.owner] += (brick_destroyed ** 1.5) * 1.0

        return rewards

    def render(self):
        if self.render_mode == "ansi":
            grid = np.full((self.grid_h, self.grid_w), '.', dtype=str)
            # Draw walls and bricks
            grid[self.map == TileType.WALL] = '#'
            grid[self.map == TileType.BRICK] = '%'
            # Draw bombs
            for b in self.bombs:
                grid[b.y, b.x] = 'B'
            # Draw flames
            grid[self.flames > 0] = '*'
            # Draw items
            for y in range(self.grid_h):
                for x in range(self.grid_w):
                    if self.items[y, x] == ItemType.BOMB_UP:
                        grid[y, x] = 'b'
                    elif self.items[y, x] == ItemType.POWER_UP:
                        grid[y, x] = 'p'
                    elif self.items[y, x] == ItemType.SPEED_UP:
                        grid[y, x] = 's'
            # Draw players
            for a, player in self.players.items():
                if player.status != "dead":
                    x, y = player.position
                    grid[y, x] = a[7]  # Player number (0, 1, 2, 3)
            # Convert to string
            return '\n'.join(''.join(row) for row in grid)
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")
