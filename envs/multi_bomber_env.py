from dataclasses import dataclass
from typing import Dict, Tuple, List

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


class MultiBomberEnv(ParallelEnv):
    metadata = {"render_modes": ["ansi"], "name": "multi_bomber_team_v1"}

    def __init__(self, grid_w: int = 28, grid_h: int = 18, max_steps: int = 3000, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
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
        self._observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.obs_shape,
            dtype=np.uint8
        )
        self._action_space = spaces.Discrete(ACTIONS)

    def observation_space(self, agent: str):
        return self._observation_space

    def action_space(self, agent: str):
        return self._action_space

    @property
    def observation_spaces(self):
        return {a: self.observation_space(a) for a in self.agents}

    @property
    def action_spaces(self):
        return {a: self.action_space(a) for a in self.agents}

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
        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        self.steps += 1
        rewards = {a: 0.0 for a in self.agents}

        # --- Process agent actions if their tick elapsed ---
        for a, player in self.players.items():
            if player.status != "alive":
                continue

            # Increase agent timer
            self.agent_timers[a] += 1
            act = actions.get(a, 0)  # default stay

            if self.agent_timers[a] >= self.player_tick_rate:
                # Agent can act
                x, y = player.position
                speed = player.speed

                if act == 5:  # Place bomb
                    can_place = False
                    target_reason = None
                    x, y = int(x), int(y)

                    if player.bombs_placed < player.bomb_limit and not any(b.x == x and b.y == y for b in self.bombs):
                        # Check bricks or enemies in bomb range
                        for dx, dy in EXPLOSION_DIRS:
                            for i in range(1, player.bomb_power + 1):
                                nx, ny = x + dx * i, y + dy * i
                                if nx < 0 or nx >= self.grid_w or ny < 0 or ny >= self.grid_h:
                                    break
                                if self.map[ny, nx] == TileType.WALL:
                                    break
                                if self.map[ny, nx] == TileType.BRICK:
                                    can_place = True
                                    target_reason = "brick"
                                    break
                                # Check if enemy in line of fire
                                for other_a, other_p in self.players.items():
                                    if other_a != a and other_p.status == "alive" and (nx, ny) == (
                                            int(other_p.position[0]), int(other_p.position[1])):
                                        can_place = True
                                        target_reason = "enemy"
                                        break
                                if can_place:
                                    break
                            if can_place:
                                break

                        if can_place:
                            # Actually place bomb
                            self.bombs.append(Bomb(x, y, self.bomb_fuse_ticks, player.bomb_power, a))
                            player.bombs_placed += 1
                            rewards[a] += 2.0 if target_reason == "brick" else 5.0  # reward more if targeting enemy
                        else:
                            rewards[a] -= 1.0  # penalty for useless bomb

                elif act == 0:
                    rewards[a] -= 0.05  # small penalty for idling
                else:  # Move
                    dx, dy = ACTION_TO_DIR[act]
                    nx, ny = x + dx, y + dy
                    # TODO: Add speed
                    # nx, ny = x + dx * speed, y + dy * speed
                    if self._is_free(nx, ny):
                        # check danger
                        if self._is_danger(nx, ny):
                            rewards[a] -= 1.0  # phạt mạnh nếu bước vào vùng nguy hiểm
                        else:
                            # exploration bonus
                            if not hasattr(player, "visited"):
                                player.visited = set()
                            if (nx, ny) not in player.visited:
                                rewards[a] += 0.2
                                player.visited.add((nx, ny))

                            # nếu né khỏi vùng nguy hiểm (từ ô cũ ra an toàn)
                            if self._is_danger(int(x), int(y)) and not self._is_danger(nx, ny):
                                rewards[a] += 0.5  # thưởng thoát chết

                            player.position = (nx, ny)
                            rewards[a] += 0.05
                            self.agent_timers[a] = 0
                    else:
                        rewards[a] -= 0.2

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
                player.score += 20
                print(f"🎁 {a} picked up item {item_type} at {(x, y)}")
                rewards[a] += 20.0

        # --- Check flames / dying ---
        dead_this_step = []
        for a, player in self.players.items():
            if player.status == "alive" and self.flames[player.position[1], player.position[0]] > 0:
                player.status = "dying"
                player.dying_ticks = self.dying_time // self.tick_rate
                dead_this_step.append(a)
                rewards[a] -= 30.0

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

        # Ghost mode interactions (rescue / kill)
        # for a, player in self.players.items():
        #     if player.status == "dead":
        #         for other_a, other_p in self.players.items():
        #             if other_a != a and other_p.status == "dying" and other_p.position == player.position:
        #                 if self.teams[a] == self.teams[other_a]:  # Rescue teammate
        #                     other_p.status = "alive"
        #                     other_p.dying_ticks = 0
        #                     other_p.invincibility_ticks = self.invincibility_ticks
        #                     rewards[a] += 100.0
        #                     rewards[other_a] += 150.0
        #                 else:  # Kill enemy
        #                     other_p.status = "dead"
        #                     other_p.alive = False
        #                     other_p.dying_ticks = 0
        #                     rewards[a] += 200.0
        #                     for team_a in self.agents:
        #                         if self.teams[team_a] == self.teams[a] and team_a != a:
        #                             rewards[team_a] += 100.0

        # Team rewards for deaths
        for a in dead_this_step:
            team = self.teams[a]
            opp_team = "A" if team == "B" else "B"
            for other_a in self.agents:
                if self.players[other_a].status == "alive":
                    if self.teams[other_a] == opp_team:
                        rewards[other_a] += 30.0
                    else:
                        rewards[other_a] -= 50.0

        # Survival bonus
        for a, player in self.players.items():
            if player.status == "alive":
                rewards[a] += 0.00001

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

        return obs, rewards, terminations, truncations, infos

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

        # Chuyển sang (H, W, C), scale về [0,255]
        obs = np.transpose(H, (1, 2, 0)) * 255.0
        return obs.astype(np.uint8)

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
            rewards[bomb.owner] -= 0.5  # phạt khi phá item

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
                    rewards[bomb.owner] -= 0.5
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
                            rewards[bomb.owner] += 50.0
                        else:
                            rewards[bomb.owner] -= 30.0

            rewards[bomb.owner] += (brick_destroyed ** 1.5) * 5.0

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
