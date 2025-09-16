import pygame
import numpy as np
from stable_baselines3 import PPO
from envs.multi_bomber_env import MultiBomberEnv
from train.train_multi_agents import PettingZooParallelToGymWrapper

# === Pygame render settings ===
TILE_SIZE = 32
FPS = 10  # tốc độ frame

# Colors
COLOR_BG = (0, 0, 0)
COLOR_WALL = (128, 128, 128)
COLOR_BRICK = (200, 100, 50)
COLOR_BOMB = (0, 0, 255)
COLOR_FLAME = (255, 0, 0)
COLOR_ITEM_BOMB = (0, 255, 255)
COLOR_ITEM_POWER = (255, 255, 0)
COLOR_ITEM_SPEED = (0, 255, 0)
COLOR_PLAYER = [(255, 255, 255), (255, 0, 255), (0, 255, 255), (255, 165, 0)]

# --- Create environment and load model ---
env = PettingZooParallelToGymWrapper(MultiBomberEnv(grid_w=28, grid_h=18, max_steps=3000))
model = PPO.load("multi_bomber_ppo_centralized.zip", env=env)

# --- Initialize Pygame ---
grid_h, grid_w = env.env.grid_h, env.env.grid_w
screen = pygame.display.set_mode((grid_w * TILE_SIZE, grid_h * TILE_SIZE))
pygame.display.set_caption("MultiBomberEnv Pygame")
clock = pygame.time.Clock()

def draw_env(env):
    screen.fill(COLOR_BG)
    game_map = env.env.map
    flames = env.env.flames
    items = env.env.items
    bombs = env.env.bombs
    players = env.env.players

    # Draw map tiles
    for y in range(grid_h):
        for x in range(grid_w):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if game_map[y, x] == 1:  # Wall
                pygame.draw.rect(screen, COLOR_WALL, rect)
            elif game_map[y, x] == 2:  # Brick
                pygame.draw.rect(screen, COLOR_BRICK, rect)

    # Draw bombs
    for b in bombs:
        rect = pygame.Rect(b.x * TILE_SIZE, b.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, COLOR_BOMB, rect)

    # Draw flames
    for y in range(grid_h):
        for x in range(grid_w):
            if flames[y, x] > 0:
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, COLOR_FLAME, rect)

    # Draw items
    for y in range(grid_h):
        for x in range(grid_w):
            item = items[y, x]
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if item == 1:
                pygame.draw.rect(screen, COLOR_ITEM_BOMB, rect)
            elif item == 2:
                pygame.draw.rect(screen, COLOR_ITEM_POWER, rect)
            elif item == 3:
                pygame.draw.rect(screen, COLOR_ITEM_SPEED, rect)

    # Draw players
    for i, (a, p) in enumerate(players.items()):
        if p.status != "dead":
            rect = pygame.Rect(p.position[0] * TILE_SIZE, p.position[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, COLOR_PLAYER[i], rect)

    pygame.display.flip()

# --- Run test episode ---
obs, _ = env.reset()
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    draw_env(env)

    done = terminated or truncated
    clock.tick(FPS)

pygame.quit()
print("Episode finished, total reward:", info.get("episode", {}).get("r", 0))
