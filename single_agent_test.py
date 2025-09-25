import time

import pygame
import torch
from stable_baselines3 import PPO

from envs.bomber_env import BomberEnv

# --- Pygame init ---
pygame.init()
CELL_SIZE = 20
FONT_SIZE = 20
font = pygame.font.SysFont("Courier", FONT_SIZE)


# --- Draw env từ ASCII render ---
def draw_env(env, screen):
    ascii_map = env.render()
    screen.fill((0, 0, 0))

    for i, line in enumerate(ascii_map.splitlines()):
        text_surface = font.render(line, True, (200, 200, 200))
        screen.blit(text_surface, (10, i * FONT_SIZE))

    pygame.display.flip()


def evaluate(model_path="train/multi_bomber_selfplay_ppo_finetuned.zip", episodes=3, fps=5):
    # Load model
    print(f"🔄 Loading model from {model_path}")
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    env = BomberEnv(max_steps=10000)

    for ep in range(episodes):
        ob, info = env.reset()
        total_reward = 0

        # màn hình pygame đủ để hiển thị ASCII map
        h = (env.grid_h + 2) * FONT_SIZE
        w = (env.grid_w + 2) * FONT_SIZE
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"MultiBomber Episode {ep + 1}")

        print(f"\n=== Episode {ep + 1} ===")

        done = False
        while not done:
            # pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return

            action, _ = model.predict(ob)

            ob, reward, termination, truncation, info = env.step(int(action))
            total_reward += reward

            draw_env(env, screen)

            done = termination or truncation
            time.sleep(1 / fps)

        print(f"Episode {ep + 1} finished. Rewards: {total_reward}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    evaluate(episodes=2, fps=30)
