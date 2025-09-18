import pygame
import time
import torch

from envs.multi_bomber_env import MultiBomberEnv
from stable_baselines3 import PPO

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


def evaluate(model_path="multi_bomber_selfplay_ppo_finetuned.zip", episodes=3, fps=5):
    # Load model
    print(f"🔄 Loading model from {model_path}")
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    env = MultiBomberEnv(max_steps=10000)

    for ep in range(episodes):
        obs, infos = env.reset()
        total_rewards = {a: 0 for a in env.agents}

        # màn hình pygame đủ để hiển thị ASCII map
        h = (env.grid_h + 2) * FONT_SIZE
        w = (env.grid_w + 2) * FONT_SIZE
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"MultiBomber Episode {ep+1}")

        print(f"\n=== Episode {ep+1} ===")

        done = False
        while not done:
            # pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return

            # lấy action từ policy đã train
            actions = {}
            for agent in env.agents:
                action, _ = model.predict(obs[agent])
                actions[agent] = int(action)

            obs, rewards, terminations, truncations, infos = env.step(actions)
            for a, r in rewards.items():
                total_rewards[a] += r

            draw_env(env, screen)

            done = all(terminations.values()) or all(truncations.values())
            time.sleep(1 / fps)

        print(f"Episode {ep+1} finished. Rewards: {total_rewards}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    evaluate(episodes=2, fps=60)
