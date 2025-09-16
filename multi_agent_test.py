import pygame
import time

from envs.multi_bomber_env import MultiBomberEnv

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


def evaluate(episodes=3, fps=5):
    env = MultiBomberEnv(grid_w=15, grid_h=11, max_steps=200)

    for ep in range(episodes):
        obs, infos = env.reset()
        terminations = {a: False for a in env.agents}
        truncations = {a: False for a in env.agents}
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

            # random action cho mỗi agent (thay bằng policy của bạn)
            actions = {a: env.action_space(a).sample() for a in env.agents}

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
    evaluate(episodes=2, fps=5)
