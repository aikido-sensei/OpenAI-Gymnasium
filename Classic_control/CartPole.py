import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
       observation, info = env.reset()


env.close()