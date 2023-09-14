import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # allows to use multiple environment at the same time
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = "CartPole-v1"


env = gym.make(environment_name, render_mode="human")
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps=2000)

PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)
del model
model = PPO.load('PPO_model', env=env)

env.close()

