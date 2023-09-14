import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # allows to use multiple environment at the same time
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v1"
env = gym.make(environment_name, )
episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        print(env.step(action))
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env = gym.make(environment_name, render_mode="human")
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps=20000)

env.close()

