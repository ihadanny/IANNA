import numpy as np

import gym
from gym_ianna.envs.ianna_env import IANNAEnv

env = gym.make("IANNA-v0")
env.render()
observation = env.reset()
print(observation)

action = np.array([1, 14, 1, 7, 0, 0])
observation, reward, done, info = env.step(action)
print(observation, reward, done)

action = np.array([2, 4, 0, 0, 0, 0])
observation, reward, done, info = env.step(action)
print(observation, reward, done)

action = np.array([2, 13, 0, 0, 0, 0])
observation, reward, done, info = env.step(action)
print(observation, reward, done)

action = np.array([0, 0, 0, 0, 0, 0])
observation, reward, done, info = env.step(action)
print(observation, reward, done)
