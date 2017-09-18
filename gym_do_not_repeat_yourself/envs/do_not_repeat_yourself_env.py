import logging
import gym
from gym import spaces
import numpy as np

logger = logging.getLogger(__name__)


class DoNotRepeatYourselfEnv(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }
    
    def __init__(self):
        high = np.array([1]*10)
        low = np.zeros_like(high)
        print("observation space from", low, "to", high, "shape", high.shape)
        self.observation_space = spaces.Box(low, high)  
        self.action_space = spaces.Discrete(10)
        print("action space", self.action_space)
            
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.step_num += 1
        if self.bit_vector[action] == 1:
            done, reward = True, -1
        else:
            self.bit_vector[action] = 1
            assert self.observation_space.contains(self.bit_vector)        
            done = self.step_num >= self.bit_vector.size
            reward = 1
        return self.bit_vector, reward, done, {}

    def _reset(self):
        self.bit_vector = np.array([0]*10)
        self.step_num = 0
        assert self.observation_space.contains(self.bit_vector)
        return self.bit_vector

    def _render(self, mode='human', close=False):
        return None

