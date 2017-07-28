import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np, pandas as pd, os
import pickle 

logger = logging.getLogger(__name__)

class IANNAEnv(gym.Env):

    def get_current_state(self):
        res = [self.data.shape[0]]
        dist_counts = self.data.nunique().to_dict()
        for col in self.columns:
            res.append(dist_counts[col])
        return np.array(res)

    def __init__(self):

        #    e.g. Nintendo Game Controller
        #    - Can be conceptualized as 3 discrete action spaces:
        #        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        #        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        #        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        #    - Can be initialized as
        #        MultiDiscrete([ [0,4], [0,1], [0,1] ])

        #        IANNA actions would be:
        #        1) action_type:            filter[0], group[1]
        #        2) field_id:               [0..num_of_fields-1]
        #        3) filter_operator:        EQ[0], GT[1]. LT[2] if the selected field was numeric (maybe change semantics if field is STR?)
        #        4) filter_decile:          [0..9] the filter operand  
        #        5) aggregation field_id:   [0..num_of_fields-1] (what do we do if the selected field is also the main field_id?)
        #        6) aggregation type:       MEAN[0], COUNT[1], SUM[2], MIN[3], MAX[4]
        dir = os.path.dirname(__file__)
        self.filename = os.path.join(dir, '../../data/1.tsv')        
        self.data = pd.read_csv(self.filename, sep = '\t')
        self.columns = self.data.columns
        print(self.columns)
        self.num_rows = self.data.shape[0]
        self.num_fields = self.data.shape[1]
        
        self.action_space = spaces.MultiDiscrete([[0,1], [0, self.num_fields-1], [0, 2], [0, 9], [0, self.num_fields-1], [0,4]])
        
        #   Two kinds of valid input:
        #   Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
        #   Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape

        #   IANNA observations would be:
        #   1) number of records
        #   2..num_fields) distinct values of every field        
        
        high = self.get_current_state()
        low = np.zeros_like(high)
        self.observation_space = spaces.Box(low, high)        
    

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        col = self.columns[action[1]]
        if action[0] == 0:
            print('filter', col)
        else:
            print('group', col)
        
        reward = 1.0
        return self.get_current_state(), reward, False, {}

    def _reset(self):
        self.data = pd.read_csv(self.filename, sep = '\t')
        return np.array(self.get_current_state())

    def _render(self, mode='human', close=False):
        return None



