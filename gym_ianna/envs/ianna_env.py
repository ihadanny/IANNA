import logging
import gym
from gym import spaces
import numpy as np, pandas as pd, os

logger = logging.getLogger(__name__)

class IANNAEnv(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }
    
    def __init__(self):

        #    e.g. Nintendo Game Controller
        #    - Can be conceptualized as 3 discrete action spaces:
        #        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        #        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        #        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        #    - Can be initialized as
        #        MultiDiscrete([ [0,4], [0,1], [0,1] ])

        #        IANNA actions would be:
        #        0) action_type:            back[0], filter[1], group[2]
        #        1) col_id:                 [0..num_of_columns-1]
        #        2) filter_operator:        LT[0], GT[1] if the selected column was numeric (maybe change semantics if column is STR?)
        #        3) filter_decile:          [0..9] the filter operand  
        #        4) aggregation column_id:  [0..num_of_columns - 1] (what do we do if the selected col is also grouped_by?)
        #        5) aggregation type:       MEAN[0], COUNT[1], SUM[2], MIN[3], MAX[4]
        dir = os.path.dirname(__file__)
        self.filename = os.path.join(dir, '../../data/1.tsv')
        self._reset()
        self.action_space = spaces.MultiDiscrete([[0,2], [0, self.data.columns.size-1], [0, 2], [0, 9], [0, self.data.columns.size-1], [0,4]])
        
        #   Two kinds of valid input:
        #   Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
        #   Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape

        #   IANNA observations would be:
        #   number of records
        #   number of groups
        #   distinct values of every column
        #   grouped_by indication for each column
        
        high = self.get_current_state()
        low = np.zeros_like(high)
        self.observation_space = spaces.Box(low, high)        
    
    def get_current_state(self):
        res = [self.data.shape[0]]
        group_cols = [k for k, v in self.grouped_by.items() if v == 1]
        if len(group_cols) > 0:
            num_groups = self.data[group_cols].drop_duplicates().shape[0]
        else:
            num_groups = self.data.shape[0]
        res.append(num_groups)
        dist_counts = self.data.nunique().to_dict()
        for col in self.data.columns:
            res.append(dist_counts[col])
        for col in self.data.columns:
            res.append(self.grouped_by[col])
        
        return np.array(res)

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        operator_type = OPERATOR_TYPE_LOOKUP[action[0]] 
        col = self.data.columns[action[1]]
        if operator_type == 'back':
            print('back')
            if len(self.history) > 0:
                self.data, self.grouped_by = self.history.pop()
        else:
            self.history.append((self.data.copy(), self.grouped_by.copy()))
            if operator_type == 'filter':
                filter_operator = FILTER_OPERATOR_LOOKUP[action[2]]
                operand = 1.0 * self.data[col].count() * action[3]/10
                print('filter', col, filter_operator, operand)
                if filter_operator == "LT":
                    self.data = self.data[self.data[col].rank() < operand]
                elif filter_operator == "GT":
                    self.data = self.data[self.data[col].rank() > operand]
                else:
                    raise Exception("unknown filter operator: " + filter_operator)
            elif operator_type == 'group':
                print('group', col)
                self.grouped_by[col] = 1
            else:
                raise Exception("unknown operator type: " + operator_type)
        
        reward = 1.0
        return self.get_current_state(), reward, False, {}

    def _reset(self):
        self.data = pd.read_csv(self.filename, sep = '\t')
        self.grouped_by = {col: 0 for col in self.data.columns}
        self.history = []
        return np.array(self.get_current_state())

    def _render(self, mode='human', close=False):
        group_cols = [k for k, v in self.grouped_by.items() if v == 1]
        print('grouping by:', group_cols)
        print(self.data.nunique().T)
        return None

OPERATOR_TYPE_LOOKUP = {
        0: "back",
        1: "filter",
        2: "group",
        }

FILTER_OPERATOR_LOOKUP = {
        0: "LT",
        1: "GT",
        }

