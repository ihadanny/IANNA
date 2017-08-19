import logging
import gym
from gym import spaces
import numpy as np, pandas as pd, os

logger = logging.getLogger(__name__)

def get_state_rep(df, grouped_by):
    res = [df.shape[0]]
    group_cols = [k for k, v in grouped_by.items() if v == 1]
    if len(group_cols) > 0:
        num_groups = df[group_cols].drop_duplicates().shape[0]
    else:
        num_groups = df.shape[0]
    res.append(num_groups)
    dist_counts = df.nunique().to_dict()
    for col in df.columns:
        res.append(dist_counts[col])
    for col in df.columns:
        res.append(grouped_by[col])
    return res


class IANNAEnv(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }
    
    def __init__(self):

        
        #   Two kinds of valid input:
        #   Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
        #   Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape

        #   IANNA observations would be:
        #   number of records
        #   number of groups
        #   distinct values of every column
        #   grouped_by indication for each column
        self.filename = os.path.join(os.path.dirname(__file__), '../../data/1.tsv')        
        print('reading input', self.filename)
        self.data = pd.read_csv(self.filename, sep = '\t', index_col=0)
        grouped_by_all = {col: 1 for col in self.data.columns}
        high = np.array(get_state_rep(self.data, grouped_by_all))
        low = np.zeros_like(high)
        print("observation space from", low, "to", high, "shape", high.shape)
        self.observation_space = spaces.Box(low, high)        

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
        self.action_space = spaces.MultiDiscrete([[0,2], [0, self.data.columns.size-1], [0, 2], [0, 9], [0, self.data.columns.size-1], [0,4]])
        print("action space", self.action_space)
    
    def _reward(self, obs):
        for s in self.states_already_seen:
            if (s == obs).all():
                return 0.0
        return 1.0
        
    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        operator_type = OPERATOR_TYPE_LOOKUP[action[0]] 
        col = self.data.columns[action[1]]
        if operator_type == 'back':
            #print('back')
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
                #print('group', col)
                self.grouped_by[col] = 1
            else:
                raise Exception("unknown operator type: " + operator_type)
        
        obs = np.array(get_state_rep(self.data, self.grouped_by))
        assert self.observation_space.contains(obs)
        
        reward = self._reward(obs)
        self.states_already_seen.append(obs)
        self.step_num += 1
        done = self.step_num >= 10
        return obs, reward, done, {}

    def _reset(self):
        #print('reading input', self.filename)
        self.data = pd.read_csv(self.filename, sep = '\t', index_col=0)
        self.grouped_by = {col: 0 for col in self.data.columns}
        self.history = []
        self.states_already_seen = []
        self.step_num = 0
        obs = np.array(get_state_rep(self.data, self.grouped_by))
        assert self.observation_space.contains(obs)
        return obs

    def _render(self, mode='human', close=False):
        if close:
            return None
        print('rendering...')
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

