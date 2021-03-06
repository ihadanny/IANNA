import logging
import gym
from gym import spaces
import numpy as np, pandas as pd, os
import scipy as sp
from scipy import stats, optimize, interpolate

logger = logging.getLogger(__name__)

def get_keys():
    KEYS=[ 'eth_dst', 'eth_src', 'highest_layer', 'info_line',
           'ip_dst', 'ip_src', 'length', 'number',
            'sniff_timestamp', 'tcp_dstport', 'tcp_srcport',
           'tcp_stream']
    return KEYS

def get_data_column_measures(column):
    #for each column, compute its: (1) normalized value entropy (2)Null count (3)Unique values count
    B=20
    u = column.nunique()
    n = column.isnull().sum()
    column_na=column.dropna()
    size=len(column)
    if column.dtype=='O':
        h=sp.stats.entropy(column_na.value_counts().values)/np.log(len(column.dropna()))
    else:
        h= sp.stats.entropy(np.histogram(column_na,bins=B)[0])/np.log(B)
    return {"unique":u/(size-n),"nulls":n/size,"entropy":h}


def get_state_rep_from_dicts(data_layer, granularity_layer):
    ret = []
    for metric in ['entropy', 'unique']:
        for k in get_keys():
            if pd.isnull(data_layer[k]) or metric not in data_layer[k]:
                ret.append(0)
            else: 
                ret.append(data_layer[k][metric])
    if granularity_layer is None:
        ret.extend([0]*(3+2*len(get_keys())))
    else:
        ret.append(granularity_layer['ngroups'])
        ret.append(granularity_layer['size_mean'])
        ret.append(granularity_layer['size_var'])

        for s in [set(granularity_layer['agg_attrs']), set(granularity_layer['group_attrs'])]:
            for k in get_keys():
                if k in s:
                    ret.append(1)
                else:
                    ret.append(0)
    return ret

    
class IANNAEnv(gym.Env):

    metadata = {
        'render.modes': ['human'],
    }
 
    def get_grouping_measures(self, df, group_obj, agg_df):
        """"number" is the unique identifier of a packet, 
        therefore we use it to count the size of each group , 
        although this may feel hacky"""
        if group_obj is None or agg_df is None:
            return None 
        B=20
        groups_num=len(group_obj)
        if len(agg_df.number) == 0:
            # no grouping
            size_var, size_mean = df.shape[0], df.shape[0]
        else:
            size_var = np.var(agg_df.number/np.sum(agg_df.number))
            size_mean = np.mean(agg_df.number)
        group_keys=group_obj.keys
        agg_keys=list(agg_df.keys()).remove("number")
        agg_nve_dict={}
        if agg_keys is not None:
            for ak in agg_keys:
                agg_nve_dict[ak]=sp.stats.entropy(np.histogram(agg_df[ak],bins=B)[0])/np.log(B)
        return {"group_attrs":group_keys,"agg_attrs":agg_nve_dict,"ngroups":groups_num,"size_var":size_var,"size_mean":size_mean}

    def get_groupby_df(self, df, grouped_by):
        #Given a dataframe, the grouping and aggregations - result (i) the aggregated dataframe, and (ii)the groupby element

        grouping_attrs = [k for k, v in grouped_by.items() if v == 1]
     
        if not grouping_attrs:
            return None,None
        
        df_gb= df.groupby(grouping_attrs)
        
        agg_dict={'number':len} #all group-by gets the count by default in REACT-UI
        agg_df = df_gb.agg(agg_dict)
        return df_gb, agg_df
           
    def calc_gran_layer(self, df, grouped_by):
        group_obj,agg_df = self.get_groupby_df(df, grouped_by)
        r = self.get_grouping_measures(df, group_obj,agg_df)
        #print("calc_gran_layer", r)
        return r

    def calc_data_layer(self, df):
        r = df[get_keys()].apply(get_data_column_measures).to_dict()
        #print("calc_data_layer", r)
        return r

    def get_state_rep(self, df, grouped_by):
        return get_state_rep_from_dicts(self.calc_data_layer(df), self.calc_gran_layer(df, grouped_by))
    
    def __init__(self):

        
        #   Two kinds of valid input:
        #   Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
        #   Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape

        #   IANNA observations would be:
        #   number of records
        #   number of groups
        #   distinct values of every column
        #   grouped_by indication for each column
        self.filename = os.path.join(os.path.dirname(__file__), '../../Data_manipulation/raw_datasets/1.tsv')        
        print('reading input', self.filename)
        self.data = pd.read_csv(self.filename, sep = '\t', index_col=0)
        print(self.data.shape)
        
        
        grouped_by_all = np.array(self.get_state_rep(self.data, {col: 1 for col in self.data.columns}))
        grouped_by_none = np.array(self.get_state_rep(self.data, {col: 0 for col in self.data.columns}))
        print("by_all", grouped_by_all)
        print("by_none", grouped_by_none)
        self.high = np.maximum(grouped_by_all, grouped_by_none)       
        self.low = np.zeros_like(self.high)
        print("observation space from", self.low, "to", self.high, "shape", self.high.shape)
        self.observation_space = spaces.Box(self.low, self.high)        

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
        self.step_num += 1
        done = self.step_num >= self.data.columns.size
        for s in self.states_already_seen:
            if (s == obs).all():
                return True, -1.0
        return done, 1.0
        
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
                #print('filter', col, filter_operator, operand)
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
        
        obs = np.array(self.get_state_rep(self.data, self.grouped_by))
        #print(obs, obs.shape)
        assert self.observation_space.contains(obs)
        
        done, reward = self._reward(obs)
        self.states_already_seen.append(obs)
        return obs, reward, done, {}

    def _reset(self):
        #print('reading input', self.filename)
        self.data = pd.read_csv(self.filename, sep = '\t', index_col=0)
        #self.data = self.data.iloc[:, :5]
        self.grouped_by = {col: 0 for col in self.data.columns}
        self.history = []
        self.states_already_seen = []
        self.step_num = 0
        obs = np.array(self.get_state_rep(self.data, self.grouped_by))
        assert self.observation_space.contains(obs)
        #print(obs, obs.shape)
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

