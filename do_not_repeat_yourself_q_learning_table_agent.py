
# coding: utf-8

# In[1]:

import sys
sys.path.append("../")


# In[2]:

import numpy as np, pandas as pd
import gym
from gym_do_not_repeat_yourself.envs.do_not_repeat_yourself_env import DoNotRepeatYourselfEnv
import gym_do_not_repeat_yourself.envs.do_not_repeat_yourself_env  as do_not_repeat_yourself_env
import tensorflow as tf


# In[3]:

print(do_not_repeat_yourself_env .__file__)


# In[4]:
default_args = {
    'ENV': 'DoNotRepeatYourself-v0'
    ,'MAX_STEPS': 10
    ,'GAMMA': 0.4
    #learning params
    ,'LEARNING_RATE': 0.1

    ,'TOTAL_EPISODES' : 100000
    ,'DISPLAY_FREQ' : 10000
}


args = default_args.copy()
print('*'*80)
print(args)

env = gym.make(args['ENV'])
# In[5]:

# In[28]:
    
episode_number = 0
rewards = []
steps=[]
max_reward=0


Q = np.zeros([2 ** env.observation_space.shape[0],env.action_space.n])
def state_to_int(s):
    ret = 0
    for v in s:
        ret = ret*2 + v
    return ret

def get_action(obsrv, episode_number):
    a = np.argmax(Q[state_to_int(obsrv), :])
    a = np.argmax(Q[state_to_int(obsrv),:] + np.random.randn(1,env.action_space.n)*(1./(episode_number+1)))
    return a

while episode_number < args['TOTAL_EPISODES']:
    obsrv = env.reset()
    ep_history=[]
    step_num=0
    total_reward=0
    done=False

    while not done and step_num < args['MAX_STEPS']:
        #Perform the game "step:"
        step_num+=1
        action = get_action(obsrv, episode_number)
        obsrv1, reward, done, info = env.step(action)
        new_estimate = reward + args['GAMMA']*np.max(Q[state_to_int(obsrv1), :])
        old_estimate = Q[obsrv, action] 
        Q[obsrv, action] = (1-args['LEARNING_RATE']) * old_estimate + args['LEARNING_RATE'] * new_estimate 
        total_reward+=reward
        ep_history.append((obsrv,action,reward))
        obsrv=obsrv1

    episode_number+=1
    ep_history= np.array(ep_history)   

    #update the rewards/steps counter, storing the data for all episodes
    rewards.append(total_reward)
    steps.append(step_num)    
    
    if episode_number%args['DISPLAY_FREQ']==0:
        print("latest game reward: ", total_reward)
        print("latest game last state: ", obsrv)
        print("Total episodes: %d"%episode_number)
        print("Average steps per %d episodes: %f"%(args['DISPLAY_FREQ'], np.mean(steps[-args['DISPLAY_FREQ']:])))
        print("Average reward per %d episodes : %f"%(args['DISPLAY_FREQ'], np.mean(rewards[-args['DISPLAY_FREQ']:])))
        args['AVERAGE_REWARD_%d' % episode_number] = np.mean(rewards[-args['DISPLAY_FREQ']:])



