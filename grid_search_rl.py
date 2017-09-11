
# coding: utf-8

# In[1]:

import sys
sys.path.append("../")


# In[2]:

import numpy as np, pandas as pd
import gym
from gym_ianna.envs.ianna_env import IANNAEnv
import tensorflow as tf


# In[3]:

import gym_ianna.envs.ianna_env as ianna_env

ianna_env.__file__


# In[4]:
default_args = {
    'ENV': 'IANNA-v0'
    ,'START_STATE_FROM': -15 # last 15 bits of the IANNA state are the groupby bits
    ,'OP_NUMBER' : 4          # how many fields can we turn on
    ,'STATE_INPUT_SIZE' : 4   # how many fields can we observe 
    ,'MAX_STEPS' : 4          # how many steps must we play in each episode    
    
    #nn params
    ,'HIDDEN_SIZE': 20
    
    #discount rewards params
    ,'GAMMA' : 0.99
    
    #ADAM Optimizer hyper-parameters:
    ,'LEARNING_RATE' : 0.01
    ,'B1' : 0.8
    ,'B2' : 0.999
    ,'EPSILON' : 1e-6
    
    #learning params
    ,'TOTAL_EPISODES' : 5000
    ,'BATCH_NUMBER' : 20
    ,'DISPLAY_FREQ' : 500
}

grid = []
for i in range(4,15):
    for _ in range(3): # repeat each experiment 3 times
        grid.append({ 'OP_NUMBER' : i          # how many fields can we turn on
        ,'STATE_INPUT_SIZE' : i   # how many fields can we observe 
        ,'MAX_STEPS' : i          # how many steps must we play in each episode    
        })

res_list = []
for g in grid:
    args = default_args.copy()
    args.update(g)
    print('*'*80)
    print(args)
    
    env = gym.make(args['ENV'])
    # In[5]:
    
    #Initializing 
    tf.reset_default_graph()
    
    W1 = tf.get_variable(shape=[args['HIDDEN_SIZE'],args['STATE_INPUT_SIZE']],name='w1',
                          initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(shape=[args['HIDDEN_SIZE'],args['HIDDEN_SIZE']],name='w2',
                          initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable(shape=[args['OP_NUMBER'],args['HIDDEN_SIZE']],name='w3',
                          initializer=tf.contrib.layers.xavier_initializer())
    
    b1 = tf.get_variable(shape=[args['HIDDEN_SIZE'],1],name='b1',
                          initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(shape=[args['HIDDEN_SIZE'],1],name='b2',
                          initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(shape=[args['OP_NUMBER'],1],name='b3',
                          initializer=tf.contrib.layers.xavier_initializer())
    
    #Layers:
    x = tf.placeholder(tf.float32, shape=[args['STATE_INPUT_SIZE'],None],name='x')
    h1 = tf.tanh(tf.matmul(W1,x) + b1)
    h2 = tf.tanh(tf.matmul(W2,h1) + b2)
    y = tf.nn.softmax(tf.matmul(W3,h2) + b3,dim=0)
    
    
    # In[6]:
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    
    
    
    # In[7]:
    
    #Loss function:
    curr_reward = tf.placeholder(shape=[None],dtype=tf.float32)
    actions_array = tf.placeholder(shape=[None],dtype=tf.int32)
    pai_array = tf.gather(y,actions_array)
    L = -tf.reduce_mean(tf.log(pai_array)*curr_reward)
    gradient_holders = []
    gradients = tf.gradients(L,tf.trainable_variables())
    
    
    # In[8]:
    
    tvars = tf.trainable_variables()
    #Initialize gradient lists for each trainable variable:
    for idx,var in enumerate(tvars):
        placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
        gradient_holders.append(placeholder)
    
    
    # In[9]:
    
    
    #Update mechanism:
    adam = tf.train.AdamOptimizer(learning_rate=args['LEARNING_RATE'],beta1=args['B1'],beta2=args['B2'],epsilon=args['EPSILON'])
    update_batch = adam.apply_gradients(zip(gradient_holders,tvars))
    
    
    # In[10]:
    
    # grad buffer is initialized to all zeros. 
    # It's used to accumulate the gradients and is a regular variable, NOT a tf variable
    def reset_graph():
        init = tf.global_variables_initializer()
        sess.run(init)
        #saver.restore(sess, "../models/ianna-nn-supervised")
    
    reset_graph()
    grad_buffer = sess.run(tf.trainable_variables())
    
    def reset_grad_buffer():
        for ix,grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0
    
    
    # In[11]:
    
    def get_action(sess,observation):
        """
        Given an observation, return action sampled according to the probabilities of the NN output
        """
        a_dist = sess.run(y,feed_dict={x:np.reshape(observation,(args['STATE_INPUT_SIZE'], 1))})
        a = np.random.choice(range(args['OP_NUMBER']),p=a_dist.reshape((args['OP_NUMBER'])))
        return a
    
    
    # In[12]:
    
    def train(sess,cur_states_array,cur_actions_array,cur_curr_reward):
        """
        NN training procedure: Given arrays of states(observations),
        actions and rewards it computes the derivatives of the loss function
        then add the derivation values to the buffer 
        """
    
        G = sess.run(gradients,feed_dict={x:cur_states_array,actions_array:cur_actions_array,curr_reward:cur_curr_reward})
        for idx,grad in enumerate(G):
            grad_buffer[idx] += grad
    
    
    # In[13]:
    
    def update(sess):
        """
        NN update procedure: apply the gradients to the NN variables
        """
        feed_dict = dict(zip(gradient_holders, grad_buffer))
        _ = sess.run(update_batch, feed_dict=feed_dict)
    
    
    # In[14]:
    
    #        IANNA actions would be:
    #        0) action_type:            back[0], filter[1], group[2]
    #        1) col_id:                 [0..num_of_columns-1]
    #        2) filter_operator:        LT[0], GT[1] if the selected column was numeric (maybe change semantics if column is STR?)
    #        3) filter_decile:          [0..9] the filter operand  
    #        4) aggregation column_id:  [0..num_of_columns - 1] (what do we do if the selected col is also grouped_by?)
    #        5) aggregation type:       MEAN[0], COUNT[1], SUM[2], MIN[3], MAX[4]
    
    def build_ianna_action_from_grouped_by_field(grouped_by_field):
        action = [2, grouped_by_field, 0, 0, 0, 0]
        return action
    
    
    # In[21]:
    
    def project_state_to_nn_input(x):
        return x[args['START_STATE_FROM']:args['START_STATE_FROM']+args['STATE_INPUT_SIZE']]
    
    
    # In[22]:
    
    
    def discount_rewards(arr):
        """
        Helper function for computing discounted rewards,
        then the delayed rewards are normalized by the mean and std as requested.
        """
        discounts = np.zeros_like(arr)
        reward = 0
        for i in reversed(range(arr.size)):
            reward=args['GAMMA']*(arr[i]+reward)
            discounts[i] = reward
        # following 3 lines destroy everything when the game is really simple: 
        # pick 4 fields out of 5 without repeating yourself
        #mean = np.mean(discounts,keepdims=True)
        #discounts = discounts - mean
        #discounts = discounts/ np.std(discounts)
        return discounts
    
    
    # In[28]:
        
    episode_number = 0
    rewards = []
    steps=[]
    max_reward=0
    reset_graph()
    reset_grad_buffer()

    
    while episode_number < args['TOTAL_EPISODES']:
        for ep in range(args['BATCH_NUMBER']):
            obsrv = project_state_to_nn_input(env.reset())
            ep_history=[]
            step_num=0
            total_reward=0
            done=False
    
            while not done and step_num < args['MAX_STEPS']:
                #Perform the game "step:"
                step_num+=1
                action = get_action(sess,obsrv)
                if args['ENV'] == 'IANNA-v0':
                    complex_action = build_ianna_action_from_grouped_by_field(action)
                    obsrv1, reward, done, info = env.step(complex_action)
                else:
                    obsrv1, reward, done, info = env.step(action)
    
                total_reward+=reward
                ep_history.append((np.array(obsrv),action,reward))
                obsrv=project_state_to_nn_input(obsrv1)
    
            episode_number+=1
            ep_history= np.array(ep_history)   
            ep_history[:,2] = discount_rewards(ep_history[:,2])
    
            """
            perform the training step, 
            feeding the network with the ep_history that contains
            the states,actions, and discounted rewards
            """
            ep_states_array = np.vstack(ep_history[:,0]).T
            ep_actions_array = ep_history[:,1].T
            ep_curr_reward = ep_history[:,2].T
            L=train(sess, ep_states_array, ep_actions_array, ep_curr_reward)
    
            #update the rewards/steps counter, storing the data for all episodes
            rewards.append(total_reward)
            steps.append(step_num)    
            
            if episode_number%args['DISPLAY_FREQ']==0:
                print("latest game actions: ", ep_actions_array.T)
                print("latest game reward: ", total_reward)
                print("latest game first state: ", ep_states_array.T[0])
                print("latest game last state: ", obsrv)
                print("Total episodes: %d"%episode_number)
                print("Average steps per %d episodes: %f"%(args['DISPLAY_FREQ'], np.mean(steps[-args['DISPLAY_FREQ']:])))
                print("Average reward per %d episodes : %f"%(args['DISPLAY_FREQ'], np.mean(rewards[-args['DISPLAY_FREQ']:])))
                args['AVERAGE_REWARD_%d' % episode_number] = np.mean(rewards[-args['DISPLAY_FREQ']:])
        update(sess)
        reset_grad_buffer()
        if np.mean(rewards[-args['BATCH_NUMBER']:])>max_reward:
            max_reward=np.mean(rewards[-args['BATCH_NUMBER']:])
            print("\t\t\tCurr Max mean reward per batch:",max_reward)
    
    res_list.append(args)
    res = pd.DataFrame(res_list)
    res.to_csv('grid_search_rl.csv')

    # In[ ]:    
    sess.close()


