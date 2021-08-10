#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 08:07:56 2021

@author: adelprete
"""

import numpy as np
import gym

env_name = "FrozenLake-v0"
#env_name = "Pendulum-v0"
#env_name = "CartPole-v1"
#env_name = "MountainCar-v0"

nbinsX = 10                             # nb of bins used to discretize state space
nbinsU = 10                             # nb of bins used to discretize control space

print("Environment", env_name)

env = gym.make(env_name)

discretize_control_space = False
Lu, Hu = None, None
if(type(env.action_space) is gym.spaces.box.Box):
    discretize_control_space = True
    du = env.action_space.shape[0]     # size of continuous state space
    Hu = env.action_space.high
    Lu = env.action_space.low
    if(env_name == "Pendulum-v0"):
        pass
    print("Control space range", Lu, Hu)
    nu = nbinsU**du
else:
    nu = env.action_space.n

discretize_state_space = False
Lx, Hx = None, None
if(type(env.observation_space) is gym.spaces.box.Box):
    discretize_state_space = True
    dx = env.observation_space.shape[0]     # size of continuous state space
    Hx = env.observation_space.high
    Lx = env.observation_space.low
    if(env_name == "CartPole-v1"):
        Hx[1] = 5
        Hx[3] = 5
        Lx[1] = -5
        Lx[3] = -5
    print("State space range", Lx, Hx)
    nx = nbinsX**dx
else:
    nx = env.observation_space.n

print("State space dimension:  ", nx)
print("Control space dimension:", nu)

def continuous2discrete(x, L, H, nbins):
    if(type(x) is int):
        return x
    
    for i in range((x.shape[0])):
        assert(x[i]<=H[i])
        assert(x[i]>=L[i])
        
    x_int = [int(np.floor(nbins*(x[i]-L[i])/(H[i]-L[i]))) for i in range(H.shape[0])]
    index = 0
    c = 1
    for i in range(x.shape[0]):
        if(x_int[i]>=nbins): x_int[i]=nbins-1
        index += x_int[i] * c
        c *= nbins
    if(index > nbins**x.shape[0]):
        print("L", L, "H", H)
        print("x=", x, "\nx_int=", x_int)
        print("x_index = ", index)
    return index

def discrete2continuous(x, L, H, nbins):
    x_int = np.zeros_like(L)
    index = 0
    c = 1
    for i in range(x_int.shape[0]):
        x_int[i] = x%(c*nbins)
        x -= x_int[i]*c
        index += x_int[i]*c
        c *= nbins        
#    print("L", L, "H", H)
#    print("x=", x, "\nx_int=", x_int)
#    print("x_index = ", index)
    return np.array([index])

#Initialize the Q-table to 0
Q = np.zeros((nx,nu))

n_episodes = 100000      #number of episode we will run
max_iter_episode = 50  #maximum of iteration per episode
exploration_prob = 1   #initialize the exploration probability to 1
                        #exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.003
min_exploration_prob = 0*0.001            # minimum of exploration proba
gamma = 0.99                            #discounted factor
lr = 0.5                                #learning rate

rewards_per_episode = list()

x = env.reset()
x_max = np.zeros_like(x)
x_min = np.zeros_like(x)

def test_greedy_policy():
    x = env.reset()
    x_index = continuous2discrete(x, Lx, Hx, nbinsX)
    done = False    
    total_episode_reward = 0    
    for i in range(max_iter_episode): 
#        env.render()
        u_index = np.argmax(Q[x_index,:])
        if(discretize_control_space):
            u = discrete2continuous(u_index, Lu, Hu, nbinsU)
        else:
            u = u_index
        x_next, reward, done, _ = env.step(u)
        total_episode_reward = total_episode_reward + reward
        if done:
            break
        x_index = continuous2discrete(x_next, Lx, Hx, nbinsX)
    env.close()
    print("Total episode return for greedy policy:", total_episode_reward)

#we iterate over episodes
Q_old = np.copy(Q)
for e in range(n_episodes):        
    #we initialize the first state of the episode
    x = env.reset()
    x_index = continuous2discrete(x, Lx, Hx, nbinsX)
    if(not discretize_state_space):
        assert(x==x_index)
    done = False
    
    #sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    
    for i in range(max_iter_episode): 
#        env.render()        
#        for i in range(x.shape[0]):
#            if(x[i]>x_max[i]): x_max[i] = x[i]
#            if(x[i]<x_min[i]): x_min[i] = x[i]
            
        if np.random.uniform(0,1) < exploration_prob:
            u = env.action_space.sample()
            u_index = continuous2discrete(u, Lu, Hu, nbinsU)
        else:
            u_index = np.argmax(Q[x_index,:])
            if(discretize_control_space):
                u = discrete2continuous(u_index, Lu, Hu, nbinsU)
            else:
                u = u_index
        
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        x_next, reward, done, _ = env.step(u)
        x_next_index = continuous2discrete(x_next, Lx, Hx, nbinsX)
        
        # We update our Q-table using the Q-learning iteration
        Q[x_index, u_index] = (1-lr) * Q[x_index, u_index] + \
                              lr*(reward + gamma*max(Q[x_next_index,:]))
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        x = x_next
        x_index = x_next_index
        
    #We update the exploration proba using exponential decay formula 
    exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay*e))
    rewards_per_episode.append(total_episode_reward)
    
    if(e%100==0):
        print("\nEpisode", e, 100*e/n_episodes, "%")
        print("\tAverage return  ", np.mean(rewards_per_episode[-100:]))
#        print("\tAverage Q table:", np.mean(Q_table))
#        print("\tMax Q table:    ", np.max(Q))
        print("\texploration_prob", 100*exploration_prob, "%")
        print("\t|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
#        plot_Q_table(Q)
        Q_old = np.copy(Q)
        test_greedy_policy()
#        print("\tx min", x_min)
#        print("\tx max", x_max)

print("\nTraining finished. Test greedy policy.")
test_greedy_policy()
  
import matplotlib.pyplot as plt
plt.plot(rewards_per_episode, 'x ')
plt.title("Return per episode")
plt.show()

smooth_return = np.zeros_like(rewards_per_episode)
for i in range(smooth_return.shape[0]):
    if(i<smooth_return.shape[0]-100):
        smooth_return[i] = np.mean(rewards_per_episode[i:i+100])
    else:
        smooth_return[i] = np.mean(rewards_per_episode[i:])
plt.plot(smooth_return)
plt.title("Smoothed Return per episode")
plt.show()
