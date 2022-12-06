#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np

def mc_policy_eval(env, gamma, pi, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    ''' Monte-Carlo Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    N = np.zeros(env.nx) # number of times each state has been visited
    C = np.zeros(env.nx) # cumulative cost associated to each state
    V = np.zeros(env.nx) # Value table
    V_err = []           # evolution of the error between real and estimated V table
    
    for k in range(nEpisodes): # for each episode
        env.reset() # reset the environment to a random state
        x_list = [] # list of states visited in this episode
        c_list = [] # list of costs received at each state in this episode
    
        for t in range(maxEpisodeLength):
            x = env.x
            x_list.append(x)    # store state
            if(callable(pi)):
                u = pi(env, x)      # compute control action according to policy pi
            else:
                u = pi[x]
            x_next, cost = env.step(u) # apply control action
            c_list.append(cost) # store cost
            
        # Update V-Table
        J = 0 # cost-to-go
        for i in range(len(x_list)-1,-1,-1): # for each visited state, going backward in time
            x = x_list[i]
            J = c_list[i] + gamma*J # compute cost-to-go of state x
            C[x] += J               # update cumulative cost associated to x
            N[x] += 1               # increment counter of visits to x
            V[x] = C[x] / N[x]      # update estimated Value of state x
            
        V_err.append(np.mean(np.abs(V-V_real))) # compute error of Value table
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("mean|V_mc - V_real|=%.5f"%(V_err[-1]))
            print("Nb of states not visited yet:", np.count_nonzero(N==0))
            if(plot): env.plot_V_table(V)
        
#        if(np.min(N)>0):
#            print("Stop at iter", k, "because all states visited at least once.")
#            break
    return V, V_err


def td0_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        learningRate: learning rate of the algorithm
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    V = np.copy(V0)
    V_err = []      # evolution of the error between real and estimated V table
    for k in range(1,nEpisodes+1): # for each episode
        env.reset()     # reset environment to random initial state
    
        for t in range(maxEpisodeLength):
#            x = env.x2i(env.x)  # convert state from discrete 2d to discrete 1d representation
            x = env.x
            if(callable(pi)):
                u = pi(env, x)      # compute control action according to policy pi
            else:
                u = pi[x]
            x_next, cost = env.step(u)  # apply action u
            
            # Update V-Table
            TD_target = cost + gamma*V[x_next]
#            V[x] += 1/(1+learningRate*k)*(TD_target - V[x])
            V[x] += learningRate*(TD_target - V[x])
    
        
        V_err.append(np.mean(np.abs(V-V_real))) # compute error of Value table
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("mean|V_td - V_real|=%.5f"%(V_err[-1]))
            if(plot): env.plot_V_table(V)
        
    return V, V_err