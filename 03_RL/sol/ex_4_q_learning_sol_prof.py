#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def q_learning(env, gamma, Q, nEpisodes, maxEpisodeLength, 
               learningRate, exploration_prob, exploration_decreasing_decay,
               min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    h_ctg = []                              # Learning history (for plot).
    Q_old = np.copy(Q)
    for episode in range(1,nEpisodes):
        x    = env.reset()
        costToGo = 0.0
        gamma_i = 1
        for steps in range(maxEpisodeLength):
            if uniform(0,1) < exploration_prob:
                u = randint(env.nu)
            else:
                u = np.argmin(Q[x,:]) # Greedy action
    #        u         = np.argmin(Q[x,:] + np.random.randn(1,NU)/episode) # Greedy action with noise
            x_next,cost = env.step(u)
    
            # Compute reference Q-value at state x respecting HJB
            Qref = cost + gamma*np.min(Q[x_next,:])
    
            # Update Q-Table to better fit HJB
            Q[x,u]      += learningRate*(Qref-Q[x,u])
            x           = x_next
            costToGo    += gamma_i*cost
            gamma_i     *= gamma
    
        exploration_prob = max(min_exploration_prob, 
                               np.exp(-exploration_decreasing_decay*episode))
        h_ctg.append(costToGo)
        if not episode%nprint: 
            print('Episode #%d done with cost %d and %.1f exploration prob' % (
                  episode, np.mean(h_ctg[-nprint:]), 100*exploration_prob))
            print("max|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
            print("avg|Q - Q_old|=%.2f"%(np.mean(np.abs(Q-Q_old))))
            if(plot):
        #        env.plot_Q_table(Q)
        #        render_greedy_policy(env, Q)
                V, pi = compute_V_pi_from_Q(env, Q)
                env.plot_V_table(V)
#                env.plot_policy(pi)
            Q_old = np.copy(Q)
    
    return Q, h_ctg