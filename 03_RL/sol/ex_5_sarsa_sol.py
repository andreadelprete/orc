#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def sarsa(env, gamma, Q, pi, nIter, nEpisodes, maxEpisodeLength, 
          learningRate, exploration_prob, exploration_decreasing_decay,
          min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' SARSA:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        pi: initial guess for policy
        nIter: number of iterations of the algorithm
        nEpisodes: number of episodes to be used for policy evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    h_ctg = []  # Cost-to-go during learning (for plot).
    # Make a copy of the initial Q table guess
    # for every iteration
    # evaluate the current policy with TD(0)
    # for every episode
    # reset the state
    # simulate the system for maxEpisodeLength steps
    # with probability exploration_prob take a random control input
    # otherwise take a greedy control
    # Compute reference Q-value at state x
    # Update Q-Table with the given learningRate
    # keep track of the cost to go
    # update the exploration probability with an exponential decay: eps = exp(-decay*episode)
    # improve policy by being greedy wrt Q
    # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
    
    return Q, h_ctg