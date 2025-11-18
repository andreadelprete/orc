#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def epsGreedy(Q, x, eps, nu):
    if uniform(0,1) < eps:
        return randint(nu)
    return np.argmin(Q[x,:])

def sarsa(env, gamma, Q, nEpisodes, maxEpisodeLength, 
          learningRate, eps, eps_decreasing_decay, min_eps, 
          compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' SARSA:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        eps: initial exploration probability for epsilon-greedy policy
        eps_decreasing_decay: rate of exponential decay of epsilon
        min_eps: lower bound of epsilon
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    h_ctg = []           # Learning history (for plots)
    # for every episode
    # reset the environment state
    # compute u with an epsilon-greedy policy
    # for every step of the episode
    # compute u_next with an epsilon-greedy policy
    # compute Q target
    # update Q function with TD
    # update x, u with x_next, u_next
    # update epsilon with exponentially decaying function: eps=e^(-decay*episode)
    # append cost-to-go to list h_ctg (for plots)
    # every nprint episodes print mean V, mean cost-to-go, and epsilon
        
    return Q, h_ctg