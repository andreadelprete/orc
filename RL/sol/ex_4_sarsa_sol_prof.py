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
    h_ctg = []                              # Learning history (for plot).
    Q_old = np.copy(Q)
    for ep in range(1, nEpisodes):
        x    = env.reset()
        u = epsGreedy(Q, x, eps, env.nu)
        costToGo = 0.0
        gamma_i = 1
        for steps in range(maxEpisodeLength):
            x_next, cost = env.step(u)
            u_next = epsGreedy(Q, x_next, eps, env.nu)
            # Compute Q-target
            Q_target = cost + gamma*Q[x_next, u_next]
            # Update Q-Table
            Q[x,u] += learningRate*(Q_target-Q[x,u])
            x       = x_next
            u       = u_next

            costToGo    += gamma_i*cost
            gamma_i     *= gamma

        eps = max(min_eps, np.exp(-eps_decreasing_decay*ep))
        h_ctg.append(costToGo)
    
        if ep%nprint==0: 
            V, pi = compute_V_pi_from_Q(env, Q)
            print('Episode #%d, mean V %.2f, mean ctg %.1f, eps %.1f' % (
                  ep, np.mean(V), np.mean(h_ctg[-nprint:]), 100*eps))
            print("max|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
            print("avg|Q - Q_old|=%.2f"%(np.mean(np.abs(Q-Q_old))))
            if(plot):
                env.plot_V_table(V)
            Q_old = np.copy(Q)
    
    return Q, h_ctg