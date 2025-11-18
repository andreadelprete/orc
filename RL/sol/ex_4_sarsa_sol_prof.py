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
    h_ctg = []                              # Learning history (for plot).
    Q_old = np.copy(Q)
    episode = 0
    for it in range(nIter):
        
        # POLICY EVALUATION
        for ep in range(nEpisodes):
            episode += 1
            x    = env.reset()
            costToGo = 0.0
            gamma_i = 1
            for steps in range(maxEpisodeLength):
                if uniform(0,1) < exploration_prob:
                    u = randint(env.nu)
                else:
                    u = pi[x]            
                x_next,cost = env.step(u)
                # Compute reference Q-value at state x respecting Bellman
                Qref = cost + gamma*Q[x_next, pi[x_next]]
                # Update Q-Table
                Q[x,u] += learningRate*(Qref-Q[x,u])
                x       = x_next
                costToGo    += gamma_i*cost
                gamma_i     *= gamma

            exploration_prob = max(min_exploration_prob, 
                               np.exp(-exploration_decreasing_decay*episode))
            h_ctg.append(costToGo)
    
            
        # improve policy by being epsilon-greedy wrt Q
        for x in range(env.nx):
            # Rather than simply using argmin we do something slightly more complex
            # to ensure simmetry of the policy when multiply control inputs
            # result in the same value. In these cases we prefer the more extreme
            # actions
    #        pi[x] = np.argmin(Q[x,:])
            u_best = np.where(Q[x,:]==np.min(Q[x,:]))[0]
            if(u_best[0]>env.c2du(0.0)):
                pi[x] = u_best[-1]
            elif(u_best[-1]<env.c2du(0.0)):
                pi[x] = u_best[0]
            else:
                pi[x] = u_best[int(u_best.shape[0]/2)]
    
        
        if not it%nprint: 
            print('Iter #%d done with cost %d and %.1f exploration prob' % (
                  it, np.mean(h_ctg[-nprint:]), 100*exploration_prob))
            print("max|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
            print("avg|Q - Q_old|=%.2f"%(np.mean(np.abs(Q-Q_old))))
    #        plot_Q_table(Q)
    #        render_greedy_policy(env, Q)
            V, pi = compute_V_pi_from_Q(env, Q)
            env.plot_V_table(V)
#            env.plot_policy(pi)
            Q_old = np.copy(Q)
    
    return Q, h_ctg