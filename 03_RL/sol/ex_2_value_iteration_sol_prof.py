#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np

def value_iteration(env, gamma, V, maxIters, value_thr, plot=False, nprint=1000):
    ''' Policy iteration algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        value_thr: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    Q  = np.zeros(env.nu)           # temporary array to store value of different controls
    for k in range(maxIters):
        if not k%nprint and plot: 
            env.plot_V_table(V)
        
        V_old = np.copy(V)  # make a copy of current Value table
        for x in range(env.nx):                     # for every state x
            for u in range(env.nu):                 # for every action u
                env.reset(x)                        # reset state to x
                x_next,cost = env.step(u)           # apply action u
                Q[u] = cost + gamma * V[x_next]  # store cost associated to u
            V[x] = np.min(Q)                        # update Value table
                
        # check for convergence
        V_err = np.max(np.abs(V-V_old))
        if(V_err<value_thr):
            print("VI converged after %d iters with error"%k, V_err)
            print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
            # -4.699560419916913 -9.999994614527743 -3.1381005754277433
            if(plot): env.plot_V_table(V)
            return V
            
        if not k%nprint: 
            print('VI - Iter #%d done' % (k))
            print("|V - V_old|=%.5f"%(V_err))
    
    print("Value iteration did NOT converge in %d iters. Error"%k, V_err)
    return V