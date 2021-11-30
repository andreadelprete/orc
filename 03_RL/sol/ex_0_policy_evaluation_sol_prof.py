#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np

def policy_eval(env, gamma, pi, V, maxIters, threshold, plot=False, nprint=1000):
    ''' Policy evaluation algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        pi: policy to evaluate
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        threshold: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    for k in range(1, maxIters):
        V_old = np.copy(V) # make a copy of the V table
        for x in range(env.nx): # for every state
            env.reset(x) # reset the environment state
            # apply the given policy
            if(callable(pi)):   # check if the policy is a function
                u = pi(env, x) 
            else:   # otherwise assume it's a vector
                u = pi[x]
            x_next, cost = env.step(u)
            
            # Update V-Table with Bellman's equation
            V[x] = cost + gamma*V_old[x_next]
    
        # compute the difference between the current and previous V table
        V_err = np.max(np.abs(V-V_old))
        if(V_err<threshold):    # check convergence
            print("Policy eval converged after %d iters with error"%k, V_err)
            if(plot): env.plot_V_table(V)
            return V
            
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("|V - V_old|=%.5f"%(V_err))
            if(plot): env.plot_V_table(V)
    print("Policy eval did NOT converge in %d iters. Error"%k, V_err)
    return V