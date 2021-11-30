#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np
#from sol.ex_0_policy_evaluation_prof import policy_eval
from ex_0_policy_evaluation import policy_eval

def policy_iteration(env, gamma, pi, V, maxEvalIters, maxImprIters, value_thr, policy_thr, plot=False, nprint=1000):
    ''' Policy iteration algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        pi: initial guess for the policy
        V: initial guess of the Value table
        maxEvalIters: max number of iterations for policy evaluation
        maxImprIters: max number of iterations for policy improvement
        value_thr: convergence threshold for policy evaluation
        policy_thr: convergence threshold for policy improvement
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # IMPLEMENT POLICY ITERATION HERE
    
    # Create an array to store the Q value of different controls
    # Iterate at most maxImprIters loops
    # Evaluate current policy using policy_eval for at most maxEvalIters iterations 
    # Make a copy of current policy table
    # The number of states is env.nx
    # The number of controls is env.nu
    # You can set the state of the robot using env.reset(x)
    # You can simulate the robot using: x_next,cost = env.step(u)
    # You can find the index corresponding to the minimum of an array with np.argmin(Q)
    # Check for convergence based on how much the policy has changed from the previous loop
    # you can plot the policy with: env.plot_policy(pi)
    # You can plot the Value table with: env.plot_V_table(V)
    # At the end return the policy pi
    return pi