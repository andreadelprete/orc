#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np
from dpendulum import plot_V_table, plot_policy
from sol.ex_0_policy_evaluation_prof import policy_eval
#from ex_0_policy_evaluation import policy_eval

def policy_iteration(env, gamma, pi, V, maxEvalIters, maxImprIters, policy_thr, value_thr, plot=False, nprint=1000):
    ''' Policy iteration algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        pi: initial guess for the policy
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        threshold: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    Q  = np.zeros(env.nu)           # temporary array to store value of different controls
    for k in range(maxImprIters):
        # evaluate current policy using policy evaluation
        V = policy_eval(env, gamma, pi, V, maxEvalIters, value_thr, False)
        if not k%nprint: 
            print('PI - Iter #%d done' % (k))
            plot_policy(env, pi)
            plot_V_table(env, V)
        
        pi_old = np.copy(pi) # make a copy of current policy table
        for x in range(env.nx):     # for every state
            for u in range(env.nu): # for every control
                env.reset(x)                        # set the environment state to x
                x_next,cost = env.step(u)           # apply the control u
                Q[u] = cost + gamma * V[x_next]  # store value associated to u
                
            # Rather than simply using argmin we do something slightly more complex
            # to ensure simmetry of the policy when multiply control inputs
            # result in the same value. In these cases we prefer the more extreme
            # actions
    #        pi[x] = np.argmin(Q)
            u_best = np.where(Q==np.min(Q))[0]
            if(u_best[0]>env.c2du(0.0)):    # if all the best action corresponds to a positive torque
                pi[x] = u_best[-1]          # take the largest torque
            elif(u_best[-1]<env.c2du(0.0)): # if all the best action corresponds to a negative torque
                pi[x] = u_best[0]           # take the smallest torque (largest in abs value)
            else:                           # otherwise take the average value among the best actions
                pi[x] = u_best[int(u_best.shape[0]/2)]
                
        # check for convergence
        pi_err = np.max(np.abs(pi-pi_old))
        if(pi_err<policy_thr):
            print("PI converged after %d iters with error"%k, pi_err)
            print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
            # 4.699 -9.99999 -3.13810
            plot_policy(env, pi)
            plot_V_table(env, V)
            return pi
            
        if not k%nprint: 
            print('PI - Iter #%d done' % (k))
            print("|pi - pi_old|=%.5f"%(pi_err))
    
    print("Policy iteration did NOT converge in %d iters. Error"%k, pi_err)
    return pi