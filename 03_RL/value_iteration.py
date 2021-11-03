'''
Example of Value iteration with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum
from dpendulum import plot_policy, plot_V_table
from policy_evaluation import render_policy

### --- Hyper paramaters
MAX_ITERS       = 200     # Max number of iterations for the algorirthm
VALUE_THR       = 1e-4    # convergence threshold
NPRINT          = 3       # print some info every NPRINT iterations
DISCOUNT        = 0.9     # Discount factor 
nq=51   # number of discretization steps for the joint angle q
nv=21   # number of discretization steps for the joint velocity v
nu=11   # number of discretization steps for the joint torque u

### --- Environment
env = DPendulum(nq, nv, nu)
V  = np.zeros(env.nx)           # Value table initialized to 0
    
def compute_policy_from_V(env, V):
    ''' Compute greedy policy with respect to given Value table V '''
    Q  = np.zeros(env.nu)   # temporary array to store value of different controls
    pi = np.zeros(env.nx)   # policy table
    for x in range(env.nx):     # for every state
        for u in range(env.nu): # for every action
            env.reset(x)        # reset state to x
            x_next,cost = env.step(u)   # apply action u
            Q[u] = cost + DISCOUNT * V[x_next] # store value associated to u
        
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
        
    return pi
    
Q  = np.zeros(env.nu)           # temporary array to store value of different controls
for k in range(MAX_ITERS):
    if not k%NPRINT: 
        plot_V_table(env, V)
    
    V_old = np.copy(V)  # make a copy of current Value table
    for x in range(env.nx):                     # for every state x
        for u in range(env.nu):                 # for every action u
            env.reset(x)                        # reset state to x
            x_next,cost = env.step(u)           # apply action u
            Q[u] = cost + DISCOUNT * V[x_next]  # store cost associated to u
        V[x] = np.min(Q)                        # update Value table
            
    # check for convergence
    V_err = np.max(np.abs(V-V_old))
    if(V_err<VALUE_THR):
        print("VI converged after %d iters with error"%k, V_err)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
        # -4.699560419916913 -9.999994614527743 -3.1381005754277433
        plot_V_table(env, V)
        break
        
    if not k%NPRINT: 
        print('VI - Iter #%d done' % (k))
        print("|V - V_old|=%.5f"%(V_err))
  
pi = compute_policy_from_V(env, V)      
plot_policy(env, pi)
#render_policy(env, policy, env.x2i(env.c2d([np.pi,0.])))
    