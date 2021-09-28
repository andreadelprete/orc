'''
Example of policy evaluation with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum, plot_V_table
import matplotlib.pyplot as plt
import signal
import time

def policy(env, x):
    q, dq = env.d2c(env.i2x(x))
    if(abs(q)<2*env.DQ):
        return env.c2du(-kp*q-kd*dq)
    if(dq>0):
        return env.c2du(env.uMax)
    if(dq<0):
        return env.c2du(-env.uMax)
    if(q>0):
        return env.c2du(env.uMax)
    return env.c2du(-env.uMax)

def render_policy(env, pi, x0=None, T=30):
    '''Roll-out from random state using greedy policy.'''
    x = env.reset(x0)
    for i in range(T):
        u = pi(env, x)
        x_next,l = env.step(u)
        env.render()
#        q,dq = env.d2c(env.i2x(x))
#        print("u=", u, "%.2f"%env.d2cu(u), "x", x, "q=%.3f"%q, 
#              "dq=%.3f"%dq, "ddq=%.3f"%env.pendulum.a[0])
#        if l!=0: print('Cost not zero!');
        x = x_next

def policy_eval(env, gamma, pi, V, maxIters, threshold, plot=False, nprint=1000):
    for k in range(1, maxIters):
        V_old = np.copy(V)
        for x in range(env.nx):
            env.reset(x)
            u = pi(env, x)
            x_next, cost = env.step(u)
            
            # Update V-Table
            V[x] = cost + gamma*V_old[x_next]
    
        V_err = np.max(np.abs(V-V_old))
        V_old = np.copy(V)
        if(V_err<threshold):
            print("Policy eval converged after %d iters with error"%k, V_err)
            if(plot): plot_V_table(env, V)
            return V
            
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("|V - V_old|=%.5f"%(V_err))
            if(plot): plot_V_table(env, V)
    print("Policy eval did NOT converge in %d iters. Error"%k, V_err)
    return V

if __name__=="__main__":
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    MAX_ITERS         = 200          # Max number of iterations
    CONVERGENCE_THR   = 1e-4
    NPRINT            = 5
    DISCOUNT          = 0.9          # Discount factor 
    
    ### --- Environment
    nq=21
    nv=11
    nu=11
    env = DPendulum(nq, nv, nu)
    kd = 1.0/env.dt
    kp = kd**2 / 2
    V     = np.zeros([env.nx])       # V-table initialized to 0
    # plot V when CTRL-Z is pressed
    signal.signal(signal.SIGTSTP, lambda x,y:plot_V_table())
    
    # use this to display policy behavior
    #render_policy(env, policy)
    V = policy_eval(env, DISCOUNT, policy, V, MAX_ITERS, CONVERGENCE_THR, True, NPRINT)
    