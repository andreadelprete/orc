'''
Example of policy evaluation with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum, plot_V_table

def policy(env, x):
    ''' The policy to be evaluated '''
    q, dq = env.d2c(env.i2x(x)) # convert state from discrete to continuous
    if(abs(q)<2*env.DQ): # if joint position close to zero
        return env.c2du(-kp*q-kd*dq)    # PD control law
    if(dq>0): # if velocity is positive
        return env.c2du(env.uMax) # accelerate as much as possible
    if(dq<0): # if velocity is negative
        return env.c2du(-env.uMax) # decelerate as much as possible
    if(q>0): # if velocity is null and q>0
        return env.c2du(env.uMax) # accelerate as much as possible
    return env.c2du(-env.uMax) # if velocity is nul and q<=0 decelerate as much as possible


def render_policy(env, pi, x0=None, T=30):
    '''Roll-out from state x0 using policy pi for T time steps'''
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
            u = pi(env, x) # apply the given policy
            x_next, cost = env.step(u)
            
            # Update V-Table with Bellman's equation
            V[x] = cost + gamma*V_old[x_next]
    
        # compute the difference between the current and previous V table
        V_err = np.max(np.abs(V-V_old))
        if(V_err<threshold):    # check convergence
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
    ### --- Hyper paramaters
    MAX_ITERS         = 200         # Max number of iterations
    CONVERGENCE_THR   = 1e-4        # convergence threshold
    NPRINT            = 5           # Print info every NPRINT iterations
    DISCOUNT          = 0.9         # Discount factor 
    
    ### --- Environment
    nq=21   # number of discretization steps for the joint angle q
    nv=11   # number of discretization steps for the joint velocity v
    nu=11   # number of discretization steps for the joint torque u
    env = DPendulum(nq, nv, nu) # create the environment
    kd = 1.0/env.dt             # derivative gain used in the control policy
    kp = kd**2 / 2              # proportional gain used in the control policy
    V  = np.zeros([env.nx])     # V-table initialized to 0
    
    # display policy behavior
    render_policy(env, policy)
    
    V = policy_eval(env, DISCOUNT, policy, V, MAX_ITERS, CONVERGENCE_THR, True, NPRINT)
    