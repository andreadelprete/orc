'''
Example of policy evaluation with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum
#from sol.ex_0_policy_evaluation_sol_prof import policy_eval
from sol.ex_0_policy_evaluation_sol import policy_eval

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
    return env.c2du(-env.uMax) # if velocity is null and q<=0 decelerate as much as possible


def render_policy(env, pi, x0=None, T=30):
    '''Roll-out (i.e., simulate) from state x0 using policy pi for T time steps'''
    x = env.reset(x0)
    for i in range(T):
        if(callable(pi)):   # if pi is a function
            u = pi(env, x)
        else:   # otherwise assume it's a vector
            u = pi[x]
        x_next,l = env.step(u)
        env.render()
#        q,dq = env.d2c(env.i2x(x))
#        print("u=", u, "%.2f"%env.d2cu(u), "x", x, "q=%.3f"%q, 
#              "dq=%.3f"%dq, "ddq=%.3f"%env.pendulum.a[0])
#        if l!=0: print('Cost not zero!');
        x = x_next


if __name__=="__main__":    
    ### --- Hyper paramaters
    MAX_ITERS         = 200         # Max number of iterations
    CONVERGENCE_THR   = 1e-4        # convergence threshold
    NPRINT            = 10          # Print info every NPRINT iterations
    PLOT              = True        # Plot the V table
    DISCOUNT          = 0.9        # Discount factor 
    
    ### --- Environment
    nq=51   # number of discretization steps for the joint angle q
    nv=21   # number of discretization steps for the joint velocity v
    nu=11   # number of discretization steps for the joint torque u
    env = DPendulum(nq, nv, nu) # create the environment
    kd = 1.0/env.dt             # derivative gain used in the control policy
    kp = kd**2 / 2              # proportional gain used in the control policy
    V  = np.zeros([env.nx])     # V-table initialized to 0
    
    # display policy behavior
#    render_policy(env, policy)
    
    V = policy_eval(env, DISCOUNT, policy, V, MAX_ITERS, CONVERGENCE_THR, PLOT, NPRINT)
    