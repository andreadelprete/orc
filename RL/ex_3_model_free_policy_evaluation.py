'''
Example of model-free policy evaluation with a simple discretized 1-DoF pendulum.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
#from sol.ex_3_model_free_policy_evaluation_sol_prof import mc_policy_eval, td0_policy_eval
from sol.ex_3_model_free_policy_evaluation_sol import mc_policy_eval, td0_policy_eval

def policy_pendulum(env, x):
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


if __name__=="__main__":
    ''' Compare learning Value table using Monte Carlo and TD(0) with different learning rates. '''
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    DO_PLOTS          = 1   # if True it plots the Value table every NPRINT iterations
    
    ### --- Environment
    env_name = 'pendulum'
#    env_name = 'ABC'
    
    if(env_name == 'pendulum'):
        DISCOUNT          = 0.9     # Discount factor 
        N_EPISODS         = 1000    # number of episodes
        MAX_EPISOD_LENGTH = 100     # max length of an episode
        LEARNING_RATES    = [0.5, 1.0]     # TD0 learning rates
        CONVERGENCE_THR   = 1e-5    # convergence threshold of policy evaluation
        NOISE_STDDEV      = 0.0     # standard deviation of the noise acting on the dynamics
        NPRINT            = 200     # print some info every NPRINT iterations
        
        from dpendulum import DPendulum
        nq=21   # number of discretization steps for the joint angle q
        nv=11   # number of discretization steps for the joint velocity v
        nu=11   # number of discretization steps for the joint torque u
        env = DPendulum(nq, nv, nu, noise_stddev=0)
        kd = 1.0/env.dt             # derivative gain used in the control policy
        kp = kd**2 / 2              # proportional gain used in the control policy
        policy = policy_pendulum
        
        # use this to display policy behavior
        #render_policy(env, policy)
        from sol.ex_0_policy_evaluation_sol_prof import policy_eval
        print("\nGonna compute real Value function")
        V_real = policy_eval(env, DISCOUNT, policy, np.zeros(env.nx), N_EPISODS, CONVERGENCE_THR, False)
    elif(env_name == 'ABC'):
        DISCOUNT          = 1     # Discount factor 
        N_EPISODS         = 100    # number of episodes
        MAX_EPISOD_LENGTH = 3     # max length of an episode
        LEARNING_RATES    = np.array([0.2])     # TD0 learning rates
        NPRINT            = 1 #N_EPISODS-1    # print some info every NPRINT iterations
        
        from abc_example import AbcExample
        env = AbcExample()
        policy = np.zeros(env.nx)
        V_real = np.array([0.5, 0.5, 0.5, 0.0])
    
    print("*** Real value function ***")
    V  = np.zeros([env.nx])     # V-table initialized to 0
    env.plot_V_table(V_real)
    
    if(env_name == 'pendulum'):
        # set noise standard deviations
        env.pendulum.noise_stddev = NOISE_STDDEV
    
    print("\nEstimate Value function with MC")
    V_mc, err_mc = mc_policy_eval(env, DISCOUNT, policy, N_EPISODS, MAX_EPISOD_LENGTH,
                       V_real, DO_PLOTS, NPRINT)
    
    V_td = len(LEARNING_RATES)*[None]
    err_td = len(LEARNING_RATES)*[None]
    for i,alpha in enumerate(LEARNING_RATES):
        print("\nEstimate Value function with TD0. Learning rate", alpha)
        time.sleep(1)
        V_td[i], err_td[i] = td0_policy_eval(env, DISCOUNT, policy, V, N_EPISODS, 
                                        MAX_EPISOD_LENGTH, V_real, alpha, DO_PLOTS, NPRINT)
    
    plt.figure()
    plt.plot(err_mc, label="MC")
    for i, alpha in enumerate(LEARNING_RATES):
        plt.plot(err_td[i], label="TD0 alpha=%.1f"%alpha)
    plt.title("Estimation error")
    plt.xlabel("Episodes")
    plt.legend(loc="best")
    plt.show()
    