'''
Example of model-free policy evaluation with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum, plot_V_table
import matplotlib.pyplot as plt
import signal
import time

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


def mc_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    ''' Monte-Carlo Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    N = np.zeros(env.nx) # number of times each state has been visited
    C = np.zeros(env.nx) # cumulative cost associated to each state
    V = np.copy(V0)      # Value table
    V_err = []           # evolution of the error between real and estimated V table
    
    for k in range(nEpisodes): # for each episode
        env.reset() # reset the environment to a random state
        x_list = [] # list of states visited in this episode
        c_list = [] # list of costs received at each state in this episode
    
        for t in range(maxEpisodeLength):
            x = env.x2i(env.x)  # convert state from discrete 2d to discrete 1d representation
            x_list.append(x)    # store state
            u = pi(env, x)      # compute control action according to policy pi
            x_next, cost = env.step(u) # apply control action
            c_list.append(cost) # store cost
            
        # Update V-Table
        J = 0 # cost-to-go
        for i in range(len(x_list)-1,-1,-1): # for each visited state, going backward in time
            x = x_list[i]
            J = c_list[i] + gamma*J # compute cost-to-go of state x
            C[x] += J               # update cumulative cost associated to x
            N[x] += 1               # increment counter of visits to x
            V[x] = C[x] / N[x]      # update estimated Value of state x
            
        V_err.append(np.mean(np.abs(V-V_real))) # compute error of Value table
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("mean|V_mc - V_real|=%.5f"%(V_err[-1]))
            print("Nb of states not visited yet:", np.count_nonzero(N==0))
            if(plot): plot_V_table(env, V)
        
        if(np.min(N)>0):
            print("Stop at iter", k, "because all states visited at least once.")
            break
    return V, V_err


def td0_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        learningRate: learning rate of the algorithm
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    V = np.copy(V0)
    V_err = []      # evolution of the error between real and estimated V table
    for k in range(1,nEpisodes+1): # for each episode
        env.reset()     # reset environment to random initial state
    
        for t in range(maxEpisodeLength):
            x = env.x2i(env.x)  # convert state from discrete 2d to discrete 1d representation
            u = pi(env, x)      # compute control action according to policy pi
            x_next, cost = env.step(u)  # apply action u
            
            # Update V-Table
            TD_target = cost + gamma*V[x_next]
            V[x] += learningRate*(TD_target - V[x])
    
        
        V_err.append(np.mean(np.abs(V-V_real))) # compute error of Value table
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("mean|V_td - V_real|=%.5f"%(V_err[-1]))
            if(plot): plot_V_table(env, V)
        
    return V, V_err


if __name__=="__main__":
    ''' Compare learning Value table using Monte Carlo and TD(0) with different learning rates. '''
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    N_EPISODS         = 1000    # number of episodes
    MAX_EPISOD_LENGTH = 100     # max length of an episode
    LEARNING_RATES    = [0.5, 0.8, 1.0]     # TD0 learning rates
    CONVERGENCE_THR   = 1e-3    # convergence threshold of policy evaluation
    NPRINT            = 100     # print some info every NPRINT iterations
    DO_PLOTS          = False   # if True it plots the Value table every NPRINT iterations
    DISCOUNT          = 0.9     # Discount factor 
    NOISE_STDDEV      = 0.1     # standard deviation of the noise acting on the dynamics
    
    ### --- Environment
    nq=21   # number of discretization steps for the joint angle q
    nv=11   # number of discretization steps for the joint velocity v
    nu=11   # number of discretization steps for the joint torque u
    env = DPendulum(nq, nv, nu, noise_stddev=0)
    kd = 1.0/env.dt             # derivative gain used in the control policy
    kp = kd**2 / 2              # proportional gain used in the control policy
    V  = np.zeros([env.nx])     # V-table initialized to 0
    
    # use this to display policy behavior
    #render_policy(env, policy)
    from policy_evaluation import policy_eval
    print("\nGonna compute real Value function")
    V_real = policy_eval(env, DISCOUNT, policy, np.copy(V), N_EPISODS, CONVERGENCE_THR, DO_PLOTS)
    print("*** Real value function ***")
    plot_V_table(env, V_real)
    
    # set noise standard deviations
    env.pendulum.noise_stddev = NOISE_STDDEV
    
    print("\nEstimate Value function with MC")
    V_mc, err_mc = mc_policy_eval(env, DISCOUNT, policy, V, N_EPISODS, MAX_EPISOD_LENGTH,
                       V_real, DO_PLOTS, NPRINT)
    
    V_td = len(LEARNING_RATES)*[None]
    err_td = len(LEARNING_RATES)*[None]
    for i,alpha in enumerate(LEARNING_RATES):
        print("\nEstimate Value function with TD0. Learning rate", alpha)
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
    