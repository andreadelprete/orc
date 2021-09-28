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

def mc_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    N = np.zeros(env.nx)
    C = np.zeros(env.nx)
    V = np.copy(V0)
    V_err = []
    
    for k in range(nEpisodes):
        env.reset()
        x_list = []
        c_list = []
    
        for t in range(maxEpisodeLength):
            x = env.x2i(env.x)
            x_list.append(x)
            u = pi(env, x)
            x_next, cost = env.step(u)
            c_list.append(cost)
            
        # Update V-Table
        J = 0 # cost-to-go
        for i in range(len(x_list)-1,-1,-1):
            x = x_list[i]
            J = c_list[i] + gamma*J
            C[x] += J
            N[x] += 1
            V[x] = C[x] / N[x]
            
        V_err.append(np.mean(np.abs(V-V_real)))
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("mean|V_mc - V_real|=%.5f"%(V_err[-1]))
            print("Nb of states not visited yet:", np.count_nonzero(N==0))
            if(plot): plot_V_table(env, V)
        
        if(np.min(N)>0):
            print("Stop because all states visited at least once.")
            break
    return V, V_err

def td0_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    V = np.copy(V0)
    V_err = []
    for k in range(1,nEpisodes+1):
        env.reset()
    
        for t in range(maxEpisodeLength):
            x = env.x2i(env.x)
            u = pi(env, x)
            x_next, cost = env.step(u)
            
            # Update V-Table
            TD_target = cost + gamma*V[x_next]
            V[x] += learningRate*(TD_target - V[x])
    
        V_err.append(np.mean(np.abs(V-V_real)))
        if not k%nprint: 
            print('Iter #%d done' % (k))
            print("mean|V_td - V_real|=%.5f"%(V_err[-1]))
            if(plot): plot_V_table(env, V)
        
    return V, V_err

if __name__=="__main__":
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    N_EPISODS         = 1000    # number of episodes
    MAX_EPISOD_LENGTH = 100     # max length of an episode
    LEARNING_RATES    = [0.5, 0.8, 1.0]     # TD0 learning rate
    CONVERGENCE_THR   = 1e-3    # convergence threshold of policy evaluation
    NPRINT            = 100
    DO_PLOTS          = False
    DISCOUNT          = 0.9     # Discount factor 
    NOISE_STDDEV      = 0.05     # standard deviation
    
    ### --- Environment
    nq=21
    nv=11
    nu=11
    env = DPendulum(nq, nv, nu, noise_stddev=NOISE_STDDEV)
    kd = 1.0/env.dt
    kp = kd**2 / 2
    V     = np.zeros([env.nx])       # V-table initialized to 0
    # plot V when CTRL-Z is pressed
    signal.signal(signal.SIGTSTP, lambda x,y:plot_V_table(env,V))
    
    # use this to display policy behavior
    #render_policy(env, policy)
    from policy_evaluation import policy_eval
    print("\nGonna compute real Value function")
    V_real = policy_eval(env, DISCOUNT, policy, np.copy(V), N_EPISODS, CONVERGENCE_THR, DO_PLOTS)
    print("*** Real value function ***")
    plot_V_table(env, V_real)
    
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
    