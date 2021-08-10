'''
Example of Value iteration with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum
from policy_evaluation import policy_eval, plot_V_table, render_policy
import matplotlib.pyplot as plt
import signal
import time

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
MAX_ITERS       = 200     # Max number of iterations for the algorirthm
VALUE_THR       = 1e-3    # convergence threshold
NPRINT          = 1
DISCOUNT        = 0.9          # Discount factor 
nq=21
nv=21
nu=11

### --- Environment
env = DPendulum(nq, nv, nu)
V  = np.zeros(env.nx)           # Value function initialized to 0

def plot_policy(env, pi):
    Q,DQ = np.meshgrid([env.d2cq(i) for i in range(env.nq)], 
                        [env.d2cv(i) for i in range(env.nv)])
    plt.pcolormesh(Q, DQ, pi.reshape((env.nv,env.nq)), cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.title('Policy')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()
    
def compute_policy_from_V(env, V):
    Q  = np.zeros(env.nu)  # temporary array to store value of different controls
    pi = np.zeros(env.nx)   # policy
    for x in range(env.nx):
        for u in range(env.nu):
            env.reset(x)
            x_next,cost = env.step(u)
            Q[u] = cost + DISCOUNT * V[x_next]
        pi[x] = np.argmin(Q)
    return pi
    
Q  = np.zeros(env.nu)           # temporary array to store value of different controls
for k in range(MAX_ITERS):
    if not k%NPRINT: 
        plot_V_table(env, V)
    
    V_old = np.copy(V)
    for x in range(env.nx):
        for u in range(env.nu):
            env.reset(x)
            x_next,cost = env.step(u)
            Q[u] = cost + DISCOUNT * V[x_next]
        V[x] = np.min(Q)
            
    V_err = np.max(np.abs(V-V_old))
    if(V_err<VALUE_THR):
        print("VI converged after %d iters with error"%k, V_err)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
        # 4.699560419916913 9.999994614527743 -3.1381005754277433
        plot_V_table(env, V)
        break
        
    if not k%NPRINT: 
        print('VI - Iter #%d done' % (k))
        print("|V - V_old|=%.5f"%(V_err))
  
pi = compute_policy_from_V(env, V)      
plot_policy(env, pi)
#render_policy(env, policy, env.x2i(env.c2d([np.pi,0.])))
    