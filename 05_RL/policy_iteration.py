'''
Example of policy iteration with a simple discretized 1-DoF pendulum.
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
MAX_EVAL_ITERS    = 200     # Max number of iterations for policy evaluation
MAX_IMPR_ITERS    = 20     # Max number of iterations for policy improvement
VALUE_THR         = 1e-3    # convergence threshold for policy evaluation
POLICY_THR        = 1e-4    # convergence threshold for policy improvement
NPRINT            = 1
DISCOUNT        = 0.9          # Discount factor 

nq=51
nv=21
nu=11

### --- Environment
env = DPendulum(nq, nv, nu)
V  = np.zeros(env.nx)           # Value function initialized to 0
pi = env.c2du(0.0)*np.ones(env.nx, np.int)   # policy

def policy(env, x):
    return pi[x]

def plot_policy(env, pi):
    Q,DQ = np.meshgrid([env.d2cq(i) for i in range(env.nq)], 
                        [env.d2cv(i) for i in range(env.nv)])
    plt.pcolormesh(Q, DQ, pi.reshape((env.nv,env.nq)), cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.title('Policy')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()
    
Q  = np.zeros(env.nu)           # temporary array to store value of different controls
for k in range(MAX_IMPR_ITERS):
    V = policy_eval(env, DISCOUNT, policy, V, MAX_EVAL_ITERS, VALUE_THR, False)
    if not k%NPRINT: 
        print('PI - Iter #%d done' % (k))
        plot_policy(env, pi)
        plot_V_table(env, V)
    
    pi_old = np.copy(pi)
    for x in range(env.nx):
        for u in range(env.nu):
            env.reset(x)
            x_next,cost = env.step(u)
            Q[u] = cost + DISCOUNT * V[x_next]
        # Rather than simply using argmin we do something slightly more complex
        # to ensure simmetry of the policy when multiply control inputs
        # result in the same value. In these cases we prefer the more extreme
        # actions
#        pi[x] = np.argmin(Q)
        u_best = np.where(Q==np.min(Q))[0]
        if(u_best[0]>env.c2du(0.0)):
            pi[x] = u_best[-1]
        elif(u_best[-1]<env.c2du(0.0)):
            pi[x] = u_best[0]
        else:
            pi[x] = u_best[int(u_best.shape[0]/2)]
            
    pi_err = np.max(np.abs(pi-pi_old))
    if(pi_err<POLICY_THR):
        print("PI converged after %d iters with error"%k, pi_err)
        print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 
        # 4.699560419916913 9.999994614527743 -3.1381005754277433
        plot_policy(env, pi)
        plot_V_table(env, V)
        break
        
    if not k%NPRINT: 
        print('PI - Iter #%d done' % (k))
        print("|pi - pi_old|=%.5f"%(pi_err))
        
render_policy(env, policy, env.x2i(env.c2d([np.pi,0.])))