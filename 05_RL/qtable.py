'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
'''

import numpy as np
from dpendulum import DPendulum, plot_V_table, plot_policy
from policy_evaluation import policy_eval
import matplotlib.pyplot as plt
import time

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 5000         # Number of training episodes
NPRINT                  = 500          # print something every NPRINT episodes
NSTEPS                  = 100          # Max episode length
LEARNING_RATE           = 0.8          # alpha coefficient of Q learning algorithm
DISCOUNT                = 0.9          # Discount factor 

exploration_prob = 1   #initialize the exploration probability to 1
                        #exploartion decay for exponential decreasing
exploration_decreasing_decay = 0.001
min_exploration_prob = 0.001            # minimum of exploration proba

### --- Environment
nq=51
nv=21
nu=11
env = DPendulum(nq, nv, nu)
Q     = np.zeros([env.nx,env.nu])       # Q-table initialized to 0

def render_greedy_policy(env, Q, x0=None, maxiter=100):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    costToGo = 0.0
    gamma_i = 1
    for i in range(maxiter):
        u = np.argmin(Q[x,:])
#        print("State", x, "Control", u, "Q", Q[x,u])
        x,c = env.step(u)
        costToGo += gamma_i*c
        gamma_i *= DISCOUNT
        env.render()
    print("Real cost to go of state", x0, ":", costToGo)

def plot_Q_table(Q):
    X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
    plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('Q table')
    plt.show()
    
def compute_V_pi_from_Q(env, Q):
    V = np.zeros(Q.shape[0])
    pi = np.zeros(Q.shape[0], np.int)
    for x in range(Q.shape[0]):
#        pi[x] = np.argmin(Q[x,:])
        # Rather than simply using argmin we do something slightly more complex
        # to ensure simmetry of the policy when multiply control inputs
        # result in the same value. In these cases we prefer the more extreme
        # actions
        V[x] = np.min(Q[x,:])
        u_best = np.where(Q[x,:]==V[x])[0]
        if(u_best[0]>env.c2du(0.0)):
            pi[x] = u_best[-1]
        elif(u_best[-1]<env.c2du(0.0)):
            pi[x] = u_best[0]
        else:
            pi[x] = u_best[int(u_best.shape[0]/2)]
        
        
    return V, pi

h_rwd = []                              # Learning history (for plot).
Q_old = np.copy(Q)
for episode in range(1,NEPISODES):
    x    = env.reset()
    costToGo = 0.0
    for steps in range(NSTEPS):
        if np.random.uniform(0,1) < exploration_prob:
            u = np.random.randint(env.nu)
        else:
            u = np.argmin(Q[x,:]) # Greedy action
#        u         = np.argmax(Q[x,:] + np.random.randn(1,NU)/episode) # Greedy action with noise
        x_next,cost = env.step(u)

        # Compute reference Q-value at state x respecting HJB
        Qref = cost + DISCOUNT*np.min(Q[x_next,:])

        # Update Q-Table to better fit HJB
        Q[x,u] += LEARNING_RATE*(Qref-Q[x,u])
        x       = x_next
        costToGo   = cost + DISCOUNT*costToGo
#        if cost!=0: break

    exploration_prob = max(min_exploration_prob, 
                           np.exp(-exploration_decreasing_decay*episode))
    h_rwd.append(costToGo)
    if not episode%NPRINT: 
        print('Episode #%d done with cost %d and %.1f exploration prob' % (
              episode, np.mean(h_rwd[-NPRINT:]), 100*exploration_prob))
        print("max|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
        print("avg|Q - Q_old|=%.2f"%(np.mean(np.abs(Q-Q_old))))
#        plot_Q_table(Q)
#        render_greedy_policy(env, Q)
        V, pi = compute_V_pi_from_Q(env, Q)
        plot_V_table(env, V)
        plot_policy(env, pi)
        Q_old = np.copy(Q)

print("\nTraining finished")
V, pi = compute_V_pi_from_Q(env,Q)
plot_V_table(env, V)
plot_policy(env, pi)
print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 

print("Compute real Value function of greedy policy")
def policy(env, x):
    return pi[x]
MAX_EVAL_ITERS    = 200     # Max number of iterations for policy evaluation
VALUE_THR         = 1e-3    # convergence threshold for policy evaluation
V_pi = policy_eval(env, DISCOUNT, policy, V, MAX_EVAL_ITERS, VALUE_THR, False)
plot_V_table(env, V_pi)
print("Average/min/max Value:", np.mean(V_pi), np.min(V_pi), np.max(V_pi)) 

print("Total rate of success: %.3f" % (-sum(h_rwd)/NEPISODES))
render_greedy_policy(env, Q)
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()
