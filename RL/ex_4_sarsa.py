'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
'''

import numpy as np
from dpendulum import DPendulum
from sol.ex_0_policy_evaluation_sol import policy_eval
from sol.ex_5_sarsa_sol import sarsa
#from sol.ex_5_sarsa_sol_prof import sarsa
import matplotlib.pyplot as plt
import time
from ex_4_q_learning import render_greedy_policy, compute_V_pi_from_Q

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
NITER                   = 10            # number of iterations of the algorithm
NEPISODES               = 500           # number of training episodes
NPRINT                  = 1             # print something every NPRINT iterations
MAX_EPISODE_LENGTH      = 100           # max episode length
LEARNING_RATE           = 0.8           # alpha coefficient of Q learning algorithm
DISCOUNT                = 0.9           # discount factor 
PLOT                    = True          # plot stuff if True
exploration_prob                = 1     # initialize the exploration probability to 1
exploration_decreasing_decay    = 0.001 # exploration decay for exponential decreasing
min_exploration_prob            = 0.001 # minimum of exploration proba

### --- Environment
nq=51   # number of discretization steps for the joint angle q
nv=21   # number of discretization steps for the joint velocity v
nu=11   # number of discretization steps for the joint torque u
env = DPendulum(nq, nv, nu)
Q     = np.zeros([env.nx,env.nu])   # Q-table initialized to 0
pi = env.c2du(0.0)*np.ones(env.nx, np.int)   # policy

Q, h_ctg = sarsa(env, DISCOUNT, Q, pi, NITER, NEPISODES, MAX_EPISODE_LENGTH, 
                 LEARNING_RATE, exploration_prob, exploration_decreasing_decay,
                 min_exploration_prob, compute_V_pi_from_Q, PLOT, NPRINT)

print("\nTraining finished")
V, pi = compute_V_pi_from_Q(env,Q)
env.plot_policy(pi)
env.plot_V_table(V)
print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 

print("\nCompute real Value function of greedy policy")
MAX_EVAL_ITERS    = 200     # Max number of iterations for policy evaluation
VALUE_THR         = 1e-3    # convergence threshold for policy evaluation
V_pi = policy_eval(env, DISCOUNT, pi, V, MAX_EVAL_ITERS, VALUE_THR, False)
env.plot_V_table(V_pi)
print("Average/min/max Value:", np.mean(V_pi), np.min(V_pi), np.max(V_pi)) 

render_greedy_policy(env, Q, DISCOUNT)
plt.plot( np.cumsum(h_ctg)/range(1,len(h_ctg)+1) )
plt.title("Average cost-to-go")
plt.show()
