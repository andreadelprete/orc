'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
'''

import numpy as np
from dpendulum import DPendulum
from sol.ex_0_policy_evaluation_sol_prof import policy_eval
from sol.ex_4_sarsa_sol import sarsa
# from sol.ex_4_sarsa_sol_prof import sarsa
import matplotlib.pyplot as plt
import time

def render_greedy_policy(env, Q, gamma, x0=None, maxiter=20):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    costToGo = 0.0
    gamma_i = 1
    for i in range(maxiter):
        u = np.argmin(Q[x,:])
#        print("State", x, "Control", u, "Q", Q[x,u])
        x,c = env.step(u)
        costToGo += gamma_i*c
        gamma_i *= gamma
        env.render()
    print("Real cost to go of state", x0, ":", costToGo)
    
def compute_V_pi_from_Q(env, Q):
    ''' Compute Value table and greedy policy pi from Q table. '''
    V = np.zeros(Q.shape[0])
    pi = np.zeros(Q.shape[0], np.int32)
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

if __name__=="__main__":
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ### --- Hyper paramaters
    NEPISODES               = 5000          # number of training episodes
    NPRINT                  = 500           # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 100           # max episode length
    LEARNING_RATE           = 0.8           # alpha coefficient of Q learning algorithm
    DISCOUNT                = 0.9           # discount factor 
    PLOT                    = False         # plot stuff if True
    epsilon                 = 1             # initialize the exploration probability to 1
    epsilon_decreasing_decay = 0.001        # exploration decay for exponential decreasing
    min_epsilon             = 0.001         # minimum of exploration proba

    ### --- Environment
    nq=51   # number of discretization steps for the joint angle q
    nv=21   # number of discretization steps for the joint velocity v
    nu=11   # number of discretization steps for the joint torque u
    env = DPendulum(nq, nv, nu, open_viewer=False)
    Q     = np.zeros([env.nx,env.nu])   # Q-table initialized to 0

    Q, h_ctg = sarsa(env, DISCOUNT, Q, NEPISODES, MAX_EPISODE_LENGTH, 
                    LEARNING_RATE, epsilon, epsilon_decreasing_decay,
                    min_epsilon, compute_V_pi_from_Q, PLOT, NPRINT)

    print("\nTraining finished")
    V, pi = compute_V_pi_from_Q(env,Q)
    # env.plot_policy(pi)
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
