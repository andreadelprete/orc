'''
Example of policy iteration with a simple discretized 1-DoF pendulum.
'''

import numpy as np
from dpendulum import DPendulum
from ex_0_policy_evaluation import render_policy
#from sol.ex_1_policy_iteration_sol_prof import policy_iteration
from sol.ex_1_policy_iteration_sol import policy_iteration
import time

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
MAX_EVAL_ITERS    = 200     # Max number of iterations for policy evaluation
MAX_IMPR_ITERS    = 20      # Max number of iterations for policy improvement
VALUE_THR         = 1e-3    # convergence threshold for policy evaluation
POLICY_THR        = 1e-4    # convergence threshold for policy improvement
DISCOUNT          = 0.9     # Discount factor 
NPRINT            = 1       # print some info every NPRINT iterations
PLOT              = False
nq=51   # number of discretization steps for the joint angle q
nv=21   # number of discretization steps for the joint velocity v
nu=11   # number of discretization steps for the joint torque u

### --- Environment
env = DPendulum(nq, nv, nu)
V  = np.zeros(env.nx)                       # Value table initialized to 0
pi = env.c2du(0.0)*np.ones(env.nx, np.int32)  # policy table initialized to zero torque
  
pi = policy_iteration(env, DISCOUNT, pi, V, MAX_EVAL_ITERS, MAX_IMPR_ITERS, VALUE_THR, POLICY_THR, PLOT, NPRINT)
        
render_policy(env, pi, env.x2i(env.c2d([np.pi,0.])))