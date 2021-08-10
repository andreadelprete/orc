'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
'''

import numpy as np
from dpendulum import DPendulum, c2d, d2c, c2du, d2cu, i2x, x2i, d2cq, d2cv
import dpendulum
import matplotlib.pyplot as plt
import signal
import time

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
MAX_ITERS         = 200          # Max number of iterations
CONVERGENCE_THR   = 1e-4
NPRINT            = 5
DECAY_RATE        = 0.9          # Discount factor 

### --- Environment
env = DPendulum()
NX  = env.nx                            # Number of (discrete) states
NQ, NV = dpendulum.NQ, dpendulum.NV
NU  = env.nu                            # Number of (discrete) controls
kd = 1.0/dpendulum.DT
kp = kd**2 / 2
V     = np.zeros([env.nx])       # V-table initialized to 0

def policy(q, dq):
    if(abs(q)<2*dpendulum.DQ):
        return c2du(-kp*q-kd*dq)
    if(dq>0):
        return c2du(dpendulum.UMAX)
    if(dq<0):
        return c2du(-dpendulum.UMAX)
    if(q>0):
        return c2du(dpendulum.UMAX)
    return c2du(-dpendulum.UMAX)

def render_policy(maxiter=50):
    '''Roll-out from random state using greedy policy.'''
    print("RENDER POLICY")
    x = env.reset()
    for i in range(maxiter):
        q,dq = d2c(i2x(x))
        u = policy(q,dq)
        x_next,l = env.step(u)
        env.render()
        print("u=", u, "%.2f"%d2cu(u), "x", x, "q=%.3f"%q, "dq=%.3f"%dq, "ddq=%.3f"%env.pendulum.a[0])
        
#        q_next,dq_next = d2c(i2x(x_next))
#        print("Time=", i, "x=", x, "q=", q, "dq=", dq, "u=", u, 
#              "x next=", x_next, "qnext=", q_next, "dqnext=", dq_next)
        x = x_next
        if l!=0: 
            print('Cost not zero!');
    print("RENDER POLICY ENDED\n")

def plot_V_table():
    Q,DQ = np.meshgrid([d2cq(i) for i in range(NQ)], [d2cv(i) for i in range(NV)])
    plt.pcolormesh(Q, DQ, V.reshape((NV,NQ)), cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('V table')
    plt.show()

# plot V when CTRL-Z is pressed
signal.signal(signal.SIGTSTP, lambda x,y:plot_V_table())

# use this to display policy behavior
#render_policy()

for k in range(1,MAX_ITERS):
    V_old = np.copy(V)
    for x in range(NX):
        env.reset(x)
        q, dq = d2c(i2x(x))
        u = policy(q, dq)
        x_next,cost = env.step(u)
        
        # Update V-Table
        V[x] = cost + DECAY_RATE*V_old[x_next]

    V_err = np.max(np.abs(V-V_old))
    V_old = np.copy(V)
    if(V_err<CONVERGENCE_THR):
        print("Algorithm converged after %d iterations with error"%k, V_err)
        plot_V_table()
        break
        
    if not k%NPRINT: 
        print('Iter #%d done' % (k))
        print("|V - V_old|=%.5f"%(V_err))
        plot_V_table()
        
    