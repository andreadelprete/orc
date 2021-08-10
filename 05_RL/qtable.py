'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
'''

import numpy as np
from dpendulum import DPendulum
import matplotlib.pyplot as plt
import signal
import time

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 100000          # Number of training episodes
NPRINT                  = 100
NSTEPS                  = 100           # Max episode length
LEARNING_RATE           = 0.5          # 
DECAY_RATE              = 0.99          # Discount factor 

exploration_prob = 1   #initialize the exploration probability to 1
                        #exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.003
min_exploration_prob = 0.001            # minimum of exploration proba


### --- Environment
env = DPendulum()
NX  = env.nx                            # Number of (discrete) states
NU  = env.nu                            # Number of (discrete) controls

Q     = np.zeros([env.nx,env.nu])       # Q-table initialized to 0

def rendertrial(maxiter=100):
    '''Roll-out from random state using greedy policy.'''
    s = env.reset()
    for i in range(maxiter):
        a = np.argmax(Q[s,:])
        s,r = env.step(a)
        env.render()
        if r==1: print('Reward!'); break

def plot_Q_table(Q):
    X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
    plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('Q table')
    #plt.axis([-3, 3, -3, 3])
    plt.show()

signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed
h_rwd = []                              # Learning history (for plot).
Q_old = np.copy(Q)
for episode in range(1,NEPISODES):
    x    = env.reset()
    rsum = 0.0
    for steps in range(NSTEPS):
        if np.random.uniform(0,1) < exploration_prob:
            u = np.random.randint(NU)
        else:
            u = np.argmax(Q[x,:]) # Greedy action
#        u         = np.argmax(Q[x,:] + np.random.randn(1,NU)/episode) # Greedy action with noise
        x2,reward = env.step(u)

        # Compute reference Q-value at state x respecting HJB
        Qref = reward + DECAY_RATE*np.max(Q[x2,:])

        # Update Q-Table to better fit HJB
        Q[x,u] += LEARNING_RATE*(Qref-Q[x,u])
        if(abs(Qref-Q[x,u]) > 5):
            print("TD error>5, Qref", Qref, "Q", Q[x,u], "reward", reward, "x", x, "u", u, "x2", x2)
        x       = x2
        rsum   += reward
        if reward==1:             break

    exploration_prob = max(min_exploration_prob, 
                           np.exp(-exploration_decreasing_decay*episode))
    h_rwd.append(rsum)
    if not episode%NPRINT: 
        print('Episode #%d done with %d successes and %.1f exploration prob' % (
              episode,sum(h_rwd[-NPRINT:]), 100*exploration_prob))
        print("|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
#        plot_Q_table(Q)
        Q_old = np.copy(Q)

print("Total rate of success: %.3f" % (sum(h_rwd)/NEPISODES))
rendertrial()
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()
