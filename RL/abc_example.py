import numpy as np
from scipy.stats import bernoulli

class AbcExample:
    ''' MDP seen in class as " ABC example".
    '''
    def __init__(self, terminal_cost_prob=0.5):
        self.terminal_cost_prob = terminal_cost_prob
        self.x = 0
        
    @property
    def nx(self): return 4
    
    @property
    def nu(self): return 1
    
    def reset(self,x=None):
        if x is None:
            # initial state is 0 with probability 90%, 1 with probability 10%
            self.x = bernoulli.rvs(0.1)
        else: 
            self.x = x

    def step(self,iu):
        if self.x == 2:
            cost = bernoulli.rvs(self.terminal_cost_prob)
            self.x = 3
        else:
            cost = 0
            if(self.x<2): # from state 0 and 1 transition to state 2
                self.x = 2
        return self.x, cost

    def render(self):
        pass
    
    def plot_V_table(self, V):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt
        x = np.arange(self.nx)
        plt.plot(x, V, 'o')
        plt.title('V table')
        plt.xlabel("x")
        plt.ylabel("V")
        plt.ylim([0, 1])
        plt.show()
        
    def plot_policy(self, pi):
        ''' Plot the given policy table pi '''
        pass
    
if __name__=="__main__":
    pass
    