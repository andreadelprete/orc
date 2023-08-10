from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
    

class DPendulum:
    ''' Discrete Pendulum environment. Joint angle, velocity and torque are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, nq=51, nv=21, nu=11, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0, which_viewer='meshcat'):
        self.pendulum = Pendulum(1,noise_stddev, which_viewer)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.nq = nq        # Number of discretization steps for joint angle
        self.nv = nv        # Number of discretization steps for joint velocity
        self.vMax = vMax    # Max velocity (v in [-vmax,vmax])
        self.nu = nu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.dt = dt        # time step
        self.DQ = 2*pi/nq   # discretization resolution for joint angle
        self.DV = 2*vMax/nv # discretization resolution for joint velocity
        self.DU = 2*uMax/nu # discretization resolution for joint torque

    @property
    def nqv(self): return [self.nq,self.nv]
    @property
    def nx(self): return self.nq*self.nv
    @property
    def goal(self): return self.x2i(self.c2d([0.,0.]))
    
    # Continuous to discrete
    def c2dq(self, q):
        q = (q+pi)%(2*pi)
        return int(np.floor(q/self.DQ))  % self.nq
    
    def c2dv(self, v):
        v = np.clip(v,-self.vMax+1e-3,self.vMax-1e-3)
        return int(np.floor((v+self.vMax)/self.DV))
    
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))
    
    def c2d(self, qv):
        '''From continuous to 2d discrete.'''
        return np.array([self.c2dq(qv[0]), self.c2dv(qv[1])])
    
    # Discrete to continuous
    def d2cq(self, iq):
        iq = np.clip(iq,0,self.nq-1)
        return iq*self.DQ - pi + 0.5*self.DQ
    
    def d2cv(self, iv):
        iv = np.clip(iv,0,self.nv-1) - (self.nv-1)/2
        return iv*self.DV
    
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def d2c(self, iqv):
        '''From 2d discrete to continuous'''
        return np.array([self.d2cq(iqv[0]), self.d2cv(iqv[1])])
    
    ''' From 2d discrete to 1d discrete '''
    def x2i(self, x): return x[0]+x[1]*self.nq
    
    ''' From 1d discrete to 2d discrete '''
    def i2x(self, i): return [ i%self.nq, int(np.floor(i/self.nq)) ]

    def reset(self,x=None):
        if x is None:
            self.x = np.random.randint(0,self.nx)
        else: 
            self.x = x
        return self.x

    def step(self,iu):
        cost     = -1 if self.x==self.goal else 0
        self.x   = self.dynamics(self.x,iu)
        return self.x, cost

    def render(self):
        q = self.d2cq(self.i2x(self.x)[0])
        self.pendulum.display(np.array([q,]))
        time.sleep(self.pendulum.DT)

    def dynamics(self,ix,iu):
        x   = self.d2c(self.i2x(ix))
        u   = self.d2cu(iu)
        
        self.xc,_ = self.pendulum.dynamics(x,u)
        return self.x2i(self.c2d(self.xc))
    
    def plot_V_table(self, V):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt
        Q,DQ = np.meshgrid([self.d2cq(i) for i in range(self.nq)], 
                            [self.d2cv(i) for i in range(self.nv)])
        plt.pcolormesh(Q, DQ, V.reshape((self.nv,self.nq)), cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('V table')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_policy(self, pi):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt
        Q,DQ = np.meshgrid([self.d2cq(i) for i in range(self.nq)], 
                            [self.d2cv(i) for i in range(self.nv)])
        plt.pcolormesh(Q, DQ, pi.reshape((self.nv,self.nq)), cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title('Policy')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_Q_table(self, Q):
        ''' Plot the given Q table '''
        import matplotlib.pyplot as plt
        X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
        plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('Q table')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.show()
    
if __name__=="__main__":
    print("Start tests")
    env = DPendulum()
    nq = env.nq
    nv = env.nv
    
    # sanity checks
    for i in range(nq*nv):
        x = env.i2x(i)
        i_test = env.x2i(x)
        if(i!=i_test):
            print("ERROR! x2i(i2x(i))=", i_test, "!= i=", i)
        
        xc = env.d2c(x)
        x_test = env.c2d(xc)
        if(x_test[0]!=x[0] or x_test[1]!=x[1]):
            print("ERROR! c2d(d2c(x))=", x_test, "!= x=", x)
        xc_test = env.d2c(x_test)
        if(np.linalg.norm(xc-xc_test)>1e-10):
            print("ERROR! xc=", xc, "xc_test=", xc_test)
    print("Tests finished")
    