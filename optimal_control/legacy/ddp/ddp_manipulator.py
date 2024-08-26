# -*- coding: utf-8 -*-
"""

@author: adelpret
"""

import numpy as np
from ddp import DDPSolver
from ddp_linear import DDPSolverLinearDyn
import pinocchio as pin
    
class DDPSolverManipulator(DDPSolverLinearDyn):
    ''' 
        Derived class of DDPSolverLinearDyn implementing the multi-body dynamics of a manipulator.
        The task is defined by a quadratic cost: 
            sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: 
            sum_{i=0}^{N-1} lmbda ||u_i||.
    '''
    
    def __init__(self, name, robot, ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG=False, simu=None):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.robot = robot
        self.H_xx = H_xx
        self.h_x = h_x
        self.h_s = h_s
        self.lmbda = lmbda
        self.nx = h_x.shape[1]
        self.nu = robot.na
        self.dt = dt
        self.simu = simu
        
        nv = self.robot.nv # number of joints
        self.Fx = np.zeros((self.nx, self.nx))
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fu = np.zeros((self.nx, self.nu))
        self.dx = np.zeros(2*nv)
        
    ''' System dynamics '''
    def f(self, x, u):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        ddq = pin.aba(model, data, q, v, u)
        self.dx[:nv] = v
        self.dx[nv:] = ddq
        return x + self.dt * self.dx
           
    def f_x_fin_diff(self, x, u, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed with finite differences'''
        f0 = self.f(x, u)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u)
            Fx[:,i] = (fp-f0)/delta
        return Fx
        
    def f_u_fin_diff(self, x, u, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed with finite differences'''
        f0 = self.f(x, u)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up)
            Fu[:,i] = (fp-f0)/delta
                
        return Fu
        
    def f_x(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
                
        # first compute Jacobians for continuous time dynamics
        pin.computeABADerivatives(model, data, q, v, u)
        self.Fx[:nv, :nv] = 0.0
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fx[nv:, :nv] = data.ddq_dq
        self.Fx[nv:, nv:] = data.ddq_dv
        self.Fu[nv:, :] = data.Minv
        
        # Convert them to discrete time
        self.Fx = np.identity(2*nv) + dt * self.Fx
        self.Fu *= dt
        
        return self.Fx
    
    def f_u(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. u '''
        return self.Fu
        
    def callback(self, X, U):
        for i in range(0, N):
            time_start = time.time()
            self.simu.display(X[i,:self.robot.nq])
            time_spent = time.time() - time_start
            if(time_spent < self.dt):
                time.sleep(self.dt-time_spent)
            
        
    def start_simu(self, X, U, K, dt_sim):
        ratio = int(self.dt/dt_sim)
        N_sim = N * ratio
        x = np.copy(X[0,:])
        for i in range(0, N_sim):
            time_start = time.time()
    
            # compute the index corresponding to the DDP time step
            j = int(np.floor(i/ratio))
            # compute joint torques
            tau = U[j,:] + K[j,:,:] @ (x - X[j,:])        
            # compute dx
            self.f(x, tau)
            # integrate dynamics with Euler using the specified dt_sim
            x += dt_sim * self.dx
            self.simu.display(x[:self.robot.nq])
            
            time_spent = time.time() - time_start
            if(time_spent < dt_sim):
                time.sleep(dt_sim-time_spent)
        print("Simulation finished")

    
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import orc.utils.plot_utils as plut
    import time
    from orc.utils.robot_loaders import loadUR
    from orc.utils.robot_wrapper import RobotWrapper
    from orc.utils.robot_simulator import RobotSimulator
    import ddp_manipulator_conf as conf
    np.set_printoptions(precision=2, suppress=True);
    
    ''' Test DDP with a manipulator
    '''
    print("".center(conf.LINE_WIDTH,'#'))
    print(" DDP - Manipulator ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH,'#'), '\n')

    N = conf.N               # horizon size
    dt = conf.dt             # control time step
    mu = 1e-4                # initial regularization
    ddp_params = {}
    ddp_params['alpha_factor'] = 0.5
    ddp_params['mu_factor'] = 10.
    ddp_params['mu_max'] = 1e0
    ddp_params['min_alpha_to_increase_mu'] = 0.1
    ddp_params['min_cost_impr'] = 1e-1
    ddp_params['max_line_search_iter'] = 10
    ddp_params['exp_improvement_threshold'] = 1e-3
    ddp_params['max_iter'] = 100
    DEBUG = False;
        
    r = loadUR()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv
    
    n = nq+nv                       # state size
    m = robot.na                    # control size
    U_bar = np.zeros((N,m));        # initial guess for control inputs
    x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))  # initial state
    x_tasks = np.concatenate((conf.qT, np.zeros(robot.nv)))  # goal state
    N_task = N;                     # time step to reach goal state
    
    tau_g = robot.nle(conf.q0, np.zeros(robot.nv))
    for i in range(N):
        U_bar[i,:] = tau_g
    
    ''' TASK FUNCTION  '''
    lmbda = 1e-3           # control regularization
    H_xx = np.zeros((N+1, n, n));
    h_x  = np.zeros((N+1, n));
    h_s  = np.zeros(N+1);
    W = np.diagflat(np.concatenate([np.ones(nq), 1e-2*np.ones(nv)]))
    # cost = ||x - x_task||_W^2 
    for i in range(N_task):
        H_xx[i,:,:]  = W
        h_x[i,:]     = -W @ x_tasks
        h_s[i]       = 0.5*x_tasks.T @ W @ x_tasks
    
    print("Displaying desired goal configuration")
    simu = RobotSimulator(conf, robot)
    simu.display(conf.qT)
    time.sleep(1.)
    
    solver = DDPSolverManipulator("ur5", robot, ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG, simu)
    
    (X,U,K) = solver.solve(x0, U_bar, mu);
    solver.print_statistics(x0, U, K, X);
    
    print("Show reference motion")
    solver.callback(X,U)
    time.sleep(1)
    
    print("Show simulation without feedback gains")
    solver.start_simu(X, U, 0*K, conf.dt_sim)
    time.sleep(1)
    
    print("Show simulation with feedback gains")
    solver.start_simu(X, U, K, conf.dt_sim)
    
    print("Max value of K matrix for each time step")
    print(np.array([np.max(np.abs(K[i,:,:])) for i in range(N)]))
