# -*- coding: utf-8 -*-
"""
Derived class of DDPSolver implementing linear dynamics and quadratic cost.

@author: adelpret
"""

import numpy as np
from math import sin, cos
from ddp import DDPSolver
    
    
class DDPSolverLinearDyn(DDPSolver):
    ''' The linear system dynamics are defined by:
            x_{t+1} = A x_t + B u_t
        The task is defined by a quadratic cost: 
            sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: 
            sum_{i=0}^{N-1} lmbda ||u_i||^2
    '''
    
    def __init__(self, name, ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG=False):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.H_xx = H_xx
        self.h_x = h_x
        self.h_s = h_s
        self.lmbda = lmbda
        self.dt = dt
        self.nx = h_x.shape[1]
        self.nu = self.nx
        
    ''' System dynamics '''
    def f(self, x, u):
        return x + self.dt*u
           
    ''' Partial derivatives of system dynamics w.r.t. x '''
    def f_x(self, x, u):
        return 1
    
    ''' Partial derivatives of system dynamics w.r.t. u '''       
    def f_u(self, x, u):
        return self.dt
        
    def cost(self, X, U):
        ''' total cost (running+final) for state trajectory X and control trajectory U '''
        N = U.shape[0]
        cost = self.cost_final(X[-1,:])
        for i in range(N):
            cost += self.cost_running(i, X[i,:], U[i,:])
        return cost
        
    def cost_running(self, i, x, u):
        ''' Running cost at time step i for state x and control u '''
        cost = 0.5*np.dot(x, np.dot(self.H_xx[i,:,:], x)) \
                + np.dot(self.h_x[i,:].T, x) + self.h_s[i] \
                + 0.5*self.lmbda*np.dot(u.T, u)
        return cost
        
    def cost_final(self, x):
        ''' Final cost for state x '''
        cost = 0.5*np.dot(x, np.dot(self.H_xx[-1,:,:], x)) \
                + np.dot(self.h_x[-1,:].T, x) + self.h_s[-1]
        return cost
        
    def cost_running_x(self, i, x, u):
        ''' Gradient of the running cost w.r.t. x '''
        c_x = self.h_x[i,:] + np.dot(self.H_xx[i,:,:], x)
        return c_x
        
    def cost_final_x(self, x):
        ''' Gradient of the final cost w.r.t. x '''
        c_x = self.h_x[-1,:] + np.dot(self.H_xx[-1,:,:], x)
        return c_x
        
    def cost_running_u(self, i, x, u):
        ''' Gradient of the running cost w.r.t. u '''
        c_u = self.lmbda * u
        return c_u
        
    def cost_running_xx(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x '''
        return self.H_xx[i,:,:]
        
    def cost_final_xx(self, x):
        ''' Hessian of the final cost w.r.t. x '''
        return self.H_xx[-1,:,:]
        
    def cost_running_uu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. u '''
        return self.lmbda * np.eye(self.nu)
        
    def cost_running_xu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x and then w.r.t. u '''
        return np.zeros((self.nx, self.nu))


class DDPSolverSinDyn(DDPSolverLinearDyn):
    ''' Simple 1d nonlinear dynamical system '''
    
    def f(self, x, u):
        ''' System dynamics '''
        return x - self.dt*x**2 + 3*sin(u)
           
    def f_x(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        return 1 - 2*self.dt*x
    
    def f_u(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. u '''
        return 3*cos(u)

    
    
if __name__=='__main__':
    np.set_printoptions(precision=3, suppress=True);
    
    ''' Test DDP with a simple 1d nonlinear system:
            x_{t+1} = x_t - dt*x_t^2 + sin(u_t)
    '''
        
    SYSTEM_ID = 2
    N = 10;                 # horizon size
    dt = 0.1;               # control time step
    mu =1e-4;               # initial regularization
    ddp_params = {}
    ddp_params['alpha_factor'] = 0.5
    ddp_params['mu_factor'] = 10.
    ddp_params['mu_max'] = 1e0
    ddp_params['min_alpha_to_increase_mu'] = 0.1
    ddp_params['min_cost_impr'] = 1e-2
    ddp_params['max_line_search_iter'] = 10
    ddp_params['exp_improvement_threshold'] = 1e-6
    ddp_params['max_iter'] = 20
    DEBUG = False;
    
    n = 1;                          # state size
    m = 1;                          # control size
    U_bar = np.zeros((N,m));        # initial guess for control inputs
    x0 = np.array([0.0]);           # initial state
    x_tasks = np.array([-5.0]);     # goal state
    N_task = N;                     # time step to reach goal state
    
    ''' TASK FUNCTION  '''
    lmbda = 1e-4;           # control regularization
    H_xx = np.zeros((N+1, n, n));
    h_x  = np.zeros((N+1, n));
    h_s  = np.zeros(N+1);
    H_xx[N_task,:,:]  = np.identity(n);
    h_x[N_task,:]     = -x_tasks;
    h_s[N_task]       = 0.5*np.dot(x_tasks.T, x_tasks);   
    
    if(SYSTEM_ID==1):
        solver = DDPSolverLinearDyn("LinDyn", ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG)
    elif(SYSTEM_ID==2):
        solver = DDPSolverSinDyn("SinDyn", ddp_params, H_xx, h_x, h_s, lmbda, dt, DEBUG)
    
    (X,U,K) = solver.solve(x0, U_bar, mu);
    solver.print_statistics(x0, U, K, X);
