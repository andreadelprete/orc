# -*- coding: utf-8 -*-
"""
Generic DDP solver class.
System dynamics and cost functions must be specified in child classes.

@author: adelpret
"""

import numpy as np
import orc.optimal_control.solutions.ddp_sol as sol

class DDPSolver:
    
    def __init__(self, name, params, DEBUG=False):
        self.name = name
        self.alpha_factor = params['alpha_factor']
        self.mu_factor = params['mu_factor']
        self.mu_max = params['mu_max']
        self.min_alpha_to_increase_mu = params['min_alpha_to_increase_mu']
        self.min_cost_impr = params['min_cost_impr']
        self.max_line_search_iter = params['max_line_search_iter']
        self.exp_improvement_threshold = params['exp_improvement_threshold']
        self.max_iter = params['max_iter']
        self.DEBUG = DEBUG
        
    ''' Simulate system forward with computed control law '''
    def simulate_system(self, x0, U_bar, K, X_bar):
        n = x0.shape[0]
        m = U_bar.shape[1]
        N = U_bar.shape[0]
        X = np.zeros((N+1, n))
        U = np.zeros((N, m))
        X[0,:] = x0
        for i in range(N):
            U[i,:] = U_bar[i,:] + K[i,:,:] @ (X[i,:]-X_bar[i,:])
            X[i+1,:] = self.f(X[i,:], U[i,:])
        return (X,U)
        
        
    def backward_pass(self, X_bar, U_bar, mu):
        return sol.backward_pass(self, X_bar, U_bar, mu)

        
    def update_expected_cost_improvement(self):
        self.d1 = 0.0
        self.d2 = 0.0
        
        for i in range(self.N):
            self.d1 += self.w[i,:].T @ self.Q_u[i,:]
            self.d2 += self.w[i,:].T @ self.Q_uu[i,:,:] @ self.w[i,:]
            
                        
    ''' Differential Dynamic Programming
        The pseudoinverses in the algorithm are regularized by the damping factor mu.
    '''
    def solve(self, x0, U_bar, mu):                    
        # each control law is composed by a feedforward w and a feedback K
        self.N = N = U_bar.shape[0]     # horizon length
        m = U_bar.shape[1]              # size of u
        n = x0.shape[0]                 # size of x
        self.w  = np.zeros((N,m))      # feedforward control inputs
        self.K  = np.zeros((N,m,n))    # feedback gains
                
        X_bar = np.zeros((N,n))    # nominal state trajectory
        
        # derivatives of the cost function w.r.t. x and u
        self.l_x = np.zeros((N+1, n))
        self.l_xx = np.zeros((N+1, n, n))
        self.l_u = np.zeros((N, m))
        self.l_uu = np.zeros((N, m, m))
        self.l_xu = np.zeros((N, n, m))
        
        # the cost-to-go is defined by a quadratic function: 0.5 x' Q_{xx,i} x + Q_{x,i} x + ...
        self.Q_xx = np.zeros((N, n, n))
        self.Q_x  = np.zeros((N, n))
        self.Q_uu = np.zeros((N, m, m))
        self.Q_u  = np.zeros((N, m))
        self.Q_xu = np.zeros((N, n, m))
        
        converged = False
        for j in range(self.max_iter):
            print("\n*** Iter %d" % j)
            
            # compute nominal state trajectory X_bar
            (X_bar, U_bar) = self.simulate_system(x0, U_bar, self.K, X_bar)
            
            self.backward_pass(X_bar, U_bar, mu)
            
            # forward pass - line search
            alpha = 1
            line_search_succeeded = False
            # compute costs for nominal trajectory and expected improvement model
            cst = self.cost(X_bar, U_bar)
            self.update_expected_cost_improvement()
            
            for jj in range(self.max_line_search_iter):
                (X,U) = self.simulate_system(x0, U_bar + alpha*self.w, self.K, X_bar)
                new_cost = self.cost(X, U);
                exp_impr = alpha*self.d1 + 0.5*(alpha**2)*self.d2
#                print("Expected improvement", exp_impr, "Real improvement", new_cost-cst)
                if(exp_impr>0.0):
                    exp_impr = -1.0
                relative_impr = (new_cost-cst)/exp_impr
                
                if(relative_impr > self.min_cost_impr):
                    print("Cost improved from %.3f to %.3f. Exp. impr %.3f. Rel. impr. %.1f%%" % (cst, new_cost, exp_impr, 1e2*relative_impr))
                    line_search_succeeded = True
                    U_bar += alpha*self.w
                    cst = new_cost
                    break
                alpha = self.alpha_factor*alpha
            
            if(not line_search_succeeded):
                mu = mu*self.mu_factor
                print("No cost improvement, increasing mu to", mu)
                if(mu>self.mu_max):
                    print("Max regularization reached. Algorithm failed to converge.")
                    converged = True
            else:
                print("Line search succeeded with alpha", alpha)
                if(alpha>=self.min_alpha_to_increase_mu):
                    mu = mu/self.mu_factor
                    print("Decreasing mu to ", mu)
                else:
                    mu = mu*self.mu_factor
                    print("Alpha is small => increasing mu to", mu)
                    if(mu>self.mu_max):
                        print("Max regularization reached. Algorithm failed to converge.")
                        converged = True
                self.callback(X_bar, U_bar)
                    
            if(abs(exp_impr) < self.exp_improvement_threshold):
                print("Algorithm converged. Expected improvement", exp_impr)
                converged = True
                
            if(converged):
                break
                    
        # compute nominal state trajectory X_bar
        (X_bar, U_bar) = self.simulate_system(x0, U_bar, self.K, X_bar)
        return (X_bar, U_bar, self.K)
        
        
    def callback(self, X, U):
        ''' callback function called at every iteration '''
        pass
    
    def print_statistics(self, x0, U_bar, K, X_bar):
        # simulate system forward with computed control law
        (X, U) = self.simulate_system( x0, U_bar, K, X_bar)
        print("\n**************************************** RESULTS ****************************************")
        
        # compute cost of each task
        cost = self.cost(X, U)
        print("Cost  ", cost)
        print("Effort", np.linalg.norm(U))
        
    ''' Discrete-time system dynamics '''
    def f(x, u):
        return None
           
    ''' Partial derivatives of discrete-time system dynamics w.r.t. x '''
    def f_x(x, u):
        return None
    
    ''' Partial derivatives of discrete-time system dynamics w.r.t. u '''       
    def f_u(x, u):
        return None
        
    def cost(self, X, U):
        ''' total cost (running+final) for state trajectory X and control trajectory U '''
        return None
        
    def cost_running(self, i, x, u):
        ''' Running cost at time step i for state x and control u '''
        return None
        
    def cost_final(self, x):
        ''' Final cost for state x '''
        return None
        
    def cost_running_x(self, i, x, u):
        ''' Gradient of the running cost w.r.t. x '''
        return None
        
    def cost_final_x(self, x):
        ''' Gradient of the final cost w.r.t. x '''
        return None
    
    def cost_final_xx(self, x):
        ''' Hessian of the final cost w.r.t. x '''
        return None
        
    def cost_running_u(self, i, x, u):
        ''' Gradient of the running cost w.r.t. u '''
        return None
        
    def cost_running_xx(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x '''
        return None
        
    def cost_running_uu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. u '''
        return None
        
    def cost_running_xu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x and then w.r.t. u '''
        return None
