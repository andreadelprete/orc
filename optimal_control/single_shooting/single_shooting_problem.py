# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: Andrea Del Prete (andrea.delprete@unitn.it)
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
#from colorama import Fore, Back, Style
from termcolor import colored, cprint

from orc.optimal_control.numerical_integration import Integrator

class Empty:
    def __init__(self):
        pass

class SingleShootingProblem:
    
    def __init__(self, name, ode, x0, dt, N, integration_scheme, simu):
        self.name = name
        self.ode = ode
        self.integrator = Integrator('integrator')
        self.x0 = x0
        self.dt = dt
        self.N = N
        self.integration_scheme = integration_scheme
        self.simu = simu
        
        self.nq = int(x0.shape[0]/2)
        self.nx = x0.shape[0]
        self.nu = self.ode.nu
        self.X = np.zeros((N+1, self.x0.shape[0]))
        self.U = np.zeros((N, self.nu))
        
        self.last_X_displayed = np.copy(self.X)
        self.last_values = Empty()
        self.last_values.cost = 0.0
        self.last_values.cost_last_display = 0.0
        self.last_values.running_cost = 0.0
        self.last_values.final_cost = 0.0
        
        self.history = Empty()
    
        self.running_costs = []
        self.final_costs = []
        self.path_ineqs = []
        self.final_ineqs = []
        self.final_eqs = []
        
        self.ineq_print_threshold = 0.05e10
        
    def add_running_cost(self, c, weight=1):
        self.running_costs += [(weight,c)]
        self.last_values.__dict__[c.name] = 0.0
        self.last_values.__dict__[c.name+'_grad'] = 0.0
    
    def add_final_cost(self, c, weight=1):
        self.final_costs += [(weight,c)]
        self.last_values.__dict__[c.name] = 0.0
        self.last_values.__dict__[c.name+'_grad'] = 0.0
        
    def add_path_ineq(self, c):
        self.path_ineqs += [c]
        self.last_values.__dict__[c.name] = 0.0
        self.last_values.__dict__[c.name+'_grad'] = 0.0
    
    def add_final_ineq(self, c):
        self.final_ineqs += [c]
        self.last_values.__dict__[c.name] = 0.0
        self.last_values.__dict__[c.name+'_grad'] = 0.0
        
    def add_final_eq(self, c):
        self.final_eqs += [c]
        self.last_values.__dict__[c.name] = 0.0
        self.last_values.__dict__[c.name+'_grad'] = 0.0
        
    '''*************************************************'''
    '''                 COST FUNCTIONS                  '''
    '''*************************************************'''
    
    def running_cost(self, X, U):
        ''' Compute the running cost integral '''
        cost = 0.0
        t = 0.0
        
        # reset the variables storing the costs
        for (w,c) in self.running_costs:
            self.last_values.__dict__[c.name] = 0.0
            
        for i in range(U.shape[0]):
            for (w,c) in self.running_costs:
                tmp = w * self.dt * c.compute(X[i,:], U[i,:], t, recompute=True)
                cost += tmp
                self.last_values.__dict__[c.name] += tmp
            t += self.dt
        return cost
        
    def running_cost_w_gradient(self, X, U, dXdU):
        ''' Compute the running cost integral and its gradient w.r.t. U'''
        cost = 0.0
        grad = np.zeros(self.N*self.nu)
        t = 0.0
        nx, nu = self.nx, self.nu
        
        # reset the variables storing the costs
        for (w,c) in self.running_costs:
            self.last_values.__dict__[c.name] = 0.0
            self.last_values.__dict__[c.name+'_grad'] = 0.0
            
        for i in range(U.shape[0]):
            for (w,c) in self.running_costs:
                ci, ci_x, ci_u = c.compute_w_gradient(X[i,:], U[i,:], t, recompute=True)
                dci = ci_x.dot(dXdU[i*nx:(i+1)*nx,:]) 
                dci[i*nu:(i+1)*nu] += ci_u
                
                cost += w * self.dt * ci
                grad += w * self.dt * dci
                self.last_values.__dict__[c.name] += w * self.dt * ci
                self.last_values.__dict__[c.name+'_grad'] += norm(w * self.dt * dci)
            t += self.dt
        return (cost, grad)
        
    
    def final_cost(self, x_N):
        ''' Compute the final cost '''
        cost = 0.0
        for (w,c) in self.final_costs:
            tmp = w * c.compute(x_N, recompute=True)
            cost += tmp
            self.last_values.__dict__[c.name] = tmp
        return cost
        
    
    def final_cost_w_gradient(self, x_N, dxN_dU):
        ''' Compute the final cost and its gradient w.r.t. U'''
        cost = 0.0
        grad = np.zeros(self.N*self.nu)
        for (w,c) in self.final_costs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU)
            cost += w * ci
            grad += w * dci
            self.last_values.__dict__[c.name] = w * ci
            self.last_values.__dict__[c.name+'_grad'] = norm(w * dci)
        return (cost, grad)
        
    
    def compute_cost(self, y):
        ''' Compute cost function '''
        if(np.any(np.isnan(y))):
            print(colored("\t[Compute cost] The solution vector (U) contains NaN!", "red"))
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0, ndt = 0.0, 1
        X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                      self.N, self.integration_scheme)
        if(np.any(np.isnan(X))):
            print(colored("\t[Compute cost] The state vector (X) contains NaN!", "red"))
        
        # compute cost
        run_cost = self.running_cost(X, U)
        fin_cost = self.final_cost(X[-1,:])
        cost = run_cost + fin_cost
        
        # store X, U and cost
        self.X, self.U = X, U
        self.last_values.cost = cost
        self.last_values.running_cost = run_cost
        self.last_values.final_cost = fin_cost
        return cost
        
    
    def compute_cost_w_gradient_fd(self, y):
        ''' Compute both the cost function and its gradient using finite differences '''
        if(np.any(np.isnan(y))):
            print(colored("\t[Compute cost] The solution vector (U) contains NaN!", "red"))
        eps = 1e-8
        y_eps = np.copy(y)
        grad = np.zeros_like(y)
        cost = self.compute_cost(y)
        for i in range(y.shape[0]):
            y_eps[i] += eps
            cost_eps = self.compute_cost(y_eps)
            y_eps[i] = y[i]
            grad[i] = (cost_eps - cost) / eps
        self.last_values.cost = cost
        self.last_values.grad = norm(grad)
        return (cost, grad)
        
    
    def compute_cost_w_gradient(self, y):
        ''' Compute cost function and its gradient '''
        if(np.any(np.isnan(y))):
            print(colored("\t[Compute cost] The solution vector (U) contains NaN!", "red"))
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0 = 0.0
        X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                        self.dt, self.N, 
                                                        self.integration_scheme)
        if(np.any(np.isnan(X))):
            print(colored("\t[Compute cost] The state vector (X) contains NaN!", "red"))
            print("Max u value:", np.max(np.abs(y)))
            for i in range(X.shape[0]):
                print("Time step", i, "||x||=", np.linalg.norm(X[i,:]))
                if(np.any(np.isnan(X[i,:]))): break
            import sys
            sys.exit()
        if(np.any(np.isnan(dXdU))):
            print(colored("\t[Compute cost] The sensitivities (dXdU) contain NaN!", "red"))
        
        # compute cost
        (run_cost, grad_run) = self.running_cost_w_gradient(X, U, dXdU)
        (fin_cost, grad_fin) = self.final_cost_w_gradient(X[-1,:], dXdU[-self.nx:,:])
        cost = run_cost + fin_cost
        grad = grad_run + grad_fin
        
        # store X, U and cost
        self.X, self.U, self.dXdU = X, U, dXdU
        self.last_values.cost = cost  
        self.last_values.running_cost = run_cost
        self.last_values.final_cost = fin_cost
        self.last_values.grad = norm(grad)
        return (cost, grad)
    
    '''*************************************************'''
    '''                 INEQUALITIES                    '''
    '''*************************************************'''
    
    def path_ineq(self, X, U):
        ''' Compute the path inequalities '''
        ineq = []
        t = 0.0
        for c in self.path_ineqs:
            self.last_values.__dict__[c.name] = []
            
        for i in range(U.shape[0]):
            for c in self.path_ineqs:
                tmp = c.compute(X[i,:], U[i,:], t, recompute=True).tolist()
                ineq.extend(tmp)
                self.last_values.__dict__[c.name].extend(tmp)
            t += self.dt
        return ineq
                
    
    def final_ineq(self, x_N):
        ''' Compute the final inequalities '''
        ineq = []
        for c in self.final_ineqs:
            tmp = c.compute(x_N, recompute=True).tolist()
            ineq.extend(tmp)
            self.last_values.__dict__[c.name] = tmp
        return ineq
    
    
    def compute_ineq(self, y):
        ''' Compute all the the inequality constraints '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        if(norm(self.U-U)!=0.0):
            t0, ndt = 0.0, 1
            X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                          self.N, self.integration_scheme)
        else:
            X = self.X
        # compute inequalities
        run_ineq = self.path_ineq(X, U)
        fin_ineq = self.final_ineq(X[-1,:])
        ineq = np.array(run_ineq + fin_ineq) # concatenation
        
        # store X, U and ineq
        self.X, self.U = X, U
        self.last_values.ineq = ineq
        return ineq
    
    
    def path_ineq_w_jac(self, X, U, dXdU):
        ''' Compute the path inequalities '''
        ineq = []
        nx, nu = self.nx, self.nu
        jac = np.zeros((self.N*1000, self.N*nu)) # assume there are <1000 ineq
        index = 0
        t = 0.0
        for c in self.path_ineqs:
            self.last_values.__dict__[c.name] = []
            
        for i in range(U.shape[0]):
            for c in self.path_ineqs:
                ci, ci_x, ci_u = c.compute_w_gradient(X[i,:], U[i,:], t, recompute=True)
                dci = ci_x.dot(dXdU[i*nx:(i+1)*nx,:]) 
                dci[:, i*nu:(i+1)*nu] += ci_u
                
                ineq.extend(ci.tolist())
                jac[index:index+dci.shape[0],:] = dci
                index += dci.shape[0]
                
                self.last_values.__dict__[c.name].extend(ci.tolist())
            t += self.dt
            
        jac = jac[:index,:]
        return (ineq, jac)
    
            
    def final_ineq_w_jac(self, x_N, dxN_dU):
        ''' Compute the final inequalities '''
        ineq = []
        jac = np.zeros((1000, self.N*self.nu)) # assume there are <1000 ineq
        index = 0
        for c in self.final_ineqs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU) 
            
            ineq.extend(ci.tolist())
            jac[index:index+ci.shape[0],:] = dci
            index += dci.shape[0]
            
            self.last_values.__dict__[c.name] = ci.tolist()
            
        jac = jac[:index,:]
        return (ineq, jac)
    
    
    def compute_ineq_jac(self, y):
        ''' Compute all the the inequality constraints '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        if(norm(self.U-U)!=0.0):
            t0 = 0.0
            X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                        self.dt, self.N, 
                                                        self.integration_scheme)
        else:
            X, dXdU = self.X, self.dXdU
            
        # compute inequalities
        (run_ineq, jac_run) = self.path_ineq_w_jac(X, U, dXdU)
        (fin_ineq, jac_fin) = self.final_ineq_w_jac(X[-1,:], dXdU[-self.nx:,:])
        ineq = np.array(run_ineq + fin_ineq) # concatenation
        jac = np.vstack((jac_run, jac_fin))
        
        # store X, U and ineq
        self.X, self.U = X, U
        self.last_values.ineq = ineq
        return jac
    
    
    '''*************************************************'''
    '''                   EQUALITIES                    '''
    '''*************************************************'''
    
    def final_eq(self, x_N):
        ''' Compute the final equalities '''
        eq = []
        for c in self.final_eqs:
            tmp = c.compute(x_N, recompute=True).tolist()
            eq.extend(tmp)
            self.last_values.__dict__[c.name] = tmp
        return eq
    
    
    def compute_eq(self, y):
        ''' Compute all the the equality constraints '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        if(norm(self.U-U)!=0.0):
            t0, ndt = 0.0, 1
            X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                          self.N, self.integration_scheme)
        else:
            X = self.X
        # compute inequalities
#        run_ineq = self.path_ineq(X, U)
        fin_eq = self.final_eq(X[-1,:])
        eq = fin_eq #np.array(run_ineq + fin_ineq) # concatenation
        
        # store X, U and ineq
        self.X, self.U = X, U
        self.last_values.eq = eq
        return eq
    
            
    def final_eq_w_jac(self, x_N, dxN_dU):
        ''' Compute the final equalities '''
        eq = []
        jac = np.zeros((1000, self.N*self.nu)) # assume there are <1000 ineq
        index = 0
        for c in self.final_eqs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU) 
            
            eq.extend(ci.tolist())
            jac[index:index+ci.shape[0],:] = dci
            index += dci.shape[0]
            
            self.last_values.__dict__[c.name] = ci.tolist()
            
        jac = jac[:index,:]
        return (eq, jac)
    
    
    def compute_eq_jac(self, y):
        ''' Compute all the the equality constraints '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        if(norm(self.U-U)!=0.0):
            t0 = 0.0
            X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                        self.dt, self.N, 
                                                        self.integration_scheme)
        else:
            X, dXdU = self.X, self.dXdU
            
        # compute inequalities
#        (run_ineq, jac_run) = self.path_eq_w_jac(X, U, dXdU)
        (fin_eq, jac_fin) = self.final_eq_w_jac(X[-1,:], dXdU[-self.nx:,:])
#        ineq = np.array(run_ineq + fin_ineq) # concatenation
#        jac = np.vstack((jac_run, jac_fin))
        
        # store X, U and ineq
        self.X, self.U = X, U
        self.last_values.ineq = fin_eq
        return jac_fin
        
        
    def solve(self, y0=None, method='SLSQP', use_finite_diff=False, max_iter=300):
        ''' Solve the optimal control problem '''
        self.history.cost = []
        self.history.grad = []
        
        # if no initial guess is given => initialize with zeros
        if(y0 is None):
            y0 = np.zeros(self.N*self.nu)
        
        self.iter = 0
        print('Start optimizing')
        if(use_finite_diff):
            cost_func = self.compute_cost_w_gradient_fd
            r = minimize(cost_func, y0, jac=True, method=method, 
                     callback=self.clbk, tol=1e-6, options={'maxiter': max_iter, 'disp': True})
        else:
            cost_func = self.compute_cost_w_gradient
            r = minimize(cost_func, y0, jac=True, method=method, 
                         callback=self.clbk, tol=1e-6, options={'maxiter': max_iter, 'disp': True},
                        constraints=[
                                     {'type': 'ineq', 
                                      'fun': self.compute_ineq,
                                      'jac': self.compute_ineq_jac},
                                      {'type': 'eq', 
                                      'fun': self.compute_eq,
                                      'jac': self.compute_eq_jac}
                                      ])
        return r
        
    
    def sanity_check_cost_gradient(self, N_TESTS=10):
        ''' Compare the gradient computed with finite differences with the one
            computed by deriving the integrator
        '''
        for i in range(N_TESTS):
            y = np.random.rand(self.N*self.nu)
            (cost, grad_fd) = self.compute_cost_w_gradient_fd(y)
            (cost, grad) = self.compute_cost_w_gradient(y)
            grad_err = np.zeros_like(grad)
            for i in range(grad_err.shape[0]):
                grad_err[i] = np.abs(grad[i]-grad_fd[i])
                if(np.abs(grad_fd[i]) > 1.0): # normalize
                    grad_err[i] = np.abs(grad[i]-grad_fd[i])/grad_fd[i]
                    
            if(np.max(grad_err)>1e-2):
                print('Errors in gradient computations:', np.max(grad_err))
                print('Grad err:\n', grad-grad_fd)
                print('Grad FD:\n', grad_fd)
            else:
                print('Everything is fine', np.max(np.abs(grad_err)))
#                print('Grad FD:\n', 1e3*grad_fd)
#                print('Grad   :\n', 1e3*grad)
        
        
    def clbk(self, xk):
        self.iter += 1
        self.history.cost.append(self.last_values.cost)
        self.history.grad.append(self.last_values.grad)
        
        print('Iter %3d, cost %7.3f, grad %7.3f'%(self.iter, self.last_values.cost, self.last_values.grad))
        if(np.any(np.isnan(xk))):
            print(colored("\tThe solution vector (U) contains NaN!", "red"))
            return True
    
        for (w,c) in self.running_costs:
            print(colored("\t Running cost %60s: %7.3f %7.3f"%(c.name, 
                                                               self.last_values.__dict__[c.name], 
                                                               self.last_values.__dict__[c.name+'_grad']),
                          "grey"))
        for (w,c) in self.final_costs:
            print(colored("\t Final cost   %60s: %7.3f %7.3f"%(c.name, 
                                                               self.last_values.__dict__[c.name], 
                                                               self.last_values.__dict__[c.name+'_grad']),
                            "grey"))
        for c in self.path_ineqs:
            v = np.min(self.last_values.__dict__[c.name])
            if(v<=0.0): color = 'red'
            else:       color = 'blue'
            if(v<=self.ineq_print_threshold):
                print(colored('\t Path ineq    %60s: %7.3f'%(c.name, v), color))
        for c in self.final_ineqs:
            v = np.min(self.last_values.__dict__[c.name])
            if(v<=0.0): color = 'red'
            else:       color = 'blue'
            if(v<=self.ineq_print_threshold):
                print(colored('\t Final ineq   %60s: %7.3f'%(c.name, v), color))
        for c in self.final_eqs:
            print(colored('\t Final eq     %60s: %7.3f'%(c.name, 
                                                         norm(self.last_values.__dict__[c.name])),
                            "grey"))
    
        # Display motion every time the trajectory has changed significantly
        # and every 50-th iteration
        if(self.iter%50==0 or np.max(np.abs(self.last_X_displayed-self.X))>0.5):
            print("Display motion...")
#            self.last_values.cost_last_display = self.last_values.cost
            self.last_X_displayed = np.copy(self.X)
            self.simu.display_motion(self.X[:,:self.nq], self.dt)
        return False
