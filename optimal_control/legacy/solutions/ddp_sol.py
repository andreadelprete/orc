#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 07:54:41 2021

@author: adelprete
"""
import numpy as np

def a2s(a, format_string ='{0:.2f} '):
    ''' array to string '''
    if(len(a.shape)==0):
        return format_string.format(a);

    if(len(a.shape)==1):
        res = '[';
        for i in range(a.shape[0]):
            res += format_string.format(a[i]);
        return res+']';
        
    res = '[[';
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            res += format_string.format(a[i,j]);
        res = res[:-1]+'] [';
    return res[:-2]+']'; 
    #[format_string.format(v,i) for i,v in enumerate(a)]
    
def backward_pass(solver, X_bar, U_bar, mu):
        n = X_bar.shape[1]      # size of x
        m = U_bar.shape[1]      # size of u
        N = U_bar.shape[0]      # size of the horizon
        rx = list(range(0,n))
        ru = list(range(0,m))
        
        # the task is defined by a quadratic cost: 
        # sum_{i=0}^N 0.5 x' l_{xx,i} x + l_{x,i} x +  0.5 u' l_{uu,i} u + l_{u,i} u + x' l_{xu,i} u
        
        # the Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
        V_xx = np.zeros((N+1, n, n))
        V_x  = np.zeros((N+1, n))
        
        # dynamics derivatives w.r.t. x and u
        A = np.zeros((N, n, n))
        B = np.zeros((N, n, m))
        
        # initialize value function
        solver.l_x[-1,:]  = ...
        solver.l_xx[-1,:,:] = ...
        V_xx[N,:,:] = ...
        V_x[N,:]    = ...
        
        for i in range(N-1, -1, -1):
            if(solver.DEBUG):
                print("\n *** Time step %d ***" % i)
                
            # compute dynamics Jacobians
            A[i,:,:] = ...
            B[i,:,:] = ...
                
            # compute the gradient of the cost function at X=X_bar
            solver.l_x[i,:]    = ...
            solver.l_xx[i,:,:] = ...
            solver.l_u[i,:]    = ...
            solver.l_uu[i,:,:] = ...
            solver.l_xu[i,:,:] = ...
            
            # compute regularized cost-to-go
            solver.Q_x[i,:]     = ...
            solver.Q_u[i,:]     = ...
            solver.Q_xx[i,:,:]  = ...
            solver.Q_uu[i,:,:]  = ...
            solver.Q_xu[i,:,:]  = ...
            
            if(solver.DEBUG):
                print("Q_x, Q_u, Q_xx, Q_uu, Q_xu", a2s(solver.Q_x[i,rx]), a2s(solver.Q_u[i,ru]), 
                        a2s(solver.Q_xx[i,rx,:]), a2s(solver.Q_uu[i,ru,:]), a2s(solver.Q_xu[i,rx,0]))
                
            Qbar_uu       = ...
            Qbar_uu_pinv  = ...
            solver.w[i,:]       = ...
            solver.K[i,:,:]     = ...
            
            if(solver.DEBUG):
                print("Qbar_uu, Qbar_uu_pinv",a2s(Qbar_uu), a2s(Qbar_uu_pinv));
                print("w, K", a2s(solver.w[i,ru]), a2s(solver.K[i,ru,rx]));
                
            # update Value function
            V_x[i,:]    = ...
            V_xx[i,:]   = ...
                    
        return (solver.w, solver.K)