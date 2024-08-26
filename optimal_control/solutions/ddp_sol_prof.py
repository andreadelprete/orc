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
    
def backward_pass(solver, X_bar, U_bar, mu, use_second_derivative=1):
        n = X_bar.shape[1]
        m = U_bar.shape[1]
        N = U_bar.shape[0]
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
        solver.l_x[-1,:]  = solver.cost_final_x(X_bar[-1,:])
        solver.l_xx[-1,:,:] = solver.cost_final_xx(X_bar[-1,:])
        V_xx[N,:,:] = solver.l_xx[N,:,:]
        V_x[N,:]    = solver.l_x[N,:]
        
        for i in range(N-1, -1, -1):
            if(solver.DEBUG):
                print("\n *** Time step %d ***" % i)
                
            # compute dynamics Jacobians
            A[i,:,:] = solver.f_x(X_bar[i,:], U_bar[i,:])
            B[i,:,:] = solver.f_u(X_bar[i,:], U_bar[i,:])
                
            # compute the gradient of the cost function at X=X_bar
            solver.l_x[i,:]    = solver.cost_running_x(i, X_bar[i,:], U_bar[i,:])
            solver.l_xx[i,:,:] = solver.cost_running_xx(i, X_bar[i,:], U_bar[i,:])
            solver.l_u[i,:]    = solver.cost_running_u(i, X_bar[i,:], U_bar[i,:])
            solver.l_uu[i,:,:] = solver.cost_running_uu(i, X_bar[i,:], U_bar[i,:])
            solver.l_xu[i,:,:] = solver.cost_running_xu(i, X_bar[i,:], U_bar[i,:])
            
            # compute regularized cost-to-go
            solver.Q_x[i,:]     = solver.l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            solver.Q_u[i,:]     = solver.l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            solver.Q_xx[i,:,:]  = solver.l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            solver.Q_uu[i,:,:]  = solver.l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            solver.Q_xu[i,:,:]  = solver.l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]

            if(use_second_derivative):
                Fxx = solver.f_xx(X_bar[i,:], U_bar[i,:])
                Fuu = solver.f_uu(X_bar[i,:], U_bar[i,:])
                solver.Q_xx[i,:,:] += Fxx * V_x[i+1,:]
                solver.Q_uu[i,:,:] += Fuu * V_x[i+1,:]
            
            if(solver.DEBUG):
                print("Q_x, Q_u, Q_xx, Q_uu, Q_xu", a2s(solver.Q_x[i,rx]), a2s(solver.Q_u[i,ru]), 
                        a2s(solver.Q_xx[i,rx,:]), a2s(solver.Q_uu[i,ru,:]), a2s(solver.Q_xu[i,rx,0]))
                
            Qbar_uu       = solver.Q_uu[i,:,:] + mu*np.identity(m)
            Qbar_uu_pinv  = np.linalg.pinv(Qbar_uu)
            solver.w[i,:]       = - Qbar_uu_pinv @ solver.Q_u[i,:]
            solver.K[i,:,:]     = - Qbar_uu_pinv @ solver.Q_xu[i,:,:].T
            if(solver.DEBUG):
                print("Qbar_uu, Qbar_uu_pinv",a2s(Qbar_uu), a2s(Qbar_uu_pinv))
                print("w, K", a2s(solver.w[i,ru]), a2s(solver.K[i,ru,rx]))
                
            # update Value function
            V_x[i,:]    = (solver.Q_x[i,:] + 
                solver.K[i,:,:].T @ solver.Q_u[i,:] +
                solver.K[i,:,:].T @ solver.Q_uu[i,:,:] @ solver.w[i,:] +
                solver.Q_xu[i,:,:] @ solver.w[i,:])
            V_xx[i,:]   = (solver.Q_xx[i,:,:] + 
                solver.K[i,:,:].T @ solver.Q_uu[i,:,:] @ solver.K[i,:,:] + 
                solver.Q_xu[i,:,:] @ solver.K[i,:,:] + 
                solver.K[i,:,:].T @ solver.Q_xu[i,:,:].T)
                
            # V_x[i,:]    = solver.Q_x[i,:]  - solver.Q_xu[i,:,:] @ Qbar_uu_pinv @ solver.Q_u[i,:]
            # V_xx[i,:]   = solver.Q_xx[i,:] - solver.Q_xu[i,:,:] @ Qbar_uu_pinv @ solver.Q_xu[i,:,:].T
                    
        return (solver.w, solver.K)


def backward_pass_box(solver, X_bar, U_bar, mu, u_min, u_max):
        ''' Backward pass of box DDP, which is DDP with control bounds. '''
        n = X_bar.shape[1]
        m = U_bar.shape[1]
        N = U_bar.shape[0]
  
        # the Value function is defined by a quadratic function: 0.5 x' V_{xx,i} x + V_{x,i} x
        V_xx = np.zeros((N+1, n, n))
        V_x  = np.zeros((N+1, n))
        
        # dynamics derivatives w.r.t. x and u
        A = np.zeros((N, n, n))
        B = np.zeros((N, n, m))
        
        # initialize value function
        solver.l_x[-1,:]  = solver.cost_final_x(X_bar[-1,:])
        solver.l_xx[-1,:,:] = solver.cost_final_xx(X_bar[-1,:])
        V_xx[N,:,:] = solver.l_xx[N,:,:]
        V_x[N,:]    = solver.l_x[N,:]
        
        for i in range(N-1, -1, -1):
            if(solver.DEBUG):
                print("\n *** Time step %d ***" % i)
                
            # compute dynamics Jacobians
            A[i,:,:] = solver.f_x(X_bar[i,:], U_bar[i,:])
            B[i,:,:] = solver.f_u(X_bar[i,:], U_bar[i,:])
                
            # compute the gradient of the cost function at X=X_bar
            solver.l_x[i,:]    = solver.cost_running_x(i, X_bar[i,:], U_bar[i,:])
            solver.l_xx[i,:,:] = solver.cost_running_xx(i, X_bar[i,:], U_bar[i,:])
            solver.l_u[i,:]    = solver.cost_running_u(i, X_bar[i,:], U_bar[i,:])
            solver.l_uu[i,:,:] = solver.cost_running_uu(i, X_bar[i,:], U_bar[i,:])
            solver.l_xu[i,:,:] = solver.cost_running_xu(i, X_bar[i,:], U_bar[i,:])
            
            # compute regularized cost-to-go
            solver.Q_x[i,:]     = solver.l_x[i,:] + A[i,:,:].T @ V_x[i+1,:]
            solver.Q_u[i,:]     = solver.l_u[i,:] + B[i,:,:].T @ V_x[i+1,:]
            solver.Q_xx[i,:,:]  = solver.l_xx[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ A[i,:,:]
            solver.Q_uu[i,:,:]  = solver.l_uu[i,:,:] + B[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            solver.Q_xu[i,:,:]  = solver.l_xu[i,:,:] + A[i,:,:].T @ V_xx[i+1,:,:] @ B[i,:,:]
            
            # account for directions in which u is saturated
            i_bool = np.logical_and(U_bar[i,:]<u_max, U_bar[i,:]>u_min)
            m_free = int(np.sum(i_bool)) # number of free control inputs
            print("Max |u| = ", np.max(np.abs(U_bar[i])))
            print("Number of saturated control inputs: ", m-m_free)
            i_free = np.where(i_bool)[0] # indeces of the free control inputs
            Q_uu = solver.Q_uu[i,:,:]
            # create a reduced Q_uu considering only the non-saturated (free) control inputs
            Q_uu_free       = Q_uu[i_bool,:][:,i_bool] + mu*np.identity(m_free)
            Q_uu_free_inv   = np.linalg.pinv(Q_uu_free)
            # map Q_uu_free_inv to Q_uu_inv
            Q_uu_inv = np.zeros((m,m))
            for j in range(m_free):
                for k in range(m_free):
                    Q_uu_inv[i_free[j], i_free[k]] = Q_uu_free_inv[j,k]
            solver.w[i,:]       = - Q_uu_inv @ solver.Q_u[i,:]
            solver.K[i,:,:]     = - Q_uu_inv @ solver.Q_xu[i,:,:].T
                
            # update Value function
            V_x[i,:]    = (solver.Q_x[i,:] + 
                solver.K[i,:,:].T @ solver.Q_u[i,:] +
                solver.K[i,:,:].T @ solver.Q_uu[i,:,:] @ solver.w[i,:] +
                solver.Q_xu[i,:,:] @ solver.w[i,:])
            V_xx[i,:]   = (solver.Q_xx[i,:,:] + 
                solver.K[i,:,:].T @ solver.Q_uu[i,:,:] @ solver.K[i,:,:] + 
                solver.Q_xu[i,:,:] @ solver.K[i,:,:] + 
                solver.K[i,:,:].T @ solver.Q_xu[i,:,:].T)
                    
        return (solver.w, solver.K)