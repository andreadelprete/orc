# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:46:28 2020

@author: student
"""

import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from quadprog import solve_qp
'''
Solve a strictly convex quadratic program

Minimize     1/2 x^T G x - a^T x
Subject to   C.T x >= b

Input Parameters
----------
G : array, shape=(n, n)
a : array, shape=(n,)
C : array, shape=(n, m) matrix defining the constraints
b : array, shape=(m), default=None, vector defining the constraints
meq : int, default=0
    the first meq constraints are treated as equality constraints,
    all further as inequality constraints
factorized : bool, default=False
    If True, then we are passing :math:`R^{−1}` (where :math:`G = R^T R`)
    instead of the matrix G in the argument G.
'''

def generate_random_problem(n):
    ''' Generate a random problem of minimization of kinetic energy
            minimize_u   0.5 * ||x0 + h*u||^2_M
            subject to   -tau_max <= M*u <= tau_max
        The cost function can be expressed in standard form as:
            0.5*h^2*u^T*M*u + h*x0^T*M*u
    '''
    U = rand(n,n)
    M = U.dot(U.T)
    x0 = rand(n)
    tau_max = rand(n)
    
    G = np.copy(M)
    a = -M.dot(x0)
    C = np.zeros((n, 2*n))
    C[:,:n] = M
    C[:,n:] = -M
    b = np.zeros(2*n)
    b[:n] = -tau_max
    b[n:] = -tau_max
    
    solution = solve_qp(G, a, C, b, 0)
    u0 = solution[0]
    x1 = x0 + u0    
    
    G = np.copy(M)
    a = -M.dot(x1)
    C[:,:n] = M
    C[:,n:] = -M
    b[:n] = -tau_max
    b[n:] = -tau_max
    
    solution = solve_qp(G, a, C, b, 0)
    u1 = solution[0]
    
    G = 4*np.copy(M)
    a = -2*M.dot(x0)
    C[:,:n] = M
    C[:,n:] = -M
    b[:n] = -tau_max
    b[n:] = -tau_max
    
    solution = solve_qp(G, a, C, b, 0)
    u = solution[0]
    
    return (u, u0, u1, M, x0, tau_max)
    
n = 3
N_TESTS = 100

print("START TESTS")
for i in range(N_TESTS):
    (u, u0, u1, M, x0, tau_max) = generate_random_problem(n)
    err = norm(2*u - u0-u1)
    if err>1e-8:
        print("Error", err)
        print('2*u  ', 2*u)
        print('u0+u1', u0+u1)
        print('2*u0', 2*u0)
        print('2*u1', 2*u1)
        print("M*u0-tau_max <0", M.dot(u0)-tau_max)
        print("M*u0+tau_max >0", M.dot(u0)+tau_max)
        print("M*u1-tau_max <0", M.dot(u1)-tau_max)
        print("M*u1+tau_max >0", M.dot(u1)+tau_max)
        print("M*u -tau_max <0", M.dot(u) -tau_max)
        print("M*u +tau_max >0", M.dot(u) +tau_max)
        print("")

print("END TESTS")
