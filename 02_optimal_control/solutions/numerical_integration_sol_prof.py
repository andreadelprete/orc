#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:53:57 2021

@author: adelprete
"""
import numpy as np
from numpy.linalg import norm, solve


def rk1(x, h, u, t, ode, jacobian=False):
    if(jacobian==False):
        dx = ode.f(x, u, t)
        x_next = x + h*dx
        return x_next, dx

    (f, f_x, f_u) = ode.f(x, u, t, jacobian=True)
    dx = f
    x_next = x + h*f
    
    nx = x.shape[0]
    I = np.identity(nx)    
    phi_x = I + h*f_x
    phi_u = h * f_u
    return x_next, dx, phi_x, phi_u

def rk2(x, h, u, t, ode):
    k1 = ode.f(x,            u, t)
    k2 = ode.f(x + 0.5*h*k1, u, t+0.5*h)
    dx = k2
    x_next = x + h*k2
    return x_next, dx

def rk2heun(x, h, u, t, ode):
    k1 = ode.f(x,        u, t)
    k2 = ode.f(x + h*k1, u, t+h)
    dx = 0.5*(k1+k2)
    x_next = x + h*dx
    return x_next, dx

def rk3(x, h, u, t, ode):
#    k1 = ode.f(x,            u, t)
#    k2 = ode.f(x + h/3*k1,   u, t+h/3)
#    k3 = ode.f(x + 2/3*h*k2, u, t+2*h/3)
#    dx = (k1 + k2 + k3)/3.0
    k1 = ode.f(x,                  u, t)
    k2 = ode.f(x + h*0.5*k1,       u, t+0.5*h)
    k3 = ode.f(x + h*(-k1 + 2*k2), u, t+h)
    dx = (k1 + 4*k2 + k3)/6.0
    x_next = x + h*dx
    return x_next, dx

def rk4(x, h, u, t, ode, jacobian=False):
    if(not jacobian):
        k1 = ode.f(x,            u, t)
        k2 = ode.f(x + 0.5*h*k1, u, t+0.5*h)
        k3 = ode.f(x + 0.5*h*k2, u, t+0.5*h)
        k4 = ode.f(x + h * k3,   u, t+h)
        dx = (k1 + 2*k2 + 2*k3 + k4)/6.0
        x_next = x + h*dx
        return x_next, dx
    nx = x.shape[0]
    I = np.identity(nx)    
    
    (k1, f1_x, f1_u) = ode.f(x, u, t, jacobian=True)
    k1_x = f1_x
    k1_u = f1_u
    
    x2 = x + 0.5*h*k1
    t2 = t+0.5*h
    (k2, f2_x, f2_u) = ode.f(x2, u, t2, jacobian=True)
    k2_x = f2_x.dot(I + 0.5*h*k1_x)
    k2_u = f2_u + 0.5*h*f2_x @ k1_u
    
    x3 = x + 0.5*h*k2
    t3 = t+0.5*h
    (k3, f3_x, f3_u) = ode.f(x3, u, t3, jacobian=True)
    k3_x = f3_x.dot(I + 0.5*h*k2_x)
    k3_u = f3_u + 0.5*h*f3_x @ k2_u
    
    x4 = x + h * k3
    t4 = t+h
    (k4, f4_x, f4_u) = ode.f(x4, u, t4, jacobian=True)
    k4_x = f4_x.dot(I + h*k3_x)
    k4_u = f4_u + h*f4_x @ k3_u
    
    dx = (k1 + 2*k2 + 2*k3 + k4)/6.0
    x_next = x + h*dx
    
    phi_x = I + h*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
    phi_u =     h*(k1_u + 2*k2_u + 2*k3_u + k4_u)/6.0
    return x_next, dx, phi_x, phi_u

def semi_implicit_euler(x, h, u, t, ode):
    nv = int(x.shape[0]/2)
    f = ode.f(x, u, t)
    v_next = x[-nv:] + h*f[-nv:]
    dx = f
    dx[:nv] = v_next
    x_next = x + h*f
    return x_next, dx

def implicit_euler(x, h, u, t, ode):
    z = x + h*ode.f(x, u, t)
    # Solve the following system of equations for z:
    #   g(z) = z - x - h*f(z) = 0
    # Start by computing the Newton step:
    #   g(z) = g(z_i) + G(z_i)*dz = 0 => dz = -G(z_i)^-1 * g(z_i)
    # where G is the Jacobian of g and z_i is our current guess of z
    #   G(z) = I - h*F(z)
    # where F(z) is the Jacobian of f wrt z.
    I = np.identity(x.shape[0])
    converged = False
    for j in range(100):
        (f, Fx, Fu) = ode.f(z, u, t, jacobian=True)
        g = z - x - h*f
#                    print(j, "|g|", norm(g))
        if(norm(g)<1e-8):
            converged = True
#                        print(j, "|g|<thr", norm(g))
            break
        G = I - h*Fx
        dz = solve(G, -g)
        z += dz

    if(not converged):
        print("Implicit Euler did not converge!!!! |g|=", norm(g))
        
    dx = f
    x_next = x + h*dx
    return x_next, dx