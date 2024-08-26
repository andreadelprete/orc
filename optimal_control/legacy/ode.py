# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Classes representing different kind of Ordinary Differential Equations (ODEs).

@author: student
"""

import numpy as np
import pinocchio as pin

class ODE:
    def __init__(self, name):
        self.name = name
        self.nu = 0
        
    def f(self, x, u, t):
        return np.zeros(x.shape)
        
        
class ODESin(ODE):
    ''' ODE defining a sinusoidal trajectory '''
    def __init__(self, name, A, f, phi):
        ODE.__init__(self, name) 
        self.A = A
        self.two_pi_f = 2*np.pi*f
        self.phi = phi
        
    def f(self, x, u, t):
        return self.two_pi_f*self.A*np.cos(self.two_pi_f*t + self.phi)
       
       
class ODELinear(ODE):
    ''' A linear ODE: dx = A*x + B*u + b
    '''
    def __init__(self, name, A, B, b):
        ODE.__init__(self, name) 
        self.A = A
        self.B = B
        self.b = b
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
    def f(self, x, u, t, jacobian=False):
        dx = self.A.dot(x) + self.b + self.B.dot(u)
        if(jacobian):
            return (np.copy(dx), np.copy(self.A), np.copy(self.B))
        return np.copy(dx)
        
        
class ODEStiffDiehl(ODE):
    def __init__(self, name=''):
        ODE.__init__(self, name) 
        
    def f(self, x, u, t, jacobian=False):
        dx = -50.0*(x - np.cos(t))
        if(not jacobian):
            return dx
        Fx = -50
        Fu = 0
        return (dx, Fx, Fu)
            
        
        
class ODEPendulum(ODE):
    def __init__(self, name=''):
        ODE.__init__(self, name) 
        self.g = -9.81
        
    def f(self, x, u, t):
        dx = np.zeros(2)
        dx[0] = x[1]
        dx[1] = self.g*np.sin(x[0])
        return dx
        
        
class ODERobot(ODE):
    ''' An ordinary differential equation representing a robotic system
    '''
    
    def __init__(self, name, robot):
        ''' robot: instance of RobotWrapper
        '''
        ODE.__init__(self, name) 
        self.robot = robot
        self.nu = robot.na
        nq, nv = self.robot.nq, self.robot.nv
        self.nx = nq+nv
        self.nu = self.robot.na
        self.Fx = np.zeros((self.nx, self.nx))
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fu = np.zeros((self.nx, self.nu))
        self.dx = np.zeros(2*nv)            
        
        
    ''' System dynamics '''
    def f(self, x, u, t, jacobian=False):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        
        if(nv==1):
            # for 1 DoF systems pin.aba does not work (I don't know why)
            pin.computeAllTerms(model, data, q, v)
            ddq = (u-data.nle) / data.M[0]
        else:
            ddq = self.robot.aba(q, v, u) #pin.aba(model, data, q, v, u) - self.B @ v

        self.dx[:nv] = v
        self.dx[nv:] = ddq
        
        if(jacobian):
            ddq_dq, ddq_dv, ddq_du = self.robot.abaDerivatives(q, v, u)
#            pin.computeABADerivatives(model, data, q, v, u)
            self.Fx[:nv, :nv] = 0.0
            self.Fx[:nv, nv:] = np.identity(nv)
            self.Fx[nv:, :nv] = ddq_dq
            self.Fx[nv:, nv:] = ddq_dv
            self.Fu[nv:, :] = ddq_du
            
            return (np.copy(self.dx), np.copy(self.Fx), np.copy(self.Fu))
        
        return np.copy(self.dx)
        
        
    def f_x_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. x computed via finite differences '''
        f0 = self.f(x, u, t)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u, t)
            Fx[:,i] = (fp-f0)/delta
        return Fx
        
        
    def f_u_fin_diff(self, x, u, t, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed via finite differences '''
        f0 = self.f(x, u, t)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up, t)
            Fu[:,i] = (fp-f0)/delta
        return Fu


if __name__=='__main__':
    from arc.utils.robot_loaders import loadUR, loadPendulum
    from example_robot_data.robots_loader import loadDoublePendulum
    from arc.utils.robot_wrapper import RobotWrapper
    import single_shooting_conf as conf
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
        
    dt = conf.dt                 # time step
    system=conf.system
    N_TESTS = 10
    
    if(system=='ur'):
        r = loadUR()
    elif(system=='double-pendulum'):
        r = loadDoublePendulum()
    elif(system=='pendulum'):
        r = loadPendulum()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv    
    n = nq+nv                       # state size
    m = robot.na                    # control size
    ode = ODERobot('ode', robot)
    t = 0.0
    
    for i in range(N_TESTS):
        x = np.random.rand(n)
        u = np.random.rand(m)
        (dx, Fx, Fu) = ode.f(x, u, t, jacobian=True)
        Fx_fd = ode.f_x_fin_diff(x, u, t)
        Fu_fd = ode.f_u_fin_diff(x, u, t)

        Fx_err = Fx-Fx_fd
        Fu_err = Fu-Fu_fd
        if(np.max(np.abs(Fx_err))>1e-4):
            print('Fx:   ', Fx)
            print('Fx FD:', Fx_fd)
        else:
            print('Fx is fine', np.max(np.abs(Fx_err)))
            
        if(np.max(np.abs(Fu_err))>1e-4):
            print('Fu:   ', Fu)
            print('Fu FD:', Fu_fd)
        else:
            print('Fu is fine', np.max(np.abs(Fu_err)))