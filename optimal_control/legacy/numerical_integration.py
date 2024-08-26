# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Test different integration schemes and their derivatives.

@author: Andrea Del Prete (andrea.delprete@unitn.it)
"""

import numpy as np
import orc.optimal_control.solutions.numerical_integration_sol as sol

class Integrator:
    ''' A class implementing different numerical integrator schemes '''
    def __init__(self, name):
        self.name = name
        
    def integrate(self, ode, x_init, U, t_init, dt, ndt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory:
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - ndt: the number of inner time steps for each time step
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        n = x_init.shape[0]
        t = np.zeros((N*ndt+1))*np.nan
        x = np.zeros((N*ndt+1,n))*np.nan
        dx = np.zeros((N*ndt,n))*np.nan
        h = dt/ndt  # inner time step
        x[0,:] = x_init
        t[0] = t_init
        
        for i in range(x.shape[0]-1):
            ii = int(np.floor(i/ndt))
            t[i+1] = t[i] + h

            if(scheme=='RK-1'):            
                x[i+1,:], dx[i,:] = sol.rk1(x[i,:], h, U[ii,:], t[i], ode)    
            elif(scheme=='RK-2'):   # explicit midpoint method
                x[i+1,:], dx[i,:] = sol.rk2(x[i,:], h, U[ii,:], t[i], ode)
            elif(scheme=='RK-2-Heun'):
                x[i+1,:], dx[i,:] = sol.rk2heun(x[i,:], h, U[ii,:], t[i], ode)
            elif(scheme=='RK-3'): # Kutta's third-order method
                x[i+1,:], dx[i,:] = sol.rk3(x[i,:], h, U[ii,:], t[i], ode)
            elif(scheme=='RK-4'):
                x[i+1,:], dx[i,:] = sol.rk4(x[i,:], h, U[ii,:], t[i], ode)
            elif(scheme=='ImpEul'):
                x[i+1,:], dx[i,:] = sol.implicit_euler(x[i,:], h, U[ii,:], t[i], ode)
            elif(scheme=='SemiImpEul'):
                x[i+1,:], dx[i,:] = sol.semi_implicit_euler(x[i,:], h, U[ii,:], t[i], ode)
            
        self.dx = dx
        self.t = t
        self.x = x        
        return x[::ndt,:]
        
        
    def integrate_w_sensitivities_u(self, ode, x_init, U, t_init, dt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory.
            Compute also the derivative of the x trajectory w.r.t. U.
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        nx = x_init.shape[0]
        nu = ode.nu
        t = np.zeros((N+1))*np.nan
        x = np.zeros((N+1,nx))*np.nan
        dx = np.zeros((N+1,nx))*np.nan
        dXdU = np.zeros(((N+1)*nx,N*nu))
        h = dt
        x[0,:] = x_init
        t[0] = t_init
        
        for i in range(N):
            if(scheme=='RK-1'):    
                x[i+1,:], dx[i,:], phi_x, phi_u = sol.rk1(x[i,:], h, U[i,:], t[i], ode, True)
            elif(scheme=='RK-4'):
                x[i+1,:], dx[i,:], phi_x, phi_u = sol.rk4(x[i,:], h, U[i,:], t[i], ode, True)
            else:
                return None
            t[i+1] = t[i] + h
            ix, ix1, ix2 = i*nx, (i+1)*nx, (i+2)*nx
            iu, iu1 = i*nu, (i+1)*nu
            dXdU[ix1:ix2,:] = phi_x.dot(dXdU[ix:ix1,:]) 
            dXdU[ix1:ix2,iu:iu1] += phi_u

        self.dx = dx
        self.t = t
        self.x = x        
        return (x, dXdU)
        
    def check_sensitivities_u(self, ode, x_init, t_init, dt, N, scheme, N_TESTS=10):
        eps = 1e-8
        nx = x_init.shape[0]
        nu = ode.nu
        for iii in range(N_TESTS):
            U = np.random.rand(N, nu)
            (X, dXdU) = self.integrate_w_sensitivities_u(ode, x_init, U, t_init, dt, N, scheme)
            X = X.reshape(X.shape[0]*X.shape[1])
            dXdU_fd = np.zeros(((N+1)*nx,N*nu))
            for i in range(N):
                for j in range(nu):
                    U_bar = np.copy(U)
                    U_bar[i,j] += eps
                    X_bar = self.integrate(ode, x_init, U_bar, t_init, dt, 1, N, scheme)
                    X_bar = X_bar.reshape(X_bar.shape[0]*X_bar.shape[1])
                    dXdU_fd[:, i*nu+j] = (X_bar-X)/eps
            dXdU_err = dXdU - dXdU_fd
            
            print("Error in sensitivities", np.max(np.abs(dXdU_err)))
            if(np.max(np.abs(dXdU_err))>np.sqrt(eps)):
                print("dXdU", dXdU)
                print("dXdU_fd", dXdU_fd)
                

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import orc.utils.plot_utils as plut
    from ode import ODERobot, ODELinear, ODESin, ODEStiffDiehl
    from orc.utils.robot_loaders import loadUR
    from orc.utils.robot_wrapper import RobotWrapper
    np.set_printoptions(precision=3, suppress=True);
    
    ''' Test numerical integration with a manipulator
    '''
    LINE_WIDTH = 60
    q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ])  # initial configuration
    dt = 0.1                     # time step
    T = 0.5                      # time horizon
    N = int(T/dt);               # horizon steps
    PLOT_STUFF = 1
    linestyles = [' *', ' o', ' v', 's']

    system = 'ur'
#    system = 'linear'
#    system = 'sin'
#    system = 'stiff-diehl'
    
    print("".center(LINE_WIDTH,'#'))
    print(" Numerical integration ".center(LINE_WIDTH, '#'))
    print("".center(LINE_WIDTH,'#'), '\n')
    
    # choose the number of inner steps so that the number of function evaluations
    # is the same for every method
    integrators = []
    integrators += [{'scheme': 'RK-4',      'ndt': 1000}]    # used as ground truth
    integrators += [{'scheme': 'RK-1',      'ndt': 12}]
    integrators += [{'scheme': 'RK-2',      'ndt': 6}]
    integrators += [{'scheme': 'RK-3',      'ndt': 4}]
    integrators += [{'scheme': 'RK-4',      'ndt': 3}]
    
        
    if(system=='ur'):
        r = loadUR()
        robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
        x0 = np.concatenate((q0, np.zeros(robot.nv)))  # initial state
        ode = ODERobot('ode', robot)
    elif(system=='linear'):
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        b = np.array([0.0])
        x0 = np.array([1.0])
        ode = ODELinear('linear', A, B, b)
    elif(system=='linear2'):
        A = np.array([[-10.0, 1.0],
                      [0.0, -100.0]])
        b = np.array([0.0, 0.0])
        x0 = np.array([10.0, 7.0])
        ode = ODELinear('linear2', A, b)
    elif(system=='sin'):
        x0 = np.array([0.0])
        ode = ODESin('sin', A=np.array([1.0]), f=np.array([20.0]), phi=np.array([0.0]))
    elif(system=='stiff-diehl'):
        x0 = np.array([0.0])
        ode = ODEStiffDiehl()
    U = np.zeros((N, ode.nu));
    
    x_coarse = {}
    x_fine = {}
    t_fine = {}
    dx = {}
    integrator = Integrator('integrator')
    
    for params in integrators:        
        scheme = params['scheme']
        name = scheme+'_ndt_'+str(params['ndt'])
        print("Integrate with ", scheme, 'ndt=', params['ndt'])
        t = 0.0
#        integrator.check_sensitivities_u(ode, x0, t, dt, N, scheme, N_TESTS=1)
        x_coarse[name] = integrator.integrate(ode, x0, U, t, dt, params['ndt'], N, scheme)
        x_fine[name] = np.copy(integrator.x)
        t_fine[name] = np.copy(integrator.t)
        dx[name] = np.copy(integrator.dx)
        
            
    # PLOT STUFF
    time = np.arange(0.0, N*dt, dt)
    
    if(PLOT_STUFF):
        max_plots = 6
        if(x0.shape[0]==1):
            nplot = 1
            (f, ax) = plut.create_empty_figure()
            ax = [ax]
        else:
            nplot = int(min(max_plots, x0.shape[0])/2)
            (f, ax) = plut.create_empty_figure(nplot,2)
            ax = ax.reshape(nplot*2)
        i_ls = 0
        for name, x in x_fine.items():
            for i in range(len(ax)):
                ls = linestyles[i_ls]
                ax[i].plot(t_fine[name], x[:,i], ls, label=name, alpha=0.7)
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel(r'$x_'+str(i)+'$')
            i_ls = (i_ls+1)%len(linestyles)
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
    
    print("Simulation finished")
    plt.show()