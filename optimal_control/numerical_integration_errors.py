# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Test different integration schemes and their derivatives.

@author: Andrea Del Prete (andrea.delprete@unitn.it)
"""

import numpy as np
    
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import orc.utils.plot_utils as plut
    from orc.utils.robot_loaders import loadUR
    from example_robot_data.robots_loader import load
    from orc.utils.robot_wrapper import RobotWrapper
    from numerical_integration import Integrator
    from ode import ODELinear, ODESin, ODERobot, ODEStiffDiehl, ODEPendulum
    import numerical_integration_conf as conf
    np.set_printoptions(precision=3, suppress=True);
    
    ''' Compute errors of different integration schemes
    '''
    print("".center(conf.LINE_WIDTH,'#'))
    print(" Numerical integration errors ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH,'#'), '\n')
    
    dt = 0.1                   # time step
    N = int(conf.T/dt);         # horizon size
    ndt_ground_truth = 1000     # number of inner time steps used for computing the ground truth
    PLOT_STUFF = 1
    linestyles = ['-*', '--*', ':*', '-.*']
    # choose which system you want to integrate
    system = 'ur'
#    system='double-pendulum'
#    system='pendulum-ode'
#    system = 'linear'
#    system = 'sin'
#    system = 'stiff-diehl'
    
    integrators = []
    integrators += [{'scheme': 'RK-1'      , 'nf': 1}]
    integrators += [{'scheme': 'RK-2'      , 'nf': 2}] # nf = number of function evaluation per step
    integrators += [{'scheme': 'RK-3'      , 'nf': 3}]
    integrators += [{'scheme': 'RK-4'      , 'nf': 4}]
    
    
    
    ndt_list = np.array([int(i) for i in 2**np.arange(1.5,8,0.5)])
    print('Testing system', system)
        
    if(system=='ur' or system=='double-pendulum'):
        if(system=='ur'):
            r = loadUR()
        elif(system=='double-pendulum'):
            r = load('double_pendulum')
        robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
        if(system=='ur'):
            x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))  # initial state
        else:
            x0 = np.zeros(robot.nq+robot.nv)
        ode = ODERobot('ode', robot)
    elif(system=='pendulum-ode'):
        x0 = np.array([np.pi/2, 0.0])
        ode = ODEPendulum()
    elif(system=='linear'):
        A = np.array([[-100.0]])
        B = np.zeros((1,0))
        b = np.array([0.0])
        x0 = np.array([100.0])        
        ode = ODELinear('linear', A, B, b)
    elif(system=='linear2'):
        A = np.array([[-10.0, 1.0],
                      [0.0, -100.0]])
        B = np.zeros((2,0))
        b = np.array([0.0, 0.0])
        x0 = np.array([10.0, 7.0])
        ode = ODELinear('linear2', A, B, b)
    elif(system=='sin'):
        x0 = np.array([0.0])
        ode = ODESin('sin', A=np.array([1.0]), f=np.array([20.0]), phi=np.array([0.0]))
    elif(system=='stiff-diehl'):
        x0 = np.array([0.0])
        ode = ODEStiffDiehl()
    U = np.zeros((N,ode.nu))        # initial guess for control inputs
    
    err_glob = {}
    integrator = Integrator('integrator')
    
    print('Compute ground truth')
    x_gt = integrator.integrate(ode, x0, U, 0.0, dt, ndt_ground_truth, N, 'RK-4')
    
    for params in integrators:
        scheme = params['scheme']
        err_glob[scheme] = np.zeros(len(ndt_list))*np.nan
        for (i,ndt) in enumerate(ndt_list):
            x = integrator.integrate(ode, x0, U, 0.0, dt, ndt, N, scheme)
            err = np.linalg.norm(x - x_gt)
            err_glob[scheme][i] = err
            print("Integration %10s"%scheme, 'ndt=%4d'%ndt, 'log(err)', np.log10(err))
            
    # PLOT STUFF    
    if(PLOT_STUFF):
        (f, ax) = plut.create_empty_figure()
        i_ls = 0
        for params in integrators:
            scheme = params['scheme']
            ls = linestyles[i_ls]
            ax.plot(ndt_list, err_glob[scheme], ls, label=scheme, alpha=0.7)
            i_ls = (i_ls+1)%len(linestyles)
        ax.set_xlabel('ndt')
        ax.set_ylabel('Global error')         
        ax.set_xscale('log')
        ax.set_yscale('log')
        leg = ax.legend()
        leg.get_frame().set_alpha(0.5)
        
        
        (f, ax) = plut.create_empty_figure()
        i_ls = 0
        for params in integrators:
            scheme = params['scheme']
            nf = params['nf']
            ls = linestyles[i_ls]
            ax.plot(nf*ndt_list, err_glob[scheme], ls, label=scheme, alpha=0.7)
            i_ls = (i_ls+1)%len(linestyles)
        ax.set_xlabel('# Function calls')
        ax.set_ylabel('Global error')         
        ax.set_xscale('log')
        ax.set_yscale('log')
        leg = ax.legend()
        leg.get_frame().set_alpha(0.5)
    
    print("Script finished")
    plt.show()