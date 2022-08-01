# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.colors as mcolors
import warnings

from ode import ODERobot
from numerical_integration import Integrator

class Empty:
    def __init__(self):
        pass
        
def compute_cost_w_gradient_fd(cost, x):
        ''' Compute both the cost function and its gradient using finite differences '''
        eps = 1e-8
        x_eps = np.copy(x)
        grad = np.zeros_like(x)
        c = cost.compute(x)
        for i in range(x.shape[0]):
            x_eps[i] += eps
            c_eps = cost.compute(x_eps)
            x_eps[i] = x[i]
            grad[i] = (c_eps - c) / eps
        return (cost, grad)
        
def check_final_cost_gradient(cost, N_TESTS=10):
    ''' Compare the gradient computed with finite differences with the one
        computed by deriving the integrator
    '''
    for i in range(N_TESTS):
        x = np.random.rand(2*cost.nq)
        (c, grad_fd) = compute_cost_w_gradient_fd(cost, x)
        (c, grad) = cost.compute_w_gradient(x)
        grad_err = np.zeros_like(grad)
        for i in range(grad_err.shape[0]):
            grad_err[i] = np.abs(grad[i]-grad_fd[i])
            if(np.abs(grad_fd[i]) > 1.0): # normalize
                grad_err[i] = np.abs(grad[i]-grad_fd[i])/grad_fd[i]
                
        if(np.max(grad_err)>1e-2):
            print('Errors in gradient computations:', np.max(grad_err))
            print('Grad err:\n', grad-grad_fd)
            print('Grad FD:\n', 1e3*grad_fd)
            print('Grad   :\n', 1e3*grad)
        else:
            print('Everything is fine', np.max(np.abs(grad_err)))
            print('Grad FD:\n', 1e3*grad_fd)
            print('Grad   :\n', 1e3*grad)


if __name__=='__main__':
    from arc.utils.robot_loaders import loadUR, loadPendulum
    from example_robot_data.robots_loader import loadDoublePendulum
    from arc.utils.robot_wrapper import RobotWrapper
    from arc.utils.robot_simulator import RobotSimulator
    import single_shooting_conf as conf
    from cost_functions import OCPFinalCostState, OCPFinalCostFramePos, OCPRunningCostQuadraticControl, OCPFinalCostFrame
    import pinocchio as pin
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
        
    dt = conf.dt                 # time step
    T = conf.T
    N = int(T/dt);         # horizon size
    PLOT_STUFF = 1
    linestyles = ['-*', '--*', ':*', '-.*']
    system=conf.system
    
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
    U = np.zeros((N,m))           # initial guess for control inputs
    ode = ODERobot('ode', robot)
    
    # create simulator 
    simu = RobotSimulator(conf, robot)
    nq = robot.model.nq      
    
    # create cost function terms
#    final_cost = OCPFinalCostFramePos(robot, conf.frame_name, conf.p_des, conf.dp_des, conf.weight_vel)
    final_cost = OCPFinalCostFrame(robot, conf.frame_name, conf.p_des, conf.dp_des, conf.R_des, conf.w_des, conf.weight_vel)
#    final_cost_state = OCPFinalCostState(robot, conf.q_des, np.zeros(nq), conf.weight_vel)
    
    check_final_cost_gradient(final_cost)
    
