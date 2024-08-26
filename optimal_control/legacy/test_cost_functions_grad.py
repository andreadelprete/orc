# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm

def compute_cost_w_gradient_x_fd(cost, x):
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
    
    
def compute_cost_w_gradient_x_u_fd(cost, x, u):
        ''' Compute both the cost function and its gradient using finite differences '''
        eps = 1e-8
        x_eps = np.copy(x)
        u_eps = np.copy(u)
        grad_x = np.zeros_like(x)
        grad_u = np.zeros_like(u)
        c = cost.compute(x, u, 0.0)
        for i in range(x.shape[0]):
            x_eps[i] += eps
            c_eps = cost.compute(x_eps, u, 0.0)
            x_eps[i] = x[i]
            grad_x[i] = (c_eps - c) / eps
        for i in range(u.shape[0]):
            u_eps[i] += eps
            c_eps = cost.compute(x, u_eps, 0.0)
            u_eps[i] = u[i]
            grad_u[i] = (c_eps - c) / eps
        return (cost, grad_x, grad_u)
    
    
def compute_w_jacobian_x_fd(f, x):
        ''' Compute both the function and its jacobian using finite differences '''
        eps = 1e-8
        x_eps = np.copy(x)
        c = f.compute(x)
        jac = np.zeros((c.shape[0], x.shape[0]))
        
        for i in range(x.shape[0]):
            x_eps[i] += eps
            c_eps = f.compute(x_eps)
            x_eps[i] = x[i]
            jac[:,i] = (c_eps - c) / eps
        return (c, jac)
        
    
def check_gradient_x(cost, N_TESTS=10):
    ''' Compare the gradient computed with finite differences with the one
        computed by deriving the integrator
    '''
    for i in range(N_TESTS):
        x = np.random.rand(2*cost.robot.nq)
        (c, grad_fd) = compute_cost_w_gradient_x_fd(cost, x)
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
            print('Everything is fine', cost.name, np.max(np.abs(grad_err)))
            

def check_gradient_x_u(cost, N_TESTS=10):
    ''' Compare the gradient computed with finite differences with the one
        computed by deriving the integrator
    '''
    for i in range(N_TESTS):
        x = np.random.rand(2*cost.robot.nq)
        u = np.random.rand(cost.robot.nq)
        (c, grad_x_fd, grad_u_fd) = compute_cost_w_gradient_x_u_fd(cost, x, u)
        (c, grad_x, grad_u)       = cost.compute_w_gradient(x, u, 0.0)
        grad_x_err = np.zeros_like(grad_x)
        grad_u_err = np.zeros_like(grad_u)
        for i in range(grad_x_err.shape[0]):
            grad_x_err[i] = np.abs(grad_x[i]-grad_x_fd[i])
            if(np.abs(grad_x_fd[i]) > 1.0): # normalize
                grad_x_err[i] = np.abs(grad_x[i]-grad_x_fd[i])/grad_x_fd[i]
        for i in range(grad_u_err.shape[0]):
            grad_u_err[i] = np.abs(grad_u[i]-grad_u_fd[i])
            if(np.abs(grad_u_fd[i]) > 1.0): # normalize
                grad_u_err[i] = np.abs(grad_u[i]-grad_u_fd[i])/grad_u_fd[i]
                
        if(np.max(grad_x_err)>1e-2):
            print('Errors in gradient x computations of %s:'%(cost.name), np.max(grad_x_err))
            print('Grad x err:\n', grad_x-grad_x_fd)
            print('Grad x FD:\n', 1e3*grad_x_fd)
            print('Grad x  :\n', 1e3*grad_x)
        else:
            print('Grad x is fine for %s'%(cost.name), np.max(np.abs(grad_x_err)))
        
        if(np.max(grad_u_err)>1e-2):
            print('Errors in gradient u computations of %s:'%(cost.name), np.max(grad_u_err))
            print('Grad u err:\n', grad_u-grad_u_fd)
            print('Grad u FD:\n', 1e3*grad_u_fd)
            print('Grad u  :\n', 1e3*grad_u)
        else:
            print('Grad u is fine for %s'%(cost.name), np.max(np.abs(grad_u_err)))


def check_jacobian_x(cost, N_TESTS=10):
    ''' Compare the Jacobian computed with finite differences with the one
        computed by deriving the integrator
    '''
    for i in range(N_TESTS):
        x = np.random.rand(2*cost.robot.nq)
        (c, jac_fd) = compute_w_jacobian_x_fd(cost, x)
        (c, jac) = cost.compute_w_gradient(x)
        jac_err = np.zeros_like(jac)
        for i in range(jac_err.shape[0]):
            for j in range(jac_err.shape[1]):
                jac_err[i,j] = np.abs(jac[i,j]-jac_fd[i,j])
                if(np.abs(jac_fd[i,j]) > 1.0): # normalize
                    jac_err[i,j] = np.abs(jac[i,j]-jac_fd[i,j])/jac_fd[i,j]
                
        if(np.max(jac_err)>1e-2):
            print('Errors in jacobian x computations:', cost.name, np.max(jac_err))
            print('Jac err:\n', jac-jac_fd)
            print('Jac FD:\n', 1e3*jac_fd)
            print('Jac   :\n', 1e3*jac)
        else:
            print('Jac x is fine', cost.name, np.max(np.abs(jac_err)))
            

if __name__=='__main__':
    from orc.utils.robot_loaders import loadUR, loadPendulum, loadURlab
    from example_robot_data.robots_loader import load
    from orc.utils.robot_wrapper import RobotWrapper
    import single_shooting_conf as conf
    from cost_functions import OCPFinalCostState, OCPFinalCostFramePos, OCPFinalCostFrame
    from cost_functions import OCPRunningCostQuadraticJointVel, OCPRunningCostQuadraticJointAcc, OCPRunningCostQuadraticControl
    from inequality_constraints import OCPFinalPlaneCollisionAvoidance
    from inequality_constraints import OCPFinalJointBounds
    from inequality_constraints import OCPFinalSelfCollisionAvoidance
    from equality_constraints import OCPFinalConstraintState
    import orc.utils.lab_utils as lab

    np.set_printoptions(precision=3, linewidth=200, suppress=True)
        
    system=conf.system
    if(system=='ur'):
        r = loadUR()
    elif(system=='ur-lab'):
        r = loadURlab() 
    elif(system=='double-pendulum'):
        r = load('double_pendulum')
    elif(system=='pendulum'):
        r = loadPendulum()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv    
    
    # create cost function terms
    final_cost = OCPFinalCostFrame('Cost final-frame', robot, conf.frame_name, conf.p_des, conf.dp_des, conf.R_des, conf.w_des, 
                                   conf.weight_final_vel)
    check_gradient_x(final_cost)
    
    final_cost = OCPFinalCostFramePos("Cost final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, 
                                          conf.weight_final_vel)
    check_gradient_x(final_cost)
    
    final_cost_state = OCPFinalCostState("Cost final state", robot, conf.q_des, np.zeros(nq), 
                                             conf.weight_final_q, conf.weight_final_dq)
    check_gradient_x(final_cost_state)
        
    effort_cost = OCPRunningCostQuadraticControl("Cost quadratic joint torques", robot, conf.dt)
    check_gradient_x_u(effort_cost)
        
    dq_cost = OCPRunningCostQuadraticJointVel("Cost quadratic joint vel", robot)
    check_gradient_x_u(dq_cost)
        
    ddq_cost = OCPRunningCostQuadraticJointAcc("cost quadratic joint acc", robot)
    check_gradient_x_u(ddq_cost)
    
    # equality constraints
    final_constr_state = OCPFinalConstraintState("final constraint state", robot, conf.q_des, np.zeros(nq))
    check_jacobian_x(final_constr_state)
    
    # inequality constraints
    q_min = robot.model.lowerPositionLimit
    q_max = robot.model.upperPositionLimit
    dq_max = robot.model.velocityLimit
    dq_min = -dq_max
    joint_bounds_final = OCPFinalJointBounds("joint bounds", robot, q_min, q_max, dq_min, dq_max)
    check_jacobian_x(joint_bounds_final)
    
    # inequalities for avoiding collisions with the table
    for (frame, dist) in conf.table_collision_frames:        
        table_avoidance = OCPFinalPlaneCollisionAvoidance("PlaneCollisionAvoidance", robot, frame, 
                                                         lab.table_normal, lab.table_pos[2]+0.5*lab.table_size[2]+dist)
        check_jacobian_x(table_avoidance)
        break
    
    # inequalities for avoiding self-collisions
    for (frame1, frame2, min_dist) in conf.self_collision_frames:        
        self_coll_avoid = OCPFinalSelfCollisionAvoidance("SelfCollisionAvoidance", robot, 
                                                        frame1, frame2, min_dist)
        check_jacobian_x(self_coll_avoid)
        break
    
