#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:58:30 2021

@author: adelprete
"""
import numpy as np
from numpy.linalg import norm, inv

def inverse_geometry_step(q, x, x_des, J, regu, i, N, robot, frame_id, conf):
    e = x_des - x
    cost = norm(e)
    
    # gradient descent
#    q[:,i+1] = q[:,i] + alpha*J.T.dot(e)
    
    # Newton method
    nv = J.shape[1]
    B = J.T.dot(J) + regu*np.eye(nv) # approximate regularized Hessian
    gradient = J.T.dot(e)   # gradient
    delta_q = inv(B).dot(gradient)
    q_next = q + delta_q
    
    # if gradient is null you are done
    grad_norm = norm(gradient)
    if(grad_norm<conf.gradient_threshold):
        print("Terminate because gradient is (almost) zero:", grad_norm)
        print("Problem solved after %d iterations with error %f"%(i, norm(e)))
        return None
    
#    if(norm(e)<conf.absolute_threshold):
#        print("Problem solved after %d iterations with error %f"%(i, norm(e)))
#        break
    
    if(not conf.line_search):
        q_next = q + delta_q
    else:
        # back-tracking line search
        alpha = 1.0
        iter_line_search = 0
        while True:
            q_next = q + alpha*delta_q
            robot.computeJointJacobians(q_next)
            robot.framesForwardKinematics(q_next)
            x_new = robot.framePlacement(q_next, frame_id).translation
            cost_new = norm(x_des - x_new)
            if cost_new < (1.0-alpha*conf.gamma)*cost:
    #            print("Backtracking line search converged with log(alpha)=%.1f"%np.log10(alpha))
                break
            else:
                alpha *= conf.beta
                iter_line_search += 1
                if(iter_line_search==N):
                    print("Backtracking line search could not converge. log(alpha)=%.1f"%np.log10(alpha))
                    break
                
    if i%conf.PRINT_N == 0:
        print("Iteration %d, ||x_des-x||=%f, norm(gradient)=%f"%(i, norm(e), grad_norm))
                
    return q_next, cost, grad_norm