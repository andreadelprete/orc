# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
import pinocchio as pin
        


class OCPPathPlaneCollisionAvoidance:
    ''' Path inequality constraint for collision avoidance with a frame of the robot
        (typically the end-effector). The constraint is defined as:
            n.T * x >= b
        where x is the 3d position of the specified frame, while n and b are user-defined values
    '''
    def __init__(self, name, robot, frame_name, n, b):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        self.frame_id = robot.model.getFrameId(frame_name)
        assert(robot.model.existFrame(frame_name))
        self.n = n   # normal direction of the plane to avoid collision with
        self.b = b   # bias
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        ineq = self.n.dot(p) - self.b
        return ineq
    
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        ineq = self.n.dot(p) - self.b
        
        # compute Jacobian J
        self.robot.computeAllTerms(q, v)
        J6 = self.robot.frameJacobian(q, self.frame_id)
        J = J6[:3,:]            # take first 3 rows of J6
            
        grad_x = np.zeros_like(x)
        grad_u = np.zeros_like(u)
        grad_x[:self.nq] =  self.n.dot(J)

        return (ineq, grad_x, grad_u)
    