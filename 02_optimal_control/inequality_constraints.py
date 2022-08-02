# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
import pinocchio as pin
        

class OCPFinalPlaneCollisionAvoidance:
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
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the state x '''
        q = x[:self.nq]
        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        ineq = self.n.dot(p) - self.b
        return np.array([ineq])
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        ineq = self.n.dot(p) - self.b
        
        # compute Jacobian J
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        J6 = self.robot.frameJacobian(q, self.frame_id)
        J = J6[:3,:]            # take first 3 rows of J6
            
        grad_x = np.zeros((1,x.shape[0]))
        grad_x[0, :self.nq] =  self.n.dot(J)

        return (np.array([ineq]), grad_x)
    
    
class OCPFinalSelfCollisionAvoidance:
    ''' Path inequality constraint for self-collision avoidance between two frames 
        of the robot. The constraint is defined as:
            ||x1 - x2||^2 >= min_dist
        where x1 and x2 are the 3d position of the specified frames, 
        while min_dist is a user-defined value.
    '''
    def __init__(self, name, robot, frame1_name, frame2_name, min_dist):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        self.frame1_id = robot.model.getFrameId(frame1_name)
        self.frame2_id = robot.model.getFrameId(frame2_name)
        assert(robot.model.existFrame(frame1_name))
        assert(robot.model.existFrame(frame2_name))
        self.min_dist = min_dist
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the state x '''
        q = x[:self.nq]
        H1 = self.robot.framePlacement(q, self.frame1_id, recompute)
        H2 = self.robot.framePlacement(q, self.frame2_id, recompute)
        d = H1.translation - H2.translation # take the 3d position of the end-effector
        ineq = d.dot(d) - self.min_dist**2
        return np.array([ineq])
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        H1 = self.robot.framePlacement(q, self.frame1_id, recompute)
        H2 = self.robot.framePlacement(q, self.frame2_id, recompute)
        d = H1.translation - H2.translation # take the 3d position of the end-effector
        ineq = d.dot(d) - self.min_dist**2
        
        # compute Jacobian J
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        J1 = self.robot.frameJacobian(q, self.frame1_id)[:3,:]
        J2 = self.robot.frameJacobian(q, self.frame2_id)[:3,:]
            
        grad_x = np.zeros((1,x.shape[0]))
        grad_x[0, :self.nq] = 2*d.dot(J1-J2)

        return (np.array([ineq]), grad_x)
    
    
class OCPFinalJointBounds:
    ''' Final inequality constraint for joint bounds. The constraint is defined as:
            q >= q_min
            -q >= q_max
            dq >= dq_min
            -dq >= dq_max
    '''
    def __init__(self, name, robot, q_min, q_max, dq_min, dq_max):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        self.q_min = q_min
        self.q_max = q_max
        self.dq_min = dq_min
        self.dq_max = dq_max
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        ineq = np.concatenate((q-self.q_min, self.q_max-q, 
                               v-self.dq_min, self.dq_max-v))
        return ineq
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        ineq = np.concatenate((q-self.q_min, self.q_max-q, 
                               v-self.dq_min, self.dq_max-v))
        
        # compute Jacobian
        nx = x.shape[0]
        grad_x = np.zeros((2*nx, nx))
        nq = self.nq
        grad_x[:nq,       :nq] = np.eye(nq)
        grad_x[nq:2*nq,   :nq] = -np.eye(nq)
        grad_x[2*nq:3*nq, nq:] = np.eye(nq)
        grad_x[3*nq:,     nq:] = -np.eye(nq)

        return (ineq, grad_x)


class OCPPathPlaneCollisionAvoidance:
    '''' Path inequality constraint for collision avoidance with a frame of the robot
        (typically the end-effector). The constraint is defined as:
            n.T * x >= b
        where x is the 3d position of the specified frame, while n and b are user-defined values
    '''
    def __init__(self, name, robot, frame_name, n, b):
        self.c = OCPFinalPlaneCollisionAvoidance(name, robot, frame_name, n, b)
        self.name = name
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost given the state x '''
        return self.c.compute(x, recompute)
    
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        (ineq, grad_x) = self.c.compute_w_gradient(x, recompute)
        grad_u = np.zeros((ineq.shape[0], u.shape[0]))
        return (ineq, grad_x, grad_u)
    

class OCPPathSelfCollisionAvoidance:
    ''' Path inequality constraint for self-collision avoidance between two frames 
        of the robot. The constraint is defined as:
            ||x1 - x2||^2 >= min_dist
        where x1 and x2 are the 3d position of the specified frames, 
        while min_dist is a user-defined value.
    '''
    def __init__(self, name, robot, frame1_name, frame2_name, min_dist):
        self.c = OCPFinalSelfCollisionAvoidance(name, robot, frame1_name, 
                                                 frame2_name, min_dist)
        self.name = name
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost given the state x '''
        return self.c.compute(x, recompute)
    
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        (ineq, grad_x) = self.c.compute_w_gradient(x, recompute)
        grad_u = np.zeros((ineq.shape[0], u.shape[0]))
        return (ineq, grad_x, grad_u)
    
    
class OCPPathJointBounds:
    ''' Path inequality constraint for joint bounds. The constraint is defined as:
            q >= q_min
            -q >= q_max
    '''
    def __init__(self, name, robot, q_min, q_max, dq_min, dq_max):
        self.c = OCPFinalJointBounds(name, robot, q_min, q_max, dq_min, dq_max)
        self.name = name
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost given the state x '''
        return self.c.compute(x, recompute)
    
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        (ineq, grad_x) = self.c.compute_w_gradient(x, recompute)
        grad_u = np.zeros((ineq.shape[0], u.shape[0]))
        return (ineq, grad_x, grad_u)
    