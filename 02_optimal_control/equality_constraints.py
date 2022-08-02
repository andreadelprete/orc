# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
import pinocchio as pin


class Empty:
    def __init__(self):
        pass
        
        
class OCPFinalConstraintState:
    ''' Equality constraint for reaching a desired state of the robot
    '''
    def __init__(self, name, robot, q_des, v_des):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        self.nv = robot.model.nv
        self.q_des = q_des   # desired joint angles
        self.v_des = v_des  # desired joint velocities
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        eq = np.concatenate((e, de))
        return eq
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        eq = np.concatenate((e, de))
        jac =  np.eye(self.nv*2)
        return (eq, jac)
        
