# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm


class Empty:
    def __init__(self):
        pass
        
        
class OCPFinalCostState:
    ''' Cost function for reaching a desired state of the robot
    '''
    def __init__(self, robot, q_des, v_des, weight_vel):
        self.robot = robot
        self.nq = robot.model.nq
        self.q_des = q_des   # desired position
        self.v_des = v_des   # desired joint velocities
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        grad =  np.concatenate((e, self.weight_vel*de))
        #print (e.shape, de.shape )
        return (cost, grad)
        
        
class OCPRunningCostQuadraticState:
    ''' Quadratic cost function for penalizing state tracking '''
    def __init__(self, robot, q_des, v_des, weight_vel, dt):
        self.robot = robot
        self.nq = robot.model.nq
        self.q_des = q_des
        self.v_des = v_des
        self.weight_vel = weight_vel
        self.dt = dt
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        i = min(self.q_des.shape[1]-1, int(np.floor(t/self.dt)))
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des[:,i]
        de = v - self.v_des[:,i]
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        i = min(self.q_des.shape[1]-1, int(np.floor(t/self.dt)))
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des[:,i]
        de = v - self.v_des[:,i]
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        grad_x = np.concatenate((e, self.weight_vel*de))
        grad_u = np.zeros(u.shape[0])
        return (cost, grad_x, grad_u)
        
        
class OCPRunningCostQuadraticControl:
    ''' Quadratic cost function for penalizing control inputs '''
    def __init__(self, robot, dt):
        self.robot = robot
        self.dt = dt
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        cost = 0.5*u.dot(u)
        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        cost = 0.5*u.dot(u)
        grad_x = np.zeros(x.shape[0])
        grad_u = u
        return (cost, grad_x, grad_u)


class OCPFinalCostFrame:
    ''' Cost function for reaching a desired position-velocity with a frame of the robot
        (typically the end-effector).
    '''
    def __init__(self, robot, frame_name, p_des, dp_des, weight_vel):
        self.robot = robot
        self.nq = robot.model.nq
        self.frame_id = robot.model.getFrameId(frame_name)
        assert(robot.model.existFrame(frame_name))
        self.p_des  = p_des   # desired 3d position of the frame
        self.dp_des = dp_des  # desired 3d velocity of the frame
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        v_frame = self.robot.frameVelocity(q, v, self.frame_id, recompute)
        dp = v_frame.linear # take linear part of 6d velocity
        
        e = p - self.p_des
        de = dp - self.dp_des 
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        
        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        q = x[:self.nq]
        v = x[self.nq:]
        
        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        v_frame = self.robot.frameVelocity(q, v, self.frame_id, recompute)
        dp = v_frame.linear # take linear part of 6d velocity
        
        e = p - self.p_des
        de = dp - self.dp_des 
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        grad =  np.concatenate((e, self.weight_vel*de))
        #print( e ,de, cost, grad)
        return (cost, grad)