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
        
        
class OCPFinalCostState:
    ''' Cost function for reaching a desired state of the robot
    '''
    def __init__(self, name, robot, q_des, v_des, weight_pos, weight_vel):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        self.q_des = q_des   # desired joint angles
        self.v_des = v_des  # desired joint velocities
        self.weight_pos = weight_pos
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        cost = 0.5*self.weight_pos*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        cost = 0.5*self.weight_pos*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        grad =  np.concatenate((self.weight_pos*e, self.weight_vel*de))
        return (cost, grad)
        
        
class OCPRunningCostQuadraticControl:
    ''' Quadratic cost function for penalizing control inputs '''
    def __init__(self, name, robot, dt):
        self.name = name
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
    
    
class OCPRunningCostQuadraticJointVel:
    ''' Quadratic cost function for penalizing the joint velocities 
    '''
    def __init__(self, name, robot):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        v = x[self.nq:]
        cost = 0.5*v.dot(v)
        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        v = x[self.nq:]

        # compute runnning cost
        cost = 0.5*v.dot(v)
        
        #Compute gradient of the runnning cost respect to x and u    
        grad_x =  np.concatenate((np.zeros(self.nq), v))
        grad_u = np.zeros(u.shape[0])
        return (cost, grad_x, grad_u)
    
    
class OCPRunningCostQuadraticJointAcc:
    ''' Quadratic cost function for penalizing the joint accelerations 
    '''
    def __init__(self, name, robot):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        q = x[:self.nq]
        v = x[self.nq:]
        dv = pin.aba(self.robot.model, self.robot.data, q, v, u)
        cost = 0.5*dv.dot(dv)
        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        q = x[:self.nq]
        v = x[self.nq:]
        dv = pin.aba(self.robot.model, self.robot.data, q, v, u)
        cost = 0.5*dv.dot(dv)
        
        #Compute gradient of the runnning cost respect to x and u    
        ddq_dq, ddq_dv, ddq_du = self.robot.abaDerivatives(q, v, u)
        grad_x =  np.concatenate((dv.dot(ddq_dq), 
                                  dv.dot(ddq_dv))) 
        grad_u = dv.dot(ddq_du)
        return (cost, grad_x, grad_u)


class OCPRunningCostQuadraticPosition:
    ''' Quadratic cost function for penalizing time to reach the desired frame position 
    '''
    def __init__(self, name, robot, dt , frame_name, p_des):
        self.name = name
        self.robot = robot
        self.dt = dt
        self.nq = robot.model.nq
        self.frame_id = robot.model.getFrameId(frame_name)
        assert(robot.model.existFrame(frame_name))
        self.p_des  = p_des   # desired 3d position of the frame
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        q = x[:self.nq]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        
        e = p-self.p_des
        cost = 0.5*e.dot(e)
        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        q = x[:self.nq]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        
        e = p-self.p_des
        # compute runnning cost
        cost = 0.5*e.dot(e)
        
        # compute Jacobian J
        self.robot.computeJointJacobians(q)
        self.robot.framesForwardKinematics(q)
        J6 = self.robot.frameJacobian(q, self.frame_id)
        J = J6[:3,:]
        
        #Compute gradient of the runnning cost respect to x and u    
        grad_x =  np.concatenate((e.dot(J), np.zeros(q.shape[0])))
        grad_u = np.zeros(u.shape[0])
        return (cost, grad_x, grad_u)


class OCPFinalCostFrame:
    ''' Cost function for reaching a desired configuration-velocity (in 6d) with a frame of the robot
        (typically the end-effector).
    '''
    def __init__(self, name, robot, frame_name, p_des, dp_des, R_des, w_des, weight_vel):
        self.name = name
        self.robot = robot
        self.nq = robot.model.nq
        self.frame_id = robot.model.getFrameId(frame_name)
        assert(robot.model.existFrame(frame_name))
        self.p_des  = p_des   # desired 3d position of the frame
        self.dp_des = dp_des  # desired 3d velocity of the frame
        self.R_des = R_des    
        self.w_des = w_des
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        R = H.rotation
        v_frame = self.robot.frameVelocity(q, v, self.frame_id, recompute)
        dp = v_frame.linear # take linear part of 6d velocity
        w = v_frame.angular
        
        e, de = np.zeros(6), np.zeros(6)
        e[:3] = p-self.p_des
        de[:3] = dp - self.dp_des
        R_err = self.R_des.T @ R
        e[3:]  = pin.log3(R_err)
        de[3:] = w - self.w_des
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de) #comment velocity in order to run only cost on position
        
        #cost = norm(p-self.p_des) + self.weight_vel*norm(dp - self.dp_des)
        
        return (cost)  
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        R = H.rotation
        v_frame = self.robot.frameVelocity(q, v, self.frame_id, recompute)
        dp = v_frame.linear # take linear part of 6d velocity
        w = v_frame.angular
        
        e, de = np.zeros(6), np.zeros(6)
        e[:3]  = p-self.p_des
        de[:3] = dp - self.dp_des
        R_err = self.R_des.T @ R
        e[3:]  = pin.log3(R_err)
        de[3:] = w - self.w_des
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        
        # compute Jacobian J
        self.robot.computeAllTerms(q, v)
        J = self.robot.frameJacobian(q, self.frame_id)
        Jpos = np.copy(J)
        Jpos[3:,:] = pin.Jlog3(R_err) @ R.T @ J[3:,:]
        
        #compute dJ (derivative of J respect to time)
#        pin.computeJointJacobiansTimeVariation(self.robot.model, self.robot.data, q, v)
#        dJ = pin.getFrameJacobianTimeVariation(self.robot.model, self.robot.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
#        grad_x =  np.concatenate((e.dot(J) + self.weight_vel*de.dot(dJ), self.weight_vel*de.dot(J))) 
        grad_x =  np.concatenate((e.dot(Jpos), self.weight_vel*de.dot(J))) 
        return (cost, grad_x)

class OCPFinalCostFramePos:
    ''' Cost function for reaching a desired position-velocity with a frame of the robot
        (typically the end-effector).
    '''
    def __init__(self, name, robot, frame_name, p_des, dp_des, weight_vel):
        self.name = name
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
        
        e = p-self.p_des
        de = dp - self.dp_des
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de) #comment velocity in order to run only cost on position
        
        #cost = norm(p-self.p_des) + self.weight_vel*norm(dp - self.dp_des)
        
        return (cost)  
    
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        v_frame = self.robot.frameVelocity(q, v, self.frame_id, recompute)
        dp = v_frame.linear # take linear part of 6d velocity
        
        e = p-self.p_des
        de = dp - self.dp_des
        cost = 0.5*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        
        # compute Jacobian J
        self.robot.computeAllTerms(q, v)
        J6 = self.robot.frameJacobian(q, self.frame_id)
        J = J6[:3,:]            # take first 3 rows of J6
        
        #compute dJ (derivative of J respect to time)
        pin.computeJointJacobiansTimeVariation(self.robot.model, self.robot.data, q, v)
        dJ6 = pin.getFrameJacobianTimeVariation(self.robot.model, self.robot.data, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        dJ = dJ6[:3,:]
    
        grad_x =  np.concatenate((e.dot(J) + self.weight_vel*de.dot(dJ), self.weight_vel*de.dot(J))) 
#        grad_x =  np.concatenate((e.dot(J), self.weight_vel*de.dot(J))) 
        return (cost, grad_x)
    