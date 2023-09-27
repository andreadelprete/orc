import pinocchio as se3
import tsid
import numpy as np
import os
from orc.utils.robot_loaders import loadUR, loadUR_urdf
import time
import subprocess


class TsidManipulator:
    ''' Standard TSID formulation for a robot manipulator
        - end-effector task
        - Postural task
        - torque limits
        - pos/vel limits
    '''
    
    def __init__(self, conf, viewer=True):
        self.conf = conf
        urdf, path = loadUR_urdf()
        self.robot = tsid.RobotWrapper(urdf, [path], False)
        robot = self.robot
        self.model = model = robot.model()
        try:
            se3.loadReferenceConfigurations(model, conf.srdf, False)
            q = model.referenceConfigurations['default']
        except:
            q = conf.q0
        v = np.zeros(robot.nv)
        
        assert model.existFrame(conf.ee_frame_name)
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
                
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv))
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv))
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        self.eeTask = tsid.TaskSE3Equality("task-ee", self.robot, self.conf.ee_frame_name)
        self.eeTask.setKp(self.conf.kp_ee * np.ones(6))
        self.eeTask.setKd(2.0 * np.sqrt(self.conf.kp_ee) * np.ones(6))
        self.eeTask.setMask(conf.ee_task_mask)
        self.eeTask.useLocalFrame(False)
        self.EE = model.getFrameId(conf.ee_frame_name)
        H_ee_ref = self.robot.framePosition(formulation.data(), self.EE)
        self.trajEE = tsid.TrajectorySE3Constant("traj-ee", H_ee_ref)
        formulation.addMotionTask(self.eeTask, conf.w_ee, 1, 0.0)
        
        self.tau_max = conf.tau_max_scaling*model.effortLimit
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if(conf.w_torque_bounds>0.0):
            formulation.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, 0, 0.0)
        
        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, conf.dt)
        self.v_max = conf.v_max_scaling * model.velocityLimit
        self.v_min = -self.v_max
        jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if(conf.w_joint_bounds>0.0):
            formulation.addMotionTask(jointBoundsTask, conf.w_joint_bounds, 0, 0.0)
        
        trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q)
        postureTask.setReference(trajPosture.computeNext())
        
        solver = tsid.SolverHQuadProgFast("qp solver")
        solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        self.trajPosture = trajPosture
        self.postureTask  = postureTask
        self.actuationBoundsTask = actuationBoundsTask
        self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.solver = solver
        self.q = q
        self.v = v
