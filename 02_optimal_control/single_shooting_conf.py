# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

T = 2.0                         # OCP horizon
dt = 0.02                     # OCP time step
integration_scheme = 'RK-1'
use_finite_diff = False

#system = 'ur'
system = 'ur-lab'
#system='double-pendulum'
#system='pendulum'

table_normal = np.array([0., 0., 1.])
table_height = -0.7
safety_margin = 0.05

weight_final_pos = 10 # cost weight for final end-effector position
weight_vel = 1e-1   # cost function weight for final end-effector velocity
weight_dq = 1e-3     # cost function weight for joint velocities
weight_u = 1e-6     # cost function weight for joint torques

if(system=='ur'):
    nq = 6
    frame_name = 'ee_link'    # name of the frame to control (end-effector)
    q0    = np.array([ 0. , -1.0,      0.7,   0. ,  0. ,  0. ])  # initial configuration
    q_des = np.array([ 0. , -np.pi/2,  0.0 ,  0. ,  0. ,  0. ])  # final configuration
    p_des = np.array([0.6, 0.2, 0.4])   # desired end-effector final position
    R_des = np.identity(3)              # desired end-effector final orientation
    B = 0*np.array([10., 10., 10., 5., 1., 1.]) # joint viscous friction coefficient
elif(system=='ur-lab'):
    nq = 6
    frame_name = 'tool0'
    collision_frames = [frame_name, 'wrist_1_joint', 'wrist_2_joint']
    q0    = np.array([ 0. , -1.0,      0.7,   0. ,  0. ,  0. ])  # initial configuration
    p_des = np.array([0.0, 0.0, -0.5])   # desired end-effector final position
    B = np.array([10., 10., 10., 5., 1., 1.]) # joint viscous friction coefficient
elif(system=='pendulum'):
    nq = 1
    frame_name = 'joint1'
    q0 = np.array([np.pi/2])
    p_des = np.array([0.6, 0.2, 0.4])   # desired end-effector final position
    B = np.zeros(nq) # joint viscous friction coefficient
elif(system=='double-pendulum'):
    nq = 2
    frame_name = 'joint2'
    q0 = np.array([np.pi+0.3, 0.0])
    p_des = np.array([0.0290872, 0, 0.135]) # upper position
    q_des = np.array([0.0, 0.0])
    B = np.zeros(nq) # joint viscous friction coefficient

x0 = np.concatenate((q0, np.zeros(nq)))  # initial state
dp_des      = np.zeros(3)                   # desired end-effector final linear velocity
w_des = np.zeros(3)                     # desired end-effector final angular velocity

# SIMULATION PARAMETERS
simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(nq)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [1.0568891763687134, 0.7100808024406433, 0.39807042479515076, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
