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

T = 1.5                       # OCP horizon
dt = 0.05                     # OCP time step
integration_scheme = 'RK-4'
use_finite_difference = 1

system = 'ur'
#system='double-pendulum'
#system='pendulum'

weight_vel = 1e-3   # cost function weight for final velocity (for position it's implicitely 1)
weight_u = 1e-6     # cost function weight for control

if(system=='ur'):
    nq = 6
    
    q0 = np.array([ 0. , 1.57,  0.,  0. ,  0. ,  0. ])  # initial configuration
    x0 = np.concatenate((q0, np.zeros(6)))               # initial state
    q_des = np.array([ 0. , -1.57,  0.,  0. ,  0. ,  0.])
    
    #q0 = np.array([0. , 0. , 0. ])  # initial configuration
    #x0 = np.concatenate((q0, np.zeros(6)))  # initial state
    p_des = np.array([0.6, 0.2, 0.4]).T   # desired end-effector final position
    
    
    dp_des = np.zeros(3)
    frame_name = 'tool0'    # name of the frame to control (end-effector)
     
elif(system=='pendulum'):
    nq = 1
    q0 = np.array([np.pi/2])
    x0 = np.concatenate((q0, np.zeros(1)))  # initial state
    p_des = np.array([0.6, 0.2, 0.4]).T   # desired end-effector final position
    frame_name = 'joint1'
elif(system=='double-pendulum'):
    nq = 2
    q0 = np.array([np.pi+0.3, 0.0])
    x0 = np.concatenate((q0, np.zeros(nq)))  # initial state
    p_des = np.array([0.0290872, 0, 0.135]) # upper position
    q_des = np.array([0.0, 0.0])
    frame_name = 'joint2'

#dp_des      = np.zeros(3)                   # desired end-effector final velocity


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
