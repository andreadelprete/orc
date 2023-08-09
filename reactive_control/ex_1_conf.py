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

q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ])  # initial configuration
T_SIMULATION = 4             # simulation time
dt = 0.001                   # controller time step
ndt = 10                      # number of integration steps for each control loop

kp = 10                 # proportional gain of end effector task
kd = 2*sqrt(kp)         # derivative gain of end effector task
kp_j = 1                 # proportional gain of end effector task
kd_j = 2*sqrt(kp)         # derivative gain of end effector task
frame_name = 'tool0'    # name of the frame to control (end-effector)

# PARAMETERS OF REFERENCE SINUSOIDAL TRAJECTORY
x0          = np.array([0.6, 0.2, 0.4])         # offset
amp         = np.array([0.1, 0.1, 0.0])           # amplitude
phi         = np.array([0.0, 0.5*np.pi, 0.0])     # phase
freq        = np.array([0.5, 0.5, 0.3])           # frequency (time 2 PI)

amp         = np.array([0.1, 0.1, 0.0])           # amplitude
freq        = np.array([1.0, 1.0, 0.3])           # frequency (time 2 PI)

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(6)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
which_viewer = 'meshcat'
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
