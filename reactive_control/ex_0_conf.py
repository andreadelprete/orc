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

T_SIMULATION = 5             # number of time steps simulated
dt = 0.01                      # controller time step
ndt = 10
q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ]).T  # initial configuration

kp = 50               # proportional gain of joint posture task
kd = 2*sqrt(kp)        # derivative gain of joint posture task

# PARAMETERS OF REFERENCE SINUSOIDAL TRAJECTORY
amp                  = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])           # amplitude
phi                  = np.array([0.0, 0.5*np.pi, 0.0, 0.0, 0.0, 0.0])     # phase
freq                 = np.array([1.0, 0.5, 0.3, 0.0, 0.0, 0.0])           # frequency (time 2 PI)

simulate_coulomb_friction = 0
simulation_type = 'timestepping' #either 'timestepping' or 'euler'
tau_coulomb_max = 1.0*np.ones(6) # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
which_viewer = 'meshcat'
simulate_real_time = True
show_floor = False
PRINT_T = 1                   # print every PRINT_N time steps
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
