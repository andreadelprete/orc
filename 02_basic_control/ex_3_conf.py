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
T_SIMULATION = 10             # simulation time
dt = 0.001                   # controller time step
ndt = 10                      # number of integration steps for each control loop

kp = 100                 # proportional gain of end effector task
kd = 2*sqrt(kp)          # derivative gain of end effector task
kp_j = 1                 # proportional gain of joint task
kd_j = 2*sqrt(kp)        # derivative gain of joint task
#frame_name = 'tool0'    # name of the frame to control (end-effector)
frame_name = 'ee_link'  # name of the frame to control (end-effector)
contact_frames = [frame_name]

# data of the contact surface
contact_surface_name = 'wall'
contact_surface_pos = np.array([0.65, 0.2, 0.4])
contact_normal = np.array([-1., 0., 0.])
K = 3e4*np.diagflat([1., 1., 1.])
B = 2e2*np.diagflat([1., 1., 1.])
mu = 0.3

# PARAMETERS OF REFERENCE SINUSOIDAL TRAJECTORY
x0            = np.array([0.65, 0.2, 0.4])          # offset
x_amp         = np.array([0.0, 0.1, 0.1])           # amplitude
x_phi         = np.array([0.0, 0.0, 0.5*np.pi])     # phase
x_freq        = np.array([0.5, 0.5, 0.5])           # frequency

# PARAMETERS OF REFERENCE FORCE TRAJECTORY
f0            = np.array([-20., 0.0, 0.0])          # offset
f_amp         = np.array([0, 0.0, 0.0])            # amplitude
f_phi         = np.array([0.0, 0.0, 0.0])           # phase
f_freq        = np.array([0.5, 0.5, 0.3])           # frequency

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'timestepping' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(6)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                   # print some info every PRINT_T seconds
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [-0.19721254706382751, 1.7667365074157715, 0.43594270944595337, -0.05127447843551636, 0.6647889018058777, 0.7360751032829285, -0.11670506000518799]
