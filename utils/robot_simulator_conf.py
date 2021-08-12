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


simulate_coulomb_friction = 1
simulation_type = 'timestepping' #either 'timestepping' or 'euler'
tau_coulomb_max = 10*np.ones(6) # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
simulate_real_time = False
show_floor = False
PRINT_T = 1                   # print every PRINT_N time steps
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [2.149773597717285, 1.4441328048706055, 0.6232147812843323, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]

ERROR_MSG = 'You should set the environment variable UR5_MODEL_DIR to something like "$DEVEL_DIR/install/share"\n';
path      = os.environ.get('UR5_MODEL_DIR', ERROR_MSG)
urdf      = path + "/ur_description/urdf/ur5_robot.urdf";
srdf      = path + '/ur_description/srdf/ur5_robot.srdf'
