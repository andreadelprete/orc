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
x_des = -np.array([0.7, 0.1, 0.2])  # test 01
#x_des = -np.array([1.2, 0.0, 0.2]) # test 02/03 hessian_regu=1e-8 ||x_des-x||=0.284725, norm(gradient)=0.016408
#x_des = -np.array([1.2, 0.0, 0.2]) # test 04    hessian_regu=1e-4 ||x_des-x||=0.284500, norm(gradient)=0.007199
#x_des = -np.array([1.2, 0.0, 0.2]) # test 05    hessian_regu=1e-3 ||x_des-x||=0.284414, norm(gradient)=0.000328
#x_des = -np.array([1.2, 0.0, 0.2]) # test 06    hessian_regu=1e-1 Iter 76, ||x_des-x||=0.284414, norm(gradient)=0.000001
#x_des = -np.array([1.2, 0.0, 0.2]) # test 07    hessian_regu=1e0  ||x_des-x||=0.288447, norm(gradient)=0.006598
#x_des = -np.array([0.7, 0.1, 0.2])  # test 08    hessian_regu=1e-1 
frame_name = 'tool0'
MAX_ITER = 300
absolute_threshold = 1e-4    # absolute tolerance on position error
gradient_threshold = 1e-6   # absolute tolerance on gradient's norm
hessian_regu = 1e-1         # Hessian regularization
beta = 0.1                  # backtracking line search parameter
gamma = 0                # line search convergence parameter
line_search = 1         # flag to enable/disable line search

randomize_robot_model = 0
model_variation = 30.0
simulate_coulomb_friction = 1
simulation_type = 'timestepping' #either 'timestepping' or 'euler'
tau_coulomb_max = 10*np.ones(6) # expressed as percentage of torque max

use_viewer = True
which_viewer = 'meshcat'
show_floor = False
PRINT_N = 1                   # print every PRINT_N time steps
DISPLAY_N = 1              # update robot configuration in viwewer every DISPLAY_N time steps
DISPLAY_T = 0.1
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
REF_SPHERE_RADIUS = 0.05
REF_SPHERE_COLOR = (1., 0., 0., 1.)
