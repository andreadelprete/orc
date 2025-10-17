# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: Andrea Del Prete (andrea.delprete@unitn.it)
"""

import numpy as np
np.set_printoptions(precision=2, linewidth=200, suppress=True)
LINE_WIDTH = 60

nq = 6
q0    = np.array([ 0. , -1.0,      0.7,   0. ,  0. ,  0. ])  # initial configuration

# SIMULATION PARAMETERS
simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'euler' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(nq)   # expressed as percentage of torque max
randomize_robot_model = 0
use_viewer = True
which_viewer = 'meshcat'
open_viewer = False
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
