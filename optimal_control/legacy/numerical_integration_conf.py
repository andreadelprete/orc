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

T = 1.0                     # integration horizon in seconds
dt = 0.1                    # time step duration in seconds
N = int(T/dt)               # number of time steps
ndt_ground_truth = 1000     # number of inner time steps used for computing the ground truth
PLOT_STUFF = 1
linestyles = ['-*', '--*', ':*', '-.*']

# choose which system you want to integrate
system = 'ur'
q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ])  # initial configuration

# system='double-pendulum'
# system='pendulum-ode'
# system = 'linear'
# system = 'sin'
# system = 'stiff-diehl'

integrators = []
integrators += [{'scheme': 'RK-1'      , 'nf': 1}]
integrators += [{'scheme': 'RK-2'      , 'nf': 2}] # nf = number of function evaluation per step
integrators += [{'scheme': 'RK-3'      , 'nf': 3}]
integrators += [{'scheme': 'RK-4'      , 'nf': 4}]

ndt_list = np.array([int(i) for i in 2**np.arange(1.5,8,0.5)])