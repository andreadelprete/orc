# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

T = 3.0                         # OCP horizon
dt = 0.02                     # OCP time step
integration_scheme = 'RK-1'
use_finite_diff = False
max_iter = 3

data_folder = '../data/'

#DATA_FILE_NAME = 'home_2_table_min_acc'
#DATA_FILE_NAME = 'table_2_belt'
DATA_FILE_NAME = data_folder+'belt_2_home_new'

#INITIAL_GUESS_FILE = None # use None if you don't have an initial guess
#INITIAL_GUESS_FILE = 'home_2_table_min_acc' 
#INITIAL_GUESS_FILE = 'table_2_belt'
INITIAL_GUESS_FILE = data_folder+'belt_2_home_new'

#system = 'ur'
#system = 'ur-lab'
#system='double-pendulum'
system='pendulum'

weight_final_pos = 0    # cost function weight for final end-effector position
weight_final_vel = 0    # cost function weight for final end-effector velocity
weight_final_q  = 0     # cost function weight for final joint positions
weight_final_dq = 0     # cost function weight for final joint velocities
weight_dq  = 1e-1       # cost function weight for joint velocities
weight_ddq = 1e-2       # cost function weight for joint accelerations
weight_u   = 0e-2       # cost function weight for joint torques

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
    # list of frames that should not collide with table, and their minimum distance
    table_collision_frames = [('gripper',       0.045),
                              ('tool0',         0.07),
                              ('wrist_1_joint', 0.07),
                              ('wrist_2_joint', 0.07),
                              ('forearm_link_end_fixed_joint', 0.07)]
    # list of frame pairs that should not collide together
    self_collision_frames = [('gripper',       'shoulder_pan_joint', 0.15),
                             ('tool0',         'shoulder_pan_joint', 0.15),
                             ('wrist_1_joint', 'shoulder_pan_joint', 0.15),
                             ('wrist_2_joint', 'shoulder_pan_joint', 0.15),
                             ('forearm_link_end_fixed_joint', 'shoulder_pan_joint', 0.15)]
#    self_collision_frames = []
    
    q_home   = np.array([-0.32932, -0.77775, -2.5674, -1.6349, -1.57867, -1.00179])  # initial configuration
    q_belt_1 = np.array([0.134582, -2.607235, -0.130109, -1.954317, -1.548532, -0.661044])
    q_table  = np.array([-0.176686, -1.049217, -1.886168, -1.754567, -1.522096, -1.082379])
    q_belt_2 = np.array([-3.313833, -2.238478, -1.068578, -1.395336, -1.554650, -0.658182])
    
#    q0, q_des = q_home, q_table
#    q0, q_des = q_table, q_belt_2
    q0, q_des = q_belt_2, q_home
    
    p_des = np.array([0.5, 0.4, 1.05])   # desired end-effector final position
    R_des = np.identity(3)              # desired end-effector final orientation
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
