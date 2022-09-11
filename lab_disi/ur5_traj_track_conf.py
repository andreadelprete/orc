# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""
import numpy as np

# HOMING PROCEDURE
v_des_homing = 0.2

# Trajectory tracking
data_folder = 'data/'
# list of trajectories file names to track in sequence
traj_file_name = [data_folder+'home_2_table_robot', 
                  data_folder+'table_2_belt_robot', 
                  data_folder+'belt_2_home_robot']
traj_slow_down_factor = 1 # how many times the reference trajectory has to be slowed down
