#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:45:47 2022

@author: adelprete
"""
import numpy as np

table_normal = np.array([0., 0., 1.])
table_pos = np.array([0.5, 0.4, 0.83]) #-0.7
table_size = np.array([1.5, 1.5, 0.04])
table_color = (0.5, 0.5, 0.5, 1.)

backwall_size = np.array([1.0, 0.01, 2.0])
backwall_pos = np.array([0.5, 0.05, 1.0])
backwall_color = (0.3, 0.3, 0.3, 1.)

def display_disi_lab(simu):
    # display table and stuff
    simu.gui.addBox("world/table", table_size[0], table_size[1], table_size[2], table_color)
    simu.robot.applyConfiguration("world/table", table_pos.tolist()+[0, 0, 0, 1])
    
    simu.gui.addBox("world/backwall", backwall_size[0], backwall_size[1], backwall_size[2], backwall_color)
    simu.robot.applyConfiguration("world/backwall", backwall_pos.tolist()+[0, 0, 0, 1])
    
    simu.gui.addLight("world/table_light", "python-pinocchio", 0.1, (1.,1,1,1))
    simu.robot.applyConfiguration("world/table_light", (table_pos[0], table_pos[1], table_pos[2]+1.5, 0, 0, 0, 1))