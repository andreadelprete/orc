#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:45:47 2022

@author: adelprete
"""
import numpy as np

fixed_world_translation = np.array([0.5, 0.35, 1.75])

TABLE_STL_FILE = '/home/adelprete/devel/src/locosim/ros_impedance_controller/worlds/models/tavolo/mesh/tavolo.stl'
table_stl_pos = np.array([0.0, 0.0, -0.02])

table_normal = np.array([0., 0., 1.])
table_pos = np.array([0.5, 0.4, 0.83])
table_size = np.array([1.2, 1.0, 0.02])
table_color = (0.2, 0.2, 0.8, 0.5)

backwall_size = np.array([1.0, 0.01, 2.0])
backwall_pos = np.array([0.5, 0.02, 1.0])
backwall_color = (0.2, 0.2, 0.8, 0.5)

def display_disi_lab(simu):
    # display table and stuff
    simu.gui.addBox("world/table_box", table_size[0], table_size[1], table_size[2], table_color)
    simu.robot.applyConfiguration("world/table_box", table_pos.tolist()+[0, 0, 0, 1])
    
    if(simu.gui.addMesh('world/table', TABLE_STL_FILE)):
        print("table added")
        simu.gui.setScale('world/table', (0.001,)*3) # scale table stl because it's expressed in mm
        simu.gui.callVoidProperty('world/table', 'ApplyScale')
    simu.robot.applyConfiguration("world/table", table_stl_pos.tolist()+[0, 0, 0, 1])
    
    simu.gui.addBox("world/backwall", backwall_size[0], backwall_size[1], backwall_size[2], backwall_color)
    simu.robot.applyConfiguration("world/backwall", backwall_pos.tolist()+[0, 0, 0, 1])
    
    simu.gui.addLight("world/table_light", "python-pinocchio", 0.1, (1.,1,1,1))
    simu.robot.applyConfiguration("world/table_light", (table_pos[0], table_pos[1], table_pos[2]+1.5, 0, 0, 0, 1))
    
#    simu.gui.addURDF('world/finger', '/home/adelprete/devel/src/locosim/gripper_description/urdf/finger.urdf')
#    simu.gui.addURDF('world/finger', '/mnt/hgfs/My Drive/[LM] Advanced Robot Control/code/lab_doc/grippers/gripper_description/urdf/finger.urdf')