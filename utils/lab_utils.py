#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:45:47 2022

@author: adelprete
"""
import numpy as np
import os

fixed_world_translation = np.array([0.5, 0.35, 1.75])
LOCOSIM_DIR = os.getenv('LOCOSIM_DIR', '/home/adelprete/devel/src/locosim')
TABLE_STL_FILE = LOCOSIM_DIR + '/ros_impedance_controller/worlds/models/tavolo/mesh/tavolo.stl'
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

def display_disi_lab_meshcat(simu):
    import meshcat.geometry as g
    import meshcat.transformations as tf
    table_normal = np.array([0., 0., 1.])
    table_stl_pos = np.array([-0.5, -0.35, -1.77])
    table_pos = np.array([0, 0.05, -0.92])
    table_size = np.array([1.2, 1.0, 0.02])
    backwall_size = np.array([1.0, 0.01, 2.0])
    backwall_pos = np.array([0., -0.33, -0.75])

    table_box = simu.viz.viewer['world/table_box'].set_object(g.Box([table_size[0], table_size[1], table_size[2]]), g.MeshBasicMaterial(color=0x0BFFAA, opacity=0.5))
    table_box = simu.viz.viewer['world/table_box']
    table_box.set_transform(tf.translation_matrix(table_pos)) # This is an absolute translation. The object will be placed at the specified position

    obj = g.StlMeshGeometry.from_file(TABLE_STL_FILE)
    table_t = simu.viz.viewer["world/table"].set_object(obj, g.MeshBasicMaterial(color=0xa5bcb6, opacity=0.5))
    table_t = simu.viz.viewer["world/table"]
    table_t.set_transform(tf.scale_matrix(0.001, table_stl_pos) )

    backwall = simu.viz.viewer['world/backwall'].set_object(g.Box([backwall_size[0], backwall_size[1], backwall_size[2]]), g.MeshBasicMaterial(color=0x0BFFAA, opacity=0.5))
    backwall = simu.viz.viewer['world/backwall']
    backwall.set_transform(tf.translation_matrix(backwall_pos))