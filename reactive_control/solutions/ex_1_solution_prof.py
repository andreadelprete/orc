#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:34:55 2021

@author: student
"""
import numpy as np
from numpy.linalg import inv

def operational_motion_control(q, v, ddx_des, h, M, J, dJdq, conf):
    use_simplified_law = False
    Minv = inv(M)
    Lambda = inv( J @ Minv @ J.T)
    if(use_simplified_law):
        tau = J.T @ Lambda @ ddx_des + h
    else:
        mu = J.T @ Lambda @ (J @ Minv @ h - dJdq)
        tau = J.T @ Lambda @ ddx_des + mu

    # add secondary task in joint space
    N = np.eye(v.shape[0]) - J.T @ Lambda @ J @ Minv
    tau1 = M @ (conf.kp_j * (conf.q0 - q) - conf.kd_j * v)
    if(use_simplified_law):
        tau += N @ tau1 
    else:
        tau += N @ (tau1 + h)

    return tau