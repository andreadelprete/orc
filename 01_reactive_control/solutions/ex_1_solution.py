#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:34:55 2021

@author: student
"""
import numpy as np
from numpy.linalg import inv

def operational_motion_control(q, v, ddx_des, h, M, J, dJdq, conf):
    Minv = inv(M)
    J_Minv = J @ Minv
    Lambda = inv(J_Minv @ J.T)
#    if(not np.isfinite(Lambda).all()):
#    print('Eigenvalues J*Minv*J.T', np.linalg.eigvals(J_Minv @ J.T))
    mu = Lambda @ (J_Minv @ h - dJdq)
    f = Lambda @ ddx_des + mu
    tau = J.T @ f
    # secondary task
    J_T_pinv = Lambda @ J_Minv
    nv = J.shape[1]
    NJ = np.eye(nv) - J.T @ J_T_pinv
    tau_0 = M @ (conf.kp_j * (conf.q0 - q) - conf.kd_j*v) + h
    tau += NJ @ tau_0
    
#    tau[:,i] = h + J.T @ Lambda @ ddx_des[:,i] + NJ @ tau_0
    
#    print("tau", tau[:,i].T)
#    print("JT*f", (J.T @ f).T)
#    print("dJdq", (J.T @ Lambda @ dJdq).T)

    return tau