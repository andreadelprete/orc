#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:34:55 2021

@author: student
"""

def joint_motion_control(q, v, q_ref,  v_ref, dv_ref, kp, kd, h, g, M):
    tau = kp*(q_ref - q) + kd*(v_ref - v)
    return tau