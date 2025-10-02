#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:34:55 2021

@author: student
"""
import numpy as np
from numpy.linalg import inv

def operational_motion_control(q, v, ddx_des, h, M, J, dJdq, conf):
    tau = h
    return tau