#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 01:17:26 2022

@author: student
"""

import numpy as np

def increment(a):
#    a = np.copy(a)
    a[0] = a[0]+1
    return a

x = np.zeros(3)
y = np.array([3., 6., 12.])
A = np.zeros((3, 3))
I = np.identity(3)
B = np.empty((3,4))*np.nan
A = np.random.rand(3,3)
z = increment(y)

