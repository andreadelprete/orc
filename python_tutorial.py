#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 00:15:31 2022

@author: student
"""
from math import sqrt

# create the variable x
x = 3.54342
'''
Print the varriable x
It's a very important variable
'''
print('x=', x, " the end.")

condition = x>0
condition = True # False
if(condition):
    print("x is posittive")
    x = x+1
else:
    print("x is negative")
    
#for i in range(5):
#    print(i)
    
#for i in range(3, 8):
#    print(i)
    
x = [3, 8, 9, 1, 3]

i= 1
while(i<5):
    print(i)
    i += 1 # i = i + 1

x = None

def maxi(x, y=0, z=0):
    if(x>y and x>z):
        return x
    if(y>x and y>z):
        return y
    return (x,z)

print("maxi=", maxi(x=1, z=5))


class Point2D:
    # "self" is equivalent to "this" in C++ 
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def norm(self):
        return sqrt(self.x**2 + self.y**2)     
    
p = Point2D(3, 4)
print("norm of p:", p.norm())
