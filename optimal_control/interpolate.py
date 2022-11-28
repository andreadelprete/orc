#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:19:54 2022

@author: adelprete
"""
import numpy as np
import orc.optimal_control.single_shooting.single_shooting_conf as conf
import orc.utils.plot_utils as plut
import matplotlib.pyplot as plt


dt_robot = 0.001

#INPUTS = [{'filename': 'home_2_table', 'pause': 1.0}, 
#          {'filename': 'table_2_belt', 'pause': 1.0},
#          {'filename': 'belt_2_home',  'pause': 0.0}]
#OUTPUT_FINE_NAME = 'home_2_table_2_belt_2_home'

#INPUTS = [{'filename': 'table_2_belt', 'pause': 0.5}]
#OUTPUT_FINE_NAME = 'table_2_belt'

#INPUTS = [{'filename': 'belt_2_home', 'pause': 0.5}]
#OUTPUT_FINE_NAME = 'belt_2_home'

INPUTS = [{'filename': 'home_2_table', 'pause': 0.1}]
OUTPUT_FINE_NAME = 'home_2_table'


q_in = []
N_out = 0
ratio = int(conf.dt/dt_robot)
for dict_in in INPUTS:
    data = np.load(conf.DATA_FOLDER+dict_in['filename']+'.npz') # , q=X[:,:nq], v=X[:,nv:], u=U
    q_in.append(data['q'])
    N_in = q_in[-1].shape[0]
    N_out += 1 + (N_in-1)*ratio + int(dict_in['pause']/dt_robot)
    
nq = q_in[-1].shape[1]
q_out = np.zeros((N_out, nq))
index = 0
for (k, dict_in) in enumerate(INPUTS):
    N_in = q_in[k].shape[0]
    for i in range(N_in-1):
        for j in range(ratio):
            q_out[index,:] = ((ratio-j)*q_in[k][i,:] + j*q_in[k][i+1,:]) / ratio
            index += 1
    q_out[index,:] = q_in[k][-1,:]
    index += 1
    
    for i in range(int(dict_in['pause']/dt_robot)):
        q_out[index,:] = q_in[k][-1,:]
        index += 1
        
print("index", index)
print("N_out", N_out)


'''**************'''
'''  PLOT STUFF  '''
'''**************'''
time_out = np.arange(0.0, N_out*dt_robot, dt_robot)[:N_out]
T0 = 0.0

(f, ax) = plut.create_empty_figure(int(nq/2),2)
ax = ax.reshape(nq)
for k in range(len(INPUTS)):
    T1 = T0 + q_in[k].shape[0]*conf.dt
    time_in  = np.arange(T0, T1, conf.dt)[:q_in[k].shape[0]]
    T0 = T1
    for i in range(nq):
        ax[i].plot(time_in,  q_in[k][:,i],  label='q_in', alpha=0.7)
        ax[i].plot(time_out, q_out[:,i], '--', label='q_out', alpha=0.7)
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
    leg = ax[-1].legend()
    leg.get_frame().set_alpha(0.5)

plt.show()

np.savez_compressed(conf.DATA_FOLDER+OUTPUT_FINE_NAME+'_robot', q=q_out)