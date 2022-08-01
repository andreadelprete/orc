#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:19:54 2022

@author: adelprete
"""
import numpy as np
import single_shooting_conf as conf
import orc.utils.plot_utils as plut

dt_robot = 0.001

INPUT_FILE_NAMES = ['home_2_table', 'table_2_belt']
#INPUT_FILE_NAME = 'table_2_belt'
#INPUT_FILE_NAME = 'belt_2_home'

OUTPUT_FINE_NAME = 'home_2_table_2_belt'

q_in = []
N_out = 0
ratio = int(conf.dt/dt_robot)
for file_name in INPUT_FILE_NAMES:
    data = np.load(file_name+'.npz') # , q=X[:,:nq], v=X[:,nv:], u=U
    q_in.append(data['q'])
    N_in = q_in[-1].shape[0]
    N_out += 1 + (N_in-1)*ratio
    
nq = q_in[-1].shape[1]
q_out = np.empty((N_out, nq))
index = 0
for k in range(len(INPUT_FILE_NAMES)):
    N_in = q_in[k].shape[0]
    for i in range(N_in-1):
        for j in range(ratio):
            q_out[index,:] = ((ratio-j)*q_in[k][i,:] + j*q_in[k][i+1,:]) / ratio
            index += 1
    q_out[index,:] = q_in[k][-1,:]
    index += 1

# PLOT STUFF
time_out = np.arange(0.0, N_out*dt_robot, dt_robot)[:N_out]
T0 = 0.0

(f, ax) = plut.create_empty_figure(int(nq/2),2)
ax = ax.reshape(nq)
for k in range(len(INPUT_FILE_NAMES)):
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

np.savez_compressed(OUTPUT_FINE_NAME+'_robot', q=q_out)