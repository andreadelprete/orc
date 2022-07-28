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
DATA_FILE_NAME = 'ur5_q_ref'

data = np.load(conf.DATA_FILE_NAME+'.npz') # , q=X[:,:nq], v=X[:,nv:], u=U
q_in = data['q']
N_in = q_in.shape[0]
nq = q_in.shape[1]

ratio = int(conf.dt/dt_robot)
N_out = 1 + (N_in-1)*ratio
q_out = np.empty((N_out, nq))
for i in range(N_in-1):
    for j in range(ratio):
        q_out[i*ratio+j,:] = ((ratio-j)*q_in[i,:] + j*q_in[i+1,:]) / ratio
q_out[-1,:] = q_in[-1,:]

# PLOT STUFF
time_in  = np.arange(0.0, N_in*conf.dt, conf.dt)[:N_in]
time_out = np.arange(0.0, N_out*dt_robot, dt_robot)[:N_out]

(f, ax) = plut.create_empty_figure(int(nq/2),2)
ax = ax.reshape(nq)
for i in range(nq):
    ax[i].plot(time_in,  q_in[:,i],  label='q_in', alpha=0.7)
    ax[i].plot(time_out, q_out[:,i], '--', label='q_out', alpha=0.7)
    ax[i].set_xlabel('Time [s]')
    ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)

np.savez_compressed(DATA_FILE_NAME, q=q_out)