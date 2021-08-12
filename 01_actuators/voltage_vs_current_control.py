from __future__ import print_function
import numpy as np
import arc.utils.plot_utils as plut
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from dc_motor import Motor, get_motor_parameters
from pos_control import PositionControl
from current_control import CurrentControl
np.set_printoptions(precision=1, linewidth=200, suppress=True)

class Empty():
    def __init__(self):
        pass

def run_simulation(T, dt, motor, pos_control, current_control, q_a, q_b, q_A, q_w):
    # simulate motor with linear+sinusoidal reference motor angle
    N = int(T/dt)   # number of time steps
    data = Empty()
    data.q = np.zeros(N+1)
    data.dq = np.zeros(N+1)
    data.current = np.zeros(N+1)
    data.V = np.zeros(N)
    data.q_ref  = np.zeros(N)
    data.dq_ref = np.zeros(N)
    data.current_ref = np.zeros(N)
    omega = 2*pi*q_w
    tau_ext = 0.0
    for i in range(N):
        t = i*dt
        data.q_ref[i]  = q_a*t + q_b + q_A*sin(omega*t)
        data.dq_ref[i] = q_a + q_A*omega*cos(omega*t)
        data.current_ref[i] = pos_control.compute(data.q[i], data.dq[i], data.q_ref[i], data.dq_ref[i])
        data.V[i] = current_control.compute(data.current[i], data.current_ref[i], data.dq[i])
        motor.simulate_voltage(data.V[i], tau_ext)
        data.q[i+1] = motor.q()
        data.dq[i+1] = motor.dq()
        data.current[i+1] = motor.i()
        if i==int(N/2):
            tau_ext = -.03
            print("apply external torque")
    return data


dt = 1e-3       # controller time step
T = 1.0         # simulation time
motor_name = 'Maxon148877'
if motor_name=='Focchi2013':
    # gains for motor Focchi2013
    k_current = 0.8 # current proportional gain
    kp_torque = 20.0       # proportional gain
    kd_torque = 5.0        # derivative gain
    kp_pos = 20.0       # proportional gain
    kd_pos = 5.0        # derivative gain
elif motor_name=='Maxon148877':
    k_current = 0.0     # current proportional gain
    kp_torque = 1.0       # proportional gain
    kd_torque = 0.1        # derivative gain
    kp_pos = 1.0       # proportional gain
    kd_pos = 0.1        # derivative gain
ki = 0.0        # integral gain
q_b = 0.0       # initial motor angle
q_a = 0.0         # linear increase in motor angle per second
q_w = 5.0       # frequency of sinusoidal reference motor angle
q_A = 10.0       # amplitude of sinusoidal reference motor angle
params = get_motor_parameters(motor_name)
#K_b *= 1e2  # increase back EMF to highlight effect of current control

# create motor and position controller
motor = Motor(dt, params)
pos_control = PositionControl(kp_pos, kd_pos, ki, dt)
torque_control = CurrentControl(params.R, 0.0, 0.0)  # current feedback is set to zero here!
# run simulation without current feedback
data_pos   = run_simulation(T, dt, motor, pos_control, torque_control, q_a, q_b,  q_A, q_w)

# create new motor and torque controller, this time with current feedback
motor = Motor(dt, params)
pos_control = PositionControl(kp_torque, kd_torque, ki, dt)
torque_control = CurrentControl(motor.R, motor.K_b, k_current)
data_torque = run_simulation(T, dt, motor, pos_control, torque_control, q_a, q_b,  q_A, q_w)

# plot motor angle, velocity and current
f, ax = plt.subplots(3,1,sharex=True)
time = np.arange(0.0, T+dt, dt)
time = time[:1+int(T/dt)]
ax[0].plot(time, data_pos.q, label ='angle (V)')
ax[0].plot(time, data_torque.q, '--', label ='angle (i)')
ax[0].plot(time[:-1], data_pos.q_ref, '--', label ='ref. angle')
ax[1].plot(time, data_pos.dq, label ='velocity (V)')
ax[1].plot(time, data_torque.dq, '--', label ='velocity (i)')
ax[2].plot(time, data_pos.current, label ='current (V)')
ax[2].plot(time, data_torque.current, label ='current (i)')
ax[2].plot(time[:-1], data_pos.current_ref, '--', label ='ref. current (V)')
ax[2].plot(time[:-1], data_torque.current_ref, '--', label ='ref. current (i)')
for i in range(3): ax[i].legend()
plt.xlabel('Time [s]')

f, ax = plt.subplots(2,1,sharex=True)
time = np.arange(0.0, T+dt, dt)
time = time[:1+int(T/dt)]
ax[0].plot(time[:-1], data_pos.V, label ='Voltage (V)')
ax[0].plot(time, data_pos.current*params.R, '--', label ='R*i (V)')
ax[0].plot(time, data_pos.dq*params.K_b, '--', label ='back-EMF (V)')
ax[1].plot(time[:-1], data_torque.V, label ='Voltage (i)')
ax[1].plot(time, data_torque.current*params.R, '--', label ='R*i (i)')
ax[1].plot(time, data_torque.dq*params.K_b, '--', label ='back-EMF (i)')
for i in range(2): ax[i].legend()
plt.show()
