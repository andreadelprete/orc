from __future__ import print_function
from dc_motor_w_coulomb_friction import MotorCoulomb
from dc_motor_w_gear import MotorWGear
from dc_motor_w_gear import get_motor_parameters
import arc.utils.plot_utils
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=1, linewidth=200, suppress=True)

class Empty:
    def __init__(self):
        pass
    
def compute_efficiency(V, motor, dq_max):
    state = np.zeros(2)         # motor state (pos, vel)
    res = Empty()
    res.dq = np.arange(0.0, dq_max, 0.01)
    N = res.dq.shape[0]
    res.tau_f = np.zeros(N)         # friction torque (Coulomb+viscous)
    res.tau = np.zeros(N)           # output torque (motor+friction)
    res.P_m = np.zeros(N)           # mechanical output power
    res.P_e = np.zeros(N)           # electrical input power
    res.efficiency = np.zeros(N)    # motor efficiency (out/in power)
    
    for i in range(N):
        state[1] = res.dq[i]    # set velocity
        motor.set_state(state)
        motor.simulate_voltage(V, method='time-stepping')    # apply constant voltage
        res.tau_f[i] = motor.tau_f + motor.b*res.dq[i]
        res.tau[i] = motor.tau() - res.tau_f[i]
        res.P_m[i] = res.tau[i] * res.dq[i]
        res.P_e[i] = motor.i() * V
#        if np.abs(res.P_e[i])>=1e-6: 
        res.efficiency[i] = res.P_m[i] / res.P_e[i]
    
    i = np.argmax(res.efficiency)
    print("Max efficiency", res.efficiency[i])
    print("reached at velocity", res.dq[i], "and torque", res.tau[i])
    return res

V = 48                      # input voltage
dt = 1e-3                   # time step
params = get_motor_parameters('Maxon148877')
params.N = 3
params.b_j = 0.001

motor = MotorCoulomb(dt, params)
motor_w_gear = MotorWGear(dt, params)

dq_max        = V / motor.K_b        # maximum motor vel for given voltage
dq_max_w_gear = dq_max / params.N    # maximum joint vel with gear

res      = compute_efficiency(V, motor, dq_max)
#res_gear = compute_efficiency(V, motor_w_gear, dq_max_w_gear)

def plot_stuff(res):
    f, ax = plt.subplots(1,1,sharex=True)
    alpha = 0.8
    ax.plot(res.tau, res.dq, label ='dq-tau', alpha=alpha)
    ax.plot(res.tau, res.P_m, label ='P_m', alpha=alpha)
    ax.plot(res.tau, res.P_e, label ='P_e', alpha=alpha)
    dq_max = np.max(res.dq)
    ax.plot(res.tau, res.efficiency * dq_max, label ='efficiency (scaled)', alpha=alpha)
    ax.legend()
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Velocity [rad/s]')
    plt.ylim([0, dq_max])
    
plot_stuff(res)
#plot_stuff(res_gear)

f, ax = plt.subplots(1,1,sharex=True)
alpha = 0.8
ax.plot(res.tau, res.efficiency, label ='efficiency (w/o gear)', alpha=alpha)
#ax.plot(res_gear.tau, res_gear.efficiency, label ='efficiency (w gear)', alpha=alpha)
ax.legend()
plt.xlabel('Torque [Nm]')
plt.ylabel('Efficiency')
plt.ylim([0, 1])
plt.show()
