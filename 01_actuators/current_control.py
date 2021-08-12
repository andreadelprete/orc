from __future__ import print_function
import numpy as np

class CurrentControl:
    ''' A simple proportional motor torque controller
    '''
    def __init__(self, R, K_b, k_current):
        self.R = R
        self.K_b = K_b
        self.k_current = k_current

    def compute(self, i, i_ref, dq=0.0):
        u = self.R*(i_ref + self.k_current*(i_ref - i)) + self.K_b*dq
        return u

if __name__=='__main__':
    import arc.utils.plot_utils as plut
    from numpy import sin, cos, pi
    import matplotlib.pyplot as plt
    from dc_motor import Motor, get_motor_parameters
    from pos_control import PositionControl
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    dt = 1e-4       # controller time step
    T = .5         # simulation time
    motor_name = 'Maxon148877'
    if motor_name=='Focchi2013':
        # gains for motor Focchi2013
        kp = 50.0       # proportional gain
        kd = 5.0        # derivative gain
        k_current = 0.5   # current proportional gain
    elif motor_name=='Maxon148877':
        kp = 1.0       # proportional gain
        kd = .1        # derivative gain
        k_current = 0.5   # current proportional gain
    ki = 0.0        # integral gain
    q_b = 0.5       # initial motor angle
    q_a = 0.       # linear increase in motor angle per second
    q_w = 1.0       # frequency of sinusoidal reference motor angle
    q_A = 0.0       # amplitude of sinusoidal reference motor angle
    params = get_motor_parameters(motor_name)
    
    # create motor and position controller
    motor = Motor(dt, params)
    pos_control = PositionControl(kp, kd, ki, dt)
    current_control = CurrentControl(params.R, params.K_b, k_current)

    # simulate motor with linear+sinusoidal reference motor angle
    N = int(T/dt)   # number of time steps
    q = np.zeros(N+1)
    dq = np.zeros(N+1)
    current = np.zeros(N+1)
    V = np.zeros(N)
    q_ref  = np.zeros(N)
    dq_ref = np.zeros(N)
    current_ref = np.zeros(N)
    omega = 2*pi*q_w
    for i in range(N):
        t = i*dt
        q_ref[i]  = q_a*t + q_b + q_A*sin(omega*t)
        dq_ref[i] = q_a + q_A*omega*cos(omega*t)
        current_ref[i] = pos_control.compute(q[i], dq[i], q_ref[i], dq_ref[i])
        V[i] = current_control.compute(current[i], current_ref[i])
        motor.simulate(V[i])
        q[i+1] = motor.q()
        dq[i+1] = motor.dq()
        current[i+1] = motor.i()

    # plot motor angle, velocity and current
    f, ax = plt.subplots(3,1,sharex=True)
    time = np.arange(0.0, T+dt, dt)
    ax[0].plot(time, q, label ='motor angle')
    ax[0].plot(time[:-1], q_ref, '--', label ='ref. motor angle')
    ax[1].plot(time, dq, label ='motor velocity')
    ax[1].plot(time[:-1], dq_ref, '--', label ='ref. motor velocity')
    ax[2].plot(time, current, label ='motor current')
    ax[2].plot(time[:-1], current_ref, '--', label ='ref. motor current')
    for i in range(3): ax[i].legend()
    plt.xlabel('Time [s]')
    plt.show()

# By increasing K_b we get more torque from same current, but we also get
# more back-EMF => current control is more important

# Increasing too much current feedback de-stabilizes system: why?