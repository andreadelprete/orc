import numpy as np

class PositionControl:
    ''' A simple PID position controller
        u = kp*(q_ref-q) + kd*(dq_ref-dq) + ki*integral(q_ref-q)
    '''
    def __init__(self, kp, kd, ki, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.error_integral = 0.0

    def compute(self, q, dq, q_ref, dq_ref):
        self.error_integral += self.ki*(q_ref-q)*self.dt
        u = self.kp*(q_ref-q) + self.kd*(dq_ref-dq) + self.error_integral
        return u

if __name__=='__main__':
    import arc.utils.plot_utils as plut
    from numpy import sin, cos, pi
    import matplotlib.pyplot as plt
    from dc_motor import Motor, get_motor_parameters
    from dc_motor_w_coulomb_friction import MotorCoulomb
    from dc_motor_w_gear import MotorWGear
    
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    dt = 1e-3               # controller time step
    T = 2.5                 # simulation time
    simulate_coulomb = 1    # flag to decide whether to simulate Coulomb friction
    use_gear = 0            # flag to decide whether to simulate a gear box
    motor_name = 'Maxon148877'
    
    if motor_name=='Focchi2013':
        # gains for motor Focchi2013
        kp = 5000.0       # proportional gain
        kd = 10.0        # derivative gain
    elif motor_name=='Maxon148877':
        if use_gear:
            kp = 10.       # proportional gain
            kd = 1.        # derivative gain            
        else:
            kp = .1       # proportional gain
            kd = .01        # derivative gain
    ki = 1.0        # integral gain
    q_b = 0.0       # initial motor angle
    q_a = 1.        # linear increase in motor angle per second
    q_w = 1.0       # frequency of sinusoidal reference motor angle
    q_A = 0.0       # amplitude of sinusoidal reference motor angle
    params = get_motor_parameters(motor_name)
    
    # create motor and position controller
    if use_gear:
        motor = MotorWGear(dt, params)
    elif simulate_coulomb:
        motor = MotorCoulomb(dt, params)
    else:
        motor = Motor(dt, params)
    controller = PositionControl(kp, kd, ki, dt)

    # simulate motor with linear+sinusoidal reference motor angle
    N = int(T/dt)   # number of time steps
    q = np.zeros(N)
    dq = np.zeros(N)
    current = np.zeros(N)
    tau = np.zeros(N)
    V = np.zeros(N)
    q_ref  = np.zeros(N)
    dq_ref = np.zeros(N)
    omega = 2*pi*q_w
    for i in range(N):
        t = i*dt
        # compute reference trajectory
        q_ref[i]  = q_a*t + q_b + q_A*sin(omega*t)
        dq_ref[i] = q_a + q_A*omega*cos(omega*t)
        q[i] = motor.q()
        dq[i] = motor.dq()
        
        # compute motor current with controller
        current[i] = controller.compute(q[i], dq[i], q_ref[i], dq_ref[i])
        # send current to the motor
        motor.simulate(current[i])

        V[i] = motor.V()
        tau[i] = motor.tau()

    # plot motor angle, velocity and current
    f, ax = plt.subplots(4,1,sharex=True)
    time = np.arange(0.0, T, dt)
    time = time[:N]
    ax[0].plot(time, q, label ='angle')
    ax[0].plot(time, q_ref, '--', label ='reference angle')
    ax[1].plot(time, dq, label ='velocity')
    ax[1].plot(time, dq_ref, '--', label ='reference velocity')
    ax[2].plot(time, V, label ='voltage')    
    ax[2].plot(time, current*params.R, label ='current (times R)')    
    ax[2].plot(time, dq*params.K_b, label='back-EMF')
    ax[3].plot(time, tau, label=r'$\tau$')
    for i in range(len(ax)): ax[i].legend()
    plt.xlabel('Time [s]')
    
    # plot powers
    P_m = tau*dq        # mechanical power
    P_e = V*current     # electrical power
    P_J = params.R*(current*current)    # Joule loss
    f, ax = plt.subplots(1,1,sharex=True)
    ax.plot(time, P_e, label='Elec. Pow.')
    ax.plot(time, P_m, label='Mech. Pow.')
    ax.plot(time, P_J, label='Joule Loss')
    ax.legend()
    plt.show()
