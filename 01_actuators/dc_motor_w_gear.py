import numpy as np
from dc_motor import get_motor_parameters as gmp

def get_motor_parameters(name):
    params = gmp(name)
    if(name=='Focchi2013'):
        params.N = 100                  # gear ratio
        params.tau_coulomb_gear = 0.5   # Coulomb friction at the gear
        params.I_j = 0.2                # joint-side inertia
        params.b_j = 1.0                # joint viscous friction
    elif(name=="Maxon148877"): # Maxon 148877 (150W,48V)
        params.N = 100                  # gear ratio
        params.tau_coulomb_gear = 5.0   # Coulomb friction at the gear
        params.I_j = 1.0                # joint-side inertia
        params.b_j = 0.1                # joint viscous friction
    return params


class MotorWGear:
    ''' A DC motor with gearbox, with the following dynamics (neglecting electric pole)
            V = R*i + K_b*dq_m
            tau_m = K_b*i
            (I_j + N^2 I_m) ddq_j + (b_j + N^2 b_m) dq_j + N tau_c_m + tau_c_g = N tau_m
        where:
            V = voltage
            i = current
            R = resistance
            K_b = motor speed/torque constant
            dq_m = motor velocity
            dq_j = joint velocity
            tau_m = motor torque
            N = gear ratio
            I_m = motor inertia
            b_m = motor viscous friction coefficient
            I_j = joint inertia
            b_j = joint viscous friction coefficient
            tau_c_m = motor Coulomb friction
            tau_c_g = gear Coulomb friction

        Define the system state as joint angle q_j and velocity dq_j:
            x = (q_j, dq_j)
        and the control input is the motor current i.
    '''
    
    def __init__(self, dt, params):
        # store motor parameters in member variables
        self.dt  = dt               # simulation time step
        self.R   = params.R         # motor resistance
        self.K_b = params.K_b       # motor speed constant
        self.I_m = params.I_m       # motor rotor inertia
        self.b_m = params.b_m       # motor viscous friction
        self.N   = params.N         # gear ratio
        self.I_j = params.I_j       # joint-side inertia
        self.b_j = params.b_j       # joint viscous friction
        self.tau_c_m = params.tau_coulomb       # motor Coulomb friction
        self.tau_c_g = params.tau_coulomb_gear  # gear Coulomb friction

        # set initial motor state to zero
        self.x = np.zeros(2)
        
        # compute total inertia, viscous friction and Coulomb friction reflected at the joint side
        self.I = self.I_j + (self.N*self.N) * self.I_m
        self.b = self.b_j + (self.N*self.N) * self.b_m
        self.tau_0 = self.tau_c_m*self.N + self.tau_c_g

    def set_state(self, x):
        self.x = np.copy(x)
        
    def simulate_voltage(self, V, method='time-stepping'):
        ''' Simulate assuming voltage as control input '''
        dq_m = self.N*self.x[1]
        i = (V - self.K_b*dq_m)/self.R
        self.simulate(i, method)
        
    def simulate(self, i, method='time-stepping'):
        dq_j = self.x[1]
        dq_m = self.N*dq_j
        self.current = i
        self.voltage = self.R*i +self.K_b*dq_m 
        self.tau_m = self.K_b*self.current
        
        # I*dq_j' + dt*tau_c = I*dq_j + dt*(N*tau_m - b*dq_j)
        s = self.dt * (self.N*self.tau_m - self.b*dq_j) + self.I * dq_j
        
        # compute friction torque
        if(method=='time-stepping'):
            if np.abs(s/self.dt) <= self.tau_0:
                self.tau_f = s/self.dt
            else:
                self.tau_f = self.tau_0*np.sign(s)
        elif(method=='standard'):
            if dq==0.0:
                if np.abs(s/self.dt) < self.tau_0:
                    self.tau_f = s/self.dt
                else:
                    self.tau_f = self.tau_0*np.sign(s)
            else:
                self.tau_f = self.tau_0*np.sign(dq)
        else:
            print("ERROR: unknown integration method:", method)
            return self.x
        
        # compute next state
        self.x[0] += self.dt*dq_j
        self.x[1] = (s-self.dt*self.tau_f)/self.I
        
        return self.x
     
    def q(self):
        return self.x[0]
        
    def dq(self):
        return self.x[1]

    def i(self):
        return self.current
           
    def tau(self):
        return self.N*self.tau_m        
        
    def V(self):
        return self.voltage
        
    def tau_coulomb(self):
        return self.tau_f
        

if __name__=='__main__':
    import arc.utils.plot_utils as plut
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    dt = 1e-3       # controller time step
    T = 3.0         # simulation time
    u_b = 0.0       # initial control input
    u_a = 0.0       # linear increase in control input per second
    u_w = 2.0       # frequency of sinusoidal control input
    u_A = 1.0       # amplitude of sinusoidal control input
    params = get_motor_parameters('Maxon148877')
#    params.N = 20
    
    # simulate motor with linear+sinusoidal input current
    motor = MotorWGear(dt, params)
    N = int(T/dt)   # number of time steps
    q = np.zeros(N)
    dq = np.zeros(N)
    current = np.zeros(N)
    tau = np.zeros(N)
    tau_f = np.zeros(N)
    V = np.zeros(N)
    for i in range(N):
        t = i*dt
        q[i] = motor.q()
        dq[i] = motor.dq()
        
        current[i] = u_a*t + u_b + u_A*np.sin(2*np.pi*u_w*t)
        motor.simulate(current[i])
        
        V[i] = motor.V()
        tau[i] = motor.tau()
        tau_f[i] = motor.tau_coulomb()

    # plot joint angle, velocity and torque
    f, ax = plt.subplots(4,1,sharex=True)
    time = np.arange(0.0, T, dt)
    ax[0].plot(time, q, label ='angle')
    ax[1].plot(time, dq, label ='velocity')
    ax[2].plot(time, tau, label ='tau')
    ax[2].plot(time, tau_f, label ='tau coulomb')
    ax[3].plot(time, V, label ='voltage')
    for i in range(4): ax[i].legend()
    plt.xlabel('Time [s]')
    plt.show()
