'''
tau = I dv + b v + tau_f

Explicit Euler:
dt tau = I (v_next - v) + dt b v + tau_f

we don't multiply tau_f times dt because it's impulsive
I v_next = dt (tau - b v) + I v - tau_f
I v_next = s - tau_f

I need to find v_next and tau_f such that

v_next * tau_f >= 0
-tau_0 <= tau_f <= tau_0

The first constraint ensures that friction is always opposed to motion.
The second one sets a limit on friction. 
To ensure maximum dissipation we could minimize | v_next |^2
This is equivalent to minimizing |s - tau_f|^2
The solution is simple.

If |s|<= tau_0 => tau_f=s 			=> v_next = 0
Otherwise => tau_f = tau_0*sign(s)	=> v_next = (s-tau_0*sign(s))/I

It remains to be proven that this solution guarantees:
    v_next * tau_f >= 0
This is trivial for the case |s| <= tau_0 because v_next=0.
In the other case (|s| > tau_0) we get:
    tau_0*sign(s)*(s-tau_0*sign(s))/I >= 0
    sign(s)*(s-tau_0*sign(s)) >= 0
    |s| - tau_0 >= 0
    |s| >= tau_0
'''
import numpy as np

class MotorCoulomb:

    def __init__(self, dt, params):
        # store motor parameters in member variables
        self.dt  = dt
        self.R   = params.R
        self.K_b = params.K_b
        self.tau_0 = params.tau_coulomb
        self.I = params.I_m
        self.b = params.b_m

        # set initial motor state (pos, vel) to zero
        self.x = np.zeros(2)
        self.torque = 0.0

    def set_state(self, x):
        self.x = np.copy(x)
        
    def simulate(self, i, method='time-stepping'):
        ''' Simulate assuming current as control input '''
        dq = self.x[1]
        self.voltage = self.R*i +self.K_b*dq 
        torque = self.K_b * i
        self.simulate_torque(torque, method)

    def simulate_voltage(self, V, method='time-stepping'):
        ''' Simulate assuming voltage as control input '''
        dq = self.x[1]
        i = (V - self.K_b*dq)/self.R
        torque = self.K_b * i
        self.simulate_torque(torque, method)

    def simulate_torque(self, torque, method='time-stepping'):
        ''' Simulate assuming torque as control input '''
        self.torque = torque
        dq = self.x[1]
        s = self.dt * (self.torque - self.b*dq) + self.I * dq
        
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
        self.x[0] += self.dt*dq
        self.x[1] = (s-self.dt*self.tau_f)/self.I
#        self.x[0] += self.dt*self.x[1] # this would be a semi-implicit integration
        return self.x
     
    def q(self):
        return self.x[0]
        
    def dq(self):
        return self.x[1]

    def i(self):
        return self.torque / self.K_b
    
    def tau(self):
        return self.torque
        
    def tau_coulomb(self):
        return self.tau_f

    def V(self):
        return self.R*self.i() + self.K_b*self.dq()

def run_open_loop(ax, dt, method='time-stepping'):
    from dc_motor import get_motor_parameters
    T = 2         # simulation time
    V_b = 0.0       # initial motor voltage
    V_a = 0.0       # linear increase in motor voltage per second
    V_w = .5
    V_A = 1.1
    params = get_motor_parameters('Focchi2013')
    params.tau_coulomb = 1
    
    # simulate motor with linear+sinusoidal input torque
    N = int(T/dt)   # number of time steps
    motor = MotorCoulomb(dt, params)
#    motor.set_state(np.array([0.0, 1e-5]))
    q = np.zeros(N+1)
    dq = np.zeros(N+1)
    tau_f = np.zeros(N+1)
    tau = np.zeros(N)
    for i in range(N):
        tau[i] = V_a*i*dt + V_b + V_A*np.sin(2*np.pi*V_w*i*dt)
        motor.simulate_torque(tau[i], method)
#        motor.simulate(tau[i], method)
        q[i+1] = motor.q()
        dq[i+1] = motor.dq()
        tau_f[i] = motor.tau_f/dt
        tau[i] = motor.tau()

    # plot motor angle, velocity and current
    time = np.arange(0.0, T+dt, dt)
    time = time[:N+1]
    alpha = 0.8
    ax[0].plot(time, q, label ='q '+method, alpha=alpha)
    ax[1].plot(time, dq, label ='dq '+method, alpha=alpha)
    ax[2].plot(time[:-1], tau, label ='tau '+method, alpha=alpha)
    ax[-1].plot(time, tau_f, '--', label ='tau_f '+method, alpha=alpha)
    for i in range(len(ax)): ax[i].legend()
    plt.xlabel('Time [s]')

if __name__=='__main__':
    import arc.utils.plot_utils as plut
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=1, linewidth=200, suppress=True)

    f, ax = plt.subplots(4,1,sharex=True)
    run_open_loop(ax, dt=1e-3, method='time-stepping')
    run_open_loop(ax, dt=1e-3, method='standard')
    plt.show()
    