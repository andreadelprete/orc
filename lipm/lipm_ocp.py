import numpy as np
import casadi
from numpy import cosh, sinh

class LipmOcp:

    def __init__(self, dt, N, com_height):
        self.dt = dt
        self.N = N
        self.w = np.sqrt(9.81/com_height)

    def solve(self, x_init, wc, wdc, wu, c_ref, dc_ref, u_ref):
        self.opti = casadi.Opti()
        N = self.N
        self.x = self.opti.variable(2, N+1)
        self.u = self.opti.variable(N)
        x = self.x
        u = self.u
        w = self.w

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] =  wc*(x[0,i]-c_ref[i])**2 + \
                                    wdc*(x[1,i]-dc_ref[i])**2
            if(i<N):
                self.running_costs[i] += wu*(u[i] - u_ref[i])**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        ch = cosh(w*self.dt)
        sh = sinh(w*self.dt)
        for i in range(N):
            self.opti.subject_to( x[0,i+1] == ch*x[0,i] + sh*x[1,i]/w + (1-ch)*u[i] )
            self.opti.subject_to( x[1,i+1] == w*sh*x[0,i] + ch*x[1,i] - w*sh*u[i])
            # self.opti.subject_to( self.opti.bounded(self.u_min, u[i], self.u_max) )
        self.opti.subject_to(x[:,0]==x_init)

        self.opti.subject_to(x[0,N]==u_ref[N-1])    # final CoM position must be on CoP ref
        self.opti.subject_to(x[1,N]==0.0)           # final CoM velocity must be null

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from reference_trajectories import manual_foot_placement, create_CoP_trajectory
    import orc.utils.plot_utils

    # cost weights in the objective function:
    wu        = 1e-1   # CoP error squared cost weight
    wc        = 0      # CoM position error squared cost weight
    wdc       = 1e-3   # CoM velocity error squared cost weight

    # Inverted pendulum parameters:
    h           = 0.80   # fixed CoM height (assuming walking on a flat terrain)
    foot_length = 0.20   # foot size in the x-direction
    foot_width  = 0.10   # foot size in the y-direciton

    # MPC Parameters:
    dt              = 0.1                         # sampling time interval
    step_time       = 0.8                         # time needed for every step
    N_1_step        = int(round(step_time/dt))    # number of time steps for a footstep

    # walking parameters:
    step_length     = 0.21                  # fixed step length in the xz-plane
    N_steps         = 10                     # number of desired walking steps
    N               = N_steps * N_1_step    # number of desired walking intervals

    # CoM initial state:
    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([-0.09, 0.0])
    foot_step_0   = np.array([0.0, -0.09])   # initial foot step position in x-y

    # compute Com/CoP reference trajectories:
    # step_width = 2*np.absolute(y_0[0])
    desired_footsteps  = manual_foot_placement(foot_step_0, step_length, N_steps)
    U_ref = create_CoP_trajectory(N_steps, desired_footsteps, N, N_1_step)
    C_ref = np.zeros(N+1) # not used
    DC_ref = np.tile(step_length/step_time, N+1)

    ocp = LipmOcp(dt, N, h)
    sol_x = ocp.solve(x_0, wc, wdc, wu, C_ref, DC_ref, U_ref[:,0])

    plt.figure()
    plt.plot(sol_x.value(ocp.x[0,:]), label="CoM X pos")
    plt.plot(sol_x.value(ocp.x[1,:]), label="CoM X vel")
    plt.plot(sol_x.value(ocp.u),      label="CoP X")
    plt.plot(U_ref[:,0], ':', label="foot steps X")
    plt.legend()

    sol_y = ocp.solve(y_0, wc, wdc, wu, C_ref, 0*DC_ref, U_ref[:,1])
    plt.figure()
    plt.plot(sol_y.value(ocp.x[0,:]), label="CoM Y pos")
    plt.plot(sol_y.value(ocp.x[1,:]), label="CoM Y vel")
    plt.plot(sol_y.value(ocp.u),      label="CoP Y")
    plt.plot(U_ref[:,1], ':', label="foot steps Y")
    plt.legend()

    plt.show()
