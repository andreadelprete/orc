import numpy as np
import casadi
from numpy import cosh, sinh

class LipmOcp:

    def __init__(self, dt, N, com_height):
        self.dt = dt
        self.N = N
        self.w = np.sqrt(9.81/com_height)


    def solve(self, x_init, wc, wdc, wu, c_ref, dc_ref, u_ref, u_max):
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
        # A = casadi.numpy.array([[ch, sh/w],
        #                         [w*sh, ch]])
        # B = casadi.numpy.array([[1-ch],
        #                         [w*sh]])
        for i in range(N):
            self.opti.subject_to( x[0,i+1] == ch*x[0,i] + sh*x[1,i]/w + (1-ch)*u[i] )
            self.opti.subject_to( x[1,i+1] == w*sh*x[0,i] + ch*x[1,i] - w*sh*u[i])
            # self.opti.subject_to( x[:,i+1] == A @ x[:,i] + B * u[i])
            self.opti.subject_to( self.opti.bounded(u_ref[i]-u_max, u[i], u_ref[i]+u_max) )
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
    import romeo_conf as conf

    # MPC Parameters:
    foot_length = conf.lxn + conf.lxp  # foot size in the x-direction
    foot_width = conf.lyn + conf.lyp  # foot size in the y-direciton
    nb_dt_per_step = int(round(conf.T_step / conf.dt_mpc))
    N = conf.nb_steps * nb_dt_per_step  # number of desired walking intervals

    # CoM initial state:
    x_0 = np.array([conf.foot_step_0[0], 0.0])
    y_0 = np.array([conf.foot_step_0[1], 0.0])

    # compute Com/CoP reference trajectories:
    foot_steps  = manual_foot_placement(conf.foot_step_0, conf.step_length, 
                                               conf.nb_steps)
    foot_steps[1:, 0] -= conf.step_length
    U_ref = create_CoP_trajectory(conf.nb_steps, foot_steps, N, nb_dt_per_step)
    C_ref = np.zeros(N+1) # not used
    DC_ref = np.tile(conf.step_length/conf.T_step, N+1)

    ocp = LipmOcp(conf.dt_mpc, N, conf.h)

    sol_x = ocp.solve(x_0, conf.wc, conf.wdc, conf.wu, C_ref, DC_ref, U_ref[:,0], foot_length/2)
    com_state_x = sol_x.value(ocp.x)
    cop_x = sol_x.value(ocp.u)

    sol_y = ocp.solve(y_0, conf.wc, conf.wdc, conf.wu, C_ref, 0*DC_ref, U_ref[:,1], foot_width/2)
    com_state_y = sol_y.value(ocp.x)
    cop_y = sol_y.value(ocp.u)

    plt.figure()
    plt.plot(com_state_x[0,:],  label="CoM X pos")
    plt.plot(com_state_x[1,:],  label="CoM X vel")
    plt.plot(DC_ref,            label="Ref CoM Vel X")
    plt.plot(cop_x,             label="CoP X")
    plt.plot(U_ref[:,0], ':',   label="foot steps X")
    plt.legend()

    plt.figure()
    plt.plot(com_state_y[0,:],  label="CoM Y pos")
    plt.plot(com_state_y[1,:],  label="CoM Y vel")
    plt.plot(cop_y,             label="CoP Y")
    plt.plot(U_ref[:,1], ':',   label="foot steps Y")
    plt.legend()

    np.savez(
        conf.DATA_FILE_LIPM,
        com_state_x=com_state_x.T,
        com_state_y=com_state_y.T,
        cop_ref=U_ref,
        cop_x=cop_x,
        cop_y=cop_y,
        foot_steps=foot_steps,
    )

    plt.show()


