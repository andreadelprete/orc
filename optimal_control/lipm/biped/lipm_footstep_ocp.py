import numpy as np
import casadi
from numpy import cosh, sinh

class LipmFootstepOcp:

    def __init__(self, dt, N, com_height):
        self.dt = dt
        self.N = N
        self.w = np.sqrt(9.81/com_height)


    def solve(self, x_init, wc, wdc, wu, c_ref, dc_ref, nb_dt_per_step, 
              u_max, lateral=None, lat_footstep_offset=None):
        # nb_dt_per_step: number of time steps per footstep
        self.opti = casadi.Opti()
        N = self.N
        self.x = self.opti.variable(2, N+1)
        self.u = self.opti.variable(N)
        # compute how many footsteps we have in the horizon N
        n_footsteps = int(np.ceil(N/nb_dt_per_step))
        self.p = self.opti.variable(n_footsteps)
        x = self.x
        u = self.u
        p = self.p
        w = self.w

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] =  wc*(x[0,i]-c_ref[i])**2 + \
                                    wdc*(x[1,i]-dc_ref[i])**2
            if(i<N):
                self.running_costs[i] += wu*(u[i] - p[i//nb_dt_per_step])**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        if(lateral is not None):
            if(lateral=='right'):
                target = 1
            else:
                target = 0
            for i in range(1, n_footsteps):
                if(i%2==target):
                    # constraint for right foot
                    self.opti.subject_to( p[i] >= x[0,i*nb_dt_per_step] + lat_footstep_offset)
                else:
                    # constraint for left foot
                    self.opti.subject_to( p[i] <= x[0,i*nb_dt_per_step] - lat_footstep_offset)

        ch = cosh(w*self.dt)
        sh = sinh(w*self.dt)
        for i in range(N):
            self.opti.subject_to( x[0,i+1] == ch*x[0,i] + sh*x[1,i]/w + (1-ch)*u[i] )
            self.opti.subject_to( x[1,i+1] == w*sh*x[0,i] + ch*x[1,i] - w*sh*u[i])
            self.opti.subject_to( self.opti.bounded(p[i//nb_dt_per_step]-u_max, u[i], p[i//nb_dt_per_step]+u_max) )
        self.opti.subject_to(x[:,0]==x_init)
        # footstep at time zero on top of CoM pos
        self.opti.subject_to(p[0]==x_init[0]) 

        self.opti.subject_to(x[0,N]==p[n_footsteps-1])  # final CoM position must be on footstep
        self.opti.subject_to(x[1,N]==0.0)               # final CoM velocity must be null

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

    # compute Com reference trajectories:
    C_ref = np.zeros(N+1) # not used
    DC_ref = np.tile(conf.step_length/conf.T_step, N+1)

    ocp = LipmFootstepOcp(conf.dt_mpc, N, conf.h)

    sol_x = ocp.solve(x_0, conf.wc, conf.wdc, conf.wu, C_ref, 0*DC_ref, nb_dt_per_step, foot_length/2)
    com_state_x = sol_x.value(ocp.x)
    cop_x = sol_x.value(ocp.u)
    foot_steps_x = sol_x.value(ocp.p)
    p_x = [sol_x.value(ocp.p[i//nb_dt_per_step]) for i in range(N)]

    sol_y = ocp.solve(y_0, conf.wc, conf.wdc, conf.wu, C_ref, DC_ref, 
                      nb_dt_per_step, foot_width/2, "right", 0.1)
    com_state_y = sol_y.value(ocp.x)
    cop_y = sol_y.value(ocp.u)
    foot_steps_y = sol_y.value(ocp.p)
    p_y = [sol_y.value(ocp.p[i//nb_dt_per_step]) for i in range(N)]

    plt.figure()
    plt.plot(com_state_x[0,:],  label="CoM X pos")
    plt.plot(com_state_x[1,:],  label="CoM X vel")
    plt.plot(DC_ref,            label="Ref CoM Vel X")
    plt.plot(cop_x,             label="CoP X")
    plt.plot(p_x, ':',   label="foot steps X")
    plt.legend()

    plt.figure()
    plt.plot(com_state_y[0,:],  label="CoM Y pos")
    plt.plot(com_state_y[1,:],  label="CoM Y vel")
    plt.plot(cop_y,             label="CoP Y")
    plt.plot(p_y, ':',   label="foot steps Y")
    plt.legend()

    U_ref = np.vstack([p_x, p_y]).T
    foot_steps = np.vstack([foot_steps_x, foot_steps_y]).T

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


