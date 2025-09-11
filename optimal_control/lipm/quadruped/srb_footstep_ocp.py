import numpy as np
import casadi
from numpy import cosh, sinh

class SrbFootstepOcp:

    def __init__(self, dt, N, com_height):
        self.dt = dt
        self.N = N
        self.w = np.sqrt(9.81/com_height)


    def solve(self, x_init, p_init, wc, wdc, wu, wp, c_ref, dc_ref, hip_pos, gait_pattern):
        # p_init = [px0, py0, px1, py1] = initial position of the two feet on the ground
        # hip_pos: dictionary (with keys "FL", "FR", "RL", "RR") containing XY pos of the hips w.r.t. the CoM
        # gait_pattern: list containing names of support feet for every time step

        self.opti = casadi.Opti()
        N = self.N
        self.x = self.opti.variable(4, N+1) # CoM XY pos + CoM XY vel
        # Bound u in [0, 1] and then compute CoP as:
        #   CoP = xy_hind_foot + u*(xy_front_foot - xy_hind_foot)
        self.u = self.opti.variable(N) 
        
        # compute how many footsteps we have in the horizon N
        n_footsteps = 1
        for i in range(1, N+1):
            if(gait_pattern[i][0] != gait_pattern[i-1][0]):
                # if the support feet have changed, add new foot positions
                n_footsteps += 1
        self.p = self.opti.variable(4, n_footsteps) # xy hind foot, xy front foot
        x, u, p, w = self.x, self.u, self.p, self.w

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        j = 0 # index used for the support feet
        ch, sh = cosh(w*self.dt), sinh(w*self.dt)
        for i in range(N+1):
            if(i>0 and gait_pattern[i][0] != gait_pattern[i-1][0]):
                # if the support feet have changed, switch to the new foot positions
                j += 1
            self.running_costs[i] =  wc*(x[0:2,i]- c_ref[:,i]).T @ (x[0:2,i]- c_ref[:,i]) + \
                                    wdc*(x[2:4,i]-dc_ref[:,i]).T @ (x[2:4,i]-dc_ref[:,i])
            hip_pos_0 = x[:2,i] + hip_pos[gait_pattern[i][0]]
            hip_pos_1 = x[:2,i] + hip_pos[gait_pattern[i][1]]
            # print("Time step", i, "Foot step", j, "Gait", gait_pattern[i], hip_pos_0, hip_pos_1)
            self.running_costs[i] += wp*(p[:2,j] - hip_pos_0).T @ (p[:2,j] - hip_pos_0)
            self.running_costs[i] += wp*(p[2:,j] - hip_pos_1).T @ (p[2:,j] - hip_pos_1)
            
            if(i<N):
                self.running_costs[i] += wu*(u[i] - 0.5)**2
                xy_foot_0, xy_foot_1 = p[:2,j], p[2:,j]
                cop = xy_foot_0 + u[i]*(xy_foot_1 - xy_foot_0)
                self.opti.subject_to( x[0:2,i+1] == ch*x[0:2,i] + sh*x[2:4,i]/w + (1-ch)*cop ) # xy pos
                self.opti.subject_to( x[2:4,i+1] == w*sh*x[0:2,i] + ch*x[2:4,i] - w*sh*cop)    # xy vel
                self.opti.subject_to( self.opti.bounded(0.0, u[i], 1.0) )
            self.cost += self.running_costs[i]

        self.opti.minimize(self.cost)
        # initial conditions
        self.opti.subject_to(x[:,0]==x_init)
        self.opti.subject_to(p[:,0]==p_init) 
        # terminal equilibrium constraints
        self.opti.subject_to(x[0:2,N]==cop)  # final CoM position must be on CoP
        self.opti.subject_to(x[2:4,N]==0.0)  # final CoM velocity must be null

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import matplotlib.pyplot as plt
    # from reference_trajectories import manual_foot_placement, create_CoP_trajectory
    import orc.utils.plot_utils
    import aliengo_conf as conf

    DO_PLOTS = 1

    # MPC Parameters:
    nb_dt_per_step = int(round(conf.T_step / conf.dt_mpc))
    N = conf.nb_steps * nb_dt_per_step  # number of desired walking intervals

    # CoM initial state:
    x_0 = np.array([0.0, 0.0, 0.0, 0.0])
    p_0 = np.concatenate([conf.hip_pos["FL"], conf.hip_pos["RR"]])

    # compute Com reference trajectories:
    C_ref = np.zeros((2,N+1)) # not used
    DC_ref = np.tile(conf.step_length/conf.T_step, N+1)
    DC_ref = np.vstack([DC_ref, np.zeros(N+1)])
    gait_pattern = []
    while(len(gait_pattern) < N+1):
        gait_pattern += nb_dt_per_step*[["FL", "RR"]]
        gait_pattern += nb_dt_per_step*[["FR", "RL"]]

    ocp = SrbFootstepOcp(conf.dt_mpc, N, conf.h)

    import time
    start = time.time()
    sol = ocp.solve(x_0, p_0, conf.wc, conf.wdc, conf.wu, conf.wp, C_ref, DC_ref, conf.hip_pos, gait_pattern)
    print("Computation time", time.time()-start)
    
    com_state, u, p = sol.value(ocp.x), sol.value(ocp.u), sol.value(ocp.p)
    foot_steps, cop, hip_pos_0, hip_pos_1 = np.zeros((4,N)), np.zeros((2,N)), np.zeros((2,N)), np.zeros((2,N))
    j = 0
    for i in range(N):
        if(i>0 and gait_pattern[i][0] != gait_pattern[i-1][0]):
            j += 1
        foot_steps[:,i] = sol.value(ocp.p[:,j])
        cop[:,i] = foot_steps[:2,i] + u[i] * (foot_steps[2:,i] - foot_steps[:2,i])
        hip_pos_0[:,i] = com_state[:2,i] + conf.hip_pos[gait_pattern[i][0]]
        hip_pos_1[:,i] = com_state[:2,i] + conf.hip_pos[gait_pattern[i][1]]
    
    if(DO_PLOTS):
        plt.figure()
        plt.plot(com_state[0,:],  label="CoM X pos")
        plt.plot(com_state[2,:],  label="CoM X vel")
        plt.plot(DC_ref[0,:],          label="Ref CoM Vel X")
        plt.plot(cop[0,:],        label="CoP X")
        plt.plot(p[0,:], ':',     label="foot steps 0 X")
        plt.plot(p[2,:], ':',     label="foot steps 1 X")
        plt.plot(hip_pos_0[0,:], ':', label="hip pos 0 X")
        plt.plot(hip_pos_1[0,:], ':', label="hip pos 1 X")
        plt.legend()

        plt.figure()
        plt.plot(com_state[1,:],  label="CoM Y pos")
        plt.plot(com_state[3,:],  label="CoM Y vel")
        plt.plot(cop[1,:],        label="CoP Y")
        plt.plot(p[1,:], ':',     label="foot steps 0 Y")
        plt.plot(p[3,:], ':',     label="foot steps 1 Y")
        plt.plot(hip_pos_0[1,:], ':', label="hip pos 0 Y")
        plt.plot(hip_pos_1[1,:], ':', label="hip pos 1 Y")
        plt.legend()

        plt.show()

    np.savez(
        conf.DATA_FILE_LIPM,
        com_ref=com_state,
        cop_ref=cop,
        p=p,                    # foot step variables as in the OCP formulation
        foot_steps=foot_steps,  # same as p, but with the same time step of the CoM
        gait_pattern=gait_pattern
    )


