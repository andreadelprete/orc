import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from time import sleep

import orc.utils.plot_utils as plut
from example_robot_data.robots_loader import load
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_wrapper import RobotWrapper
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.mujoco_simulator import MujocoSimulator

time_start = clock()
print("Load robot model")
robot = load("ur5")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
nq = len(joints_name_list)  # number of joints
nx = 2*nq # size of the state variable
kinDyn = KinDynComputations(robot.urdf, joints_name_list)

ADD_SPHERE = 1
SPHERE_POS = np.array([0.7, -0.7, 0.])
SPHERE_SIZE = np.ones(3)*0.2
SPHERE_RGBA = np.array([0, 1, 0, .5])
OBSTACLE_MIN_DISTANCE = 0.2

USE_MUJOCO_SIMULATOR = 1
DO_PLOTS = 0
dt = 0.01           # OCP time step
dt_sim = 0.002      # simulator time step
N = 60              # OCP time horizon
q0 = np.zeros(nq)   # initial joint configuration
dq0= np.zeros(nq)   # initial joint velocities
frame_name = "ee_link"
if(frame_name not in kinDyn.rbdalgos.model.links.keys()):
    print("ERROR. Frame name can only take values from this list")
ee_des = np.array([0, -0.75, 0]) # desired end-effector position
w_v = 1e-4          # velocity weight
w_a = 1e-6          # acceleration weight
w_final = 1e4       # final cost weight

# JOINT FEEDBACK GAINS USED FOR THE SIMULATION
kp = 10
kd = np.sqrt(kp)

if(USE_MUJOCO_SIMULATOR):
    print("Creating robot simulator...")
    simu = MujocoSimulator("ur5", dt_sim)
    simu.add_visual_sphere("ee_target", ee_des, 0.05, np.array([1, 0, 0, 0.5]))
    simu.add_visual_sphere("ee_pos", np.zeros(3), 0.05, np.array([0, 0, 1, 0.5]))

    if(ADD_SPHERE):
        simu.add_sphere(SPHERE_POS, SPHERE_SIZE, SPHERE_RGBA)
    
    print("Simulator created")

print("Create optimization parameters")
''' The parameters P contain:
    - the initial state (first 12 values)
    - the target configuration (last 6 values)
'''
opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_ee_des = opti.parameter(3)
cost = 0

# create the dynamics function
q   = cs.SX.sym('q', nq)
dq  = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs    = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# create a Casadi inverse dynamics function
H_b = cs.SX.eye(4)     # base configuration
v_b = cs.SX.zeros(6)   # base velocity
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
# discard the first 6 elements because they are associated to the robot base
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:,6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# create a Casadi forward kinematics function
fk_fun = kinDyn.forward_kinematics_fun(frame_name)
ee_pos = fk_fun(H_b, q)[:3, 3]
fk = cs.Function('fk', [q], [ee_pos])

# pre-compute state and torque bounds
lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()

# create all the decision variables
X, U = [], []
for k in range(N+1): 
    X += [opti.variable(nx)]
    opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
for k in range(N): 
    U += [opti.variable(nq)]

print("Add initial conditions")
opti.subject_to(X[0] == param_x_init)
for k in range(N):     
    # print("Compute cost function")
    ee_pos = fk(X[k][:nq])
    cost += (ee_pos - param_ee_des).T @ (ee_pos - param_ee_des)
    cost += w_v * X[k][nq:].T @ X[k][nq:]
    cost += w_a * U[k].T @ U[k]

    # print("Add dynamics constraints")
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # print("Add torque constraints")
    opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))

    # add collision avoidance constraint
    if(ADD_SPHERE):
        opti.subject_to( cs.norm_2(fk(X[k][:nq]) - SPHERE_POS) >= SPHERE_SIZE[0]+OBSTACLE_MIN_DISTANCE )

# add the final cost
ee_pos = fk(X[-1][:nq])
cost += w_final * (ee_pos - param_ee_des).T @ (ee_pos - param_ee_des)
cost += w_final * X[-1][nq:].T @ X[-1][nq:]
opti.minimize(cost)

print("Create the optimization problem")
opts = {
    "ipopt.print_level": 0,
    "ipopt.tol": 1e-6,
    "ipopt.constr_viol_tol": 1e-6,
    "ipopt.compl_inf_tol": 1e-6,
    "print_time": 0,                # print information about execution time
    "detect_simple_bounds": True
}
opti.solver("ipopt", opts)

print("Start solving the optimization problem")
opti.set_value(param_ee_des, ee_des)
opti.set_value(param_x_init, np.concatenate([q0, dq0]))
sol = opti.solve()
x_sol = np.array([sol.value(X[k]) for k in range(N+1)]).T
u_sol = np.array([sol.value(U[k]) for k in range(N)]).T
q_sol = x_sol[:nq,:]
dq_sol = x_sol[nq:,:]

print("Total script time:", clock()-time_start)

# compute joint torques trajectory
tau = np.zeros((nq, N))
for i in range(N):
    tau[:,i] = inv_dyn(x_sol[:,i], u_sol[:,i]).toarray().squeeze()

# compute end-effector trajectory
ee = np.zeros((3, N+1))
for i in range(N+1):
    ee[:,i] = fk(x_sol[:nq,i]).toarray().squeeze()

print("ee desired:  ", ee_des)
print("ee final:    ", ee[:,-1])
print("dq final:    ", dq_sol[:,-1])


  
if(USE_MUJOCO_SIMULATOR):
    print("Display optimized motion")
    simu.add_visual_trajectory("ee-traj", ee, 10, np.array((1,1,0,1)))
    # simu.display_motion(q_sol.T, dt)
    def display_motion(dt):
        for i in range(N):
            simu.display(q_sol[:,i])
            simu.move_visual_sphere("ee_pos", fk(q_sol[:,i]).toarray().squeeze())
            sleep(dt)
    display_motion(dt)
    sleep(1)
    print("To display the optimized motion again run: display_motion(dt)")

    # closed-loop simulation
    print("Simulate robot to track optimized motion")
    def simulate():
        simu.set_state(q_sol[:,0], dq_sol[:,0])
        for i in range(N):
            simu.step(tau[:,i] + kp*(q_sol[:,i] - simu.data.qpos) + kd*(dq_sol[:,i] - simu.data.qvel), dt)
            simu.move_visual_sphere("ee_pos", fk(simu.data.qpos).toarray().squeeze())
            sleep(dt)
    simulate()
    print("To simulate the robot tracking again run: simulate()")
    print("x final simu:", fk(simu.data.qpos))

    

# # display sparsity patter of constraint Jacobian matrix
# # sol.value(cs.jacobian(opti.g,opti.x))

# plot joint trajectories
if(DO_PLOTS):
    time = np.arange(0, (N+1)*dt, dt)

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot([time[0], time[-1]], [ee_des[i], ee_des[i]], ':', label=f'EE des {i}', alpha=0.7)
        plt.plot(time, ee[i,:].T, label=f'EE {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('End-effector pos [m]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    for i in range(dq_sol.shape[0]):
        plt.plot(time, dq_sol[i,:].T, label=f'dq {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocity [rad/s]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    for i in range(tau.shape[0]):
        plt.plot(time[:-1], tau[i,:].T, label=f'tau {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torque [Nm]')
    plt.legend()
    plt.grid(True)
    plt.show()