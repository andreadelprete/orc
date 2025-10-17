'''
COLLOCATION FORMULATION
JOINT SPACE TASK
MODEL PREDICTIVE CONTROL
'''
import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from time import sleep
from termcolor import colored

import orc.utils.plot_utils as plut
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_loaders import loadUR
from orc.utils.robot_wrapper import RobotWrapper

print("Load robot model")
robot = load("ur5")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
nq = len(joints_name_list)  # number of joints
nx = 2*nq # size of the state variable
kinDyn = KinDynComputations(robot.urdf, joints_name_list)

# Add a sphere in the simulator (only possible with Mujoco)
ADD_SPHERE = 0
SPHERE_POS = np.array([0.2, -0.10, 0.5])
SPHERE_SIZE = np.ones(3)*0.1
SPHERE_RGBA = np.array([1, 0, 0, 1.])

DO_WARM_START = True
SOLVER_TOLERANCE = 1e-4
SOLVER_MAX_ITER = 3

SIMULATOR = "mujoco" #"mujoco" or "pinocchio" or "ideal"
POS_BOUNDS_SCALING_FACTOR = 0.2
VEL_BOUNDS_SCALING_FACTOR = 2.0
qMin = POS_BOUNDS_SCALING_FACTOR * robot.model.lowerPositionLimit
qMax = POS_BOUNDS_SCALING_FACTOR * robot.model.upperPositionLimit
vMax = VEL_BOUNDS_SCALING_FACTOR * robot.model.velocityLimit
dt_sim = 0.002      # simulation time step
N_sim = 100
q0 = np.zeros(nq)  # initial joint configuration
dq0= np.zeros(nq)  # initial joint velocities

dt = 0.010      # MPC time step
N = 6           # MPC time horizon
# q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
q_des = q0.copy()
J = 1
q_des[J] = qMin[J] + 0.01*(qMax[J] - qMin[J])
w_p = 1e2   # position weight
w_v = 0e-6  # velocity weight
w_a = 1e-5  # acceleration weight
w_final_v = 0e0 # final velocity cost weight

if(SIMULATOR=="mujoco"):
    from orc.utils.mujoco_simulator import MujocoSimulator
    print("Creating simulator...")
    simu = MujocoSimulator("ur5", dt_sim)
    simu.set_state(q0, dq0)
else:
    r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    simu = RobotSimulator(conf_ur5, r)
    simu.init(q0, dq0)
    simu.display(q0)
    

print("Create optimization parameters")
opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_q_des = opti.parameter(nq)
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

# pre-compute state and torque bounds
lbx = qMin.tolist() + (-vMax).tolist()
ubx = qMax.tolist() + vMax.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()

# create all the decision variables
X, U = [], []
X += [opti.variable(nx)] # do not apply pos/vel bounds on initial state
for k in range(1, N+1): 
    X += [opti.variable(nx)]
    opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
for k in range(N): 
    U += [opti.variable(nq)]

print("Add initial conditions")
opti.subject_to(X[0] == param_x_init)
for k in range(N):     
    # print("Compute cost function")
    cost += w_p * (X[k][:nq] - param_q_des).T @ (X[k][:nq] - param_q_des)
    cost += w_v * X[k][nq:].T @ X[k][nq:]
    cost += w_a * U[k].T @ U[k]

    # print("Add dynamics constraints")
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # print("Add torque constraints")
    opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))

# add the final cost
cost += w_final_v * X[-1][nq:].T @ X[-1][nq:]

opti.minimize(cost)

print("Create the optimization problem")
opts = {
    "error_on_fail": False,
    "ipopt.print_level": 0,
    "ipopt.tol": SOLVER_TOLERANCE,
    "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
    "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
    "print_time": 0,                # print information about execution time
    "detect_simple_bounds": True,
    "ipopt.max_iter": 1000
}
opti.solver("ipopt", opts)

# set up simulation environment
if(SIMULATOR=="mujoco" and ADD_SPHERE):
    simu.add_sphere(pos=SPHERE_POS, size=SPHERE_SIZE, rgba=SPHERE_RGBA)

# Solve the problem to convergence the first time
x = np.concatenate([q0, dq0])
opti.set_value(param_q_des, q_des)
opti.set_value(param_x_init, x)
sol = opti.solve()

# set the maximum number of iterations to a small number
opts["ipopt.max_iter"] = SOLVER_MAX_ITER 
opti.solver("ipopt", opts)

print("Start the MPC loop")
for i in range(N_sim):

    # implement here the MPC loop with warm-start
    
    tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
    if(SIMULATOR=="mujoco"):
        # do a proper simulation with Mujoco
        simu.step(tau, dt)
        x = np.concatenate([simu.data.qpos, simu.data.qvel])
    elif(SIMULATOR=="pinocchio"):
        # do a proper simulation with Pinocchio
        simu.simulate(tau, dt, int(dt/dt_sim))
        x = np.concatenate([simu.q, simu.v])
    elif(SIMULATOR=="ideal"):
        # use state predicted by the MPC as next state
        x = sol.value(X[1])
        simu.display(x[:nq])
        
    
    if( np.any(x[:nq] > qMax)):
        print(colored("\nUPPER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]>qMax)[0])
    if( np.any(x[:nq] < qMin)):
        print(colored("\nLOWER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(x[:nq]<qMin)[0])