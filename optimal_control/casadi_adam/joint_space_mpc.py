import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from time import sleep
from termcolor import colored

import orc.utils.plot_utils as plut
from example_robot_data.robots_loader import load
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_wrapper import RobotWrapper
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.mujoco_simulator import MujocoSimulator

print("Load robot model")
robot = load("ur5")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
nq = len(joints_name_list)  # number of joints
nx = 2*nq # size of the state variable
kinDyn = KinDynComputations(robot.urdf, joints_name_list)

ADD_SPHERE = 0
SPHERE_POS = np.array([0.2, -0.10, 0.5])
SPHERE_SIZE = np.ones(3)*0.1
SPHERE_RGBA = np.array([1, 0, 0, 1.])

# WITH THIS CONFIGURATION THE SOLVER ENDS UP VIOLATING THE JOINT LIMITS
# ADDING THE TERMINAL CONSTRAINT FIXES EVERYTHING!
# BUT SO DOES:
# - DECREASING THE POSITION WEIGHT IN THE COST
# - INCREASING THE ACCELERATION WEIGHT IN THE COST
# - INCREASING THE MAX NUMBER OF ITERATIONS OF THE SOLVER
DO_WARM_START = True
SOLVER_TOLERANCE = 1e-4
SOLVER_MAX_ITER = 3

DO_IDEAL_SIMULATION = False
POS_BOUNDS_SCALING_FACTOR = 0.2
VEL_BOUNDS_SCALING_FACTOR = 2.0
qMin = POS_BOUNDS_SCALING_FACTOR * robot.model.lowerPositionLimit
qMax = POS_BOUNDS_SCALING_FACTOR * robot.model.upperPositionLimit
vMax = VEL_BOUNDS_SCALING_FACTOR * robot.model.velocityLimit
dt_sim = 0.002
N_sim = 100
q0 = np.zeros(nq)  # initial joint configuration
dq0= np.zeros(nq)  # initial joint velocities

dt = 0.010 # time step MPC
N = 10  # time horizon MPC
# q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
q_des = q0.copy()
J = 1
q_des[J] = qMin[J] - 0.0*(qMax[J] - qMin[J])
w_p = 1e2   # position weight
w_v = 0e-6  # velocity weight
w_a = 1e-5  # acceleration weight
w_final_v = 0e0 # final velocity cost weight
USE_TERMINAL_CONSTRAINT = 0


print("Creating simulator...")
simu = MujocoSimulator("ur5", dt_sim)

print("Create optimization parameters")
''' The parameters P contain:
    - the initial state (first 12 values)
    - the target configuration (last 6 values)
'''
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

if(USE_TERMINAL_CONSTRAINT):
    opti.subject_to(X[-1][nq:] == 0.0)

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
if(ADD_SPHERE):
    simu.add_sphere(pos=SPHERE_POS, size=SPHERE_SIZE, rgba=SPHERE_RGBA)
simu.set_state(q0, dq0)

# Solve the problem to convergence the first time
opti.set_value(param_q_des, q_des)
opti.set_value(param_x_init, np.concatenate([simu.data.qpos, simu.data.qvel]))
sol = opti.solve()
opts["ipopt.max_iter"] = SOLVER_MAX_ITER
opti.solver("ipopt", opts)

print("Start the MPC loop")
for i in range(N_sim):
    start_time = clock()

    if(DO_WARM_START):
        # use current solution as initial guess for next problem
        for t in range(N):
            opti.set_initial(X[t], sol.value(X[t+1]))
        for t in range(N-1):
            opti.set_initial(U[t], sol.value(U[t+1]))
        opti.set_initial(X[N], sol.value(X[N]))
        opti.set_initial(U[N-1], sol.value(U[N-1]))
        # initialize dual variables
        lam_g0 = sol.value(opti.lam_g)
        opti.set_initial(opti.lam_g, lam_g0)
    
    print("Time step", i, "State", simu.data.qpos, simu.data.qvel)
    opti.set_value(param_x_init, np.concatenate([simu.data.qpos, simu.data.qvel]))
    try:
        sol = opti.solve()
    except:
        sol = opti.debug
        # print("Convergence failed!")
    end_time = clock()

    print("Comput. time: %.3f s"%(end_time-start_time), 
          "Iters: %3d"%sol.stats()['iter_count'], 
          "Tracking err: %.3f"%np.linalg.norm(q_des-simu.data.qpos))
    
    if(DO_IDEAL_SIMULATION):
        # use state predicted by the MPC as next state
        x_next = sol.value(X[1])
        simu.set_state(x_next[:nq], x_next[nq:])
        simu.update_viewer()
    else:
        # do a proper simulation
        tau = inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
        simu.step(tau, dt)
    
    if( np.any(simu.data.qpos > qMax)):
        print(colored("\nUPPER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(simu.data.qpos>qMax)[0])
    if( np.any(simu.data.qpos < qMin)):
        print(colored("\nLOWER POSITION LIMIT VIOLATED ON JOINTS", "red"), np.where(simu.data.qpos<qMin)[0])