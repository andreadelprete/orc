'''
SINGLE SHOOTING FORMULATION
JOINT SPACE TASK
'''

import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from time import time as clock
from time import sleep

import orc.utils.plot_utils as plut
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5

time_start = clock()
print("Load robot model")
robot = load("ur5")

print("Create KinDynComputations object")
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"
nq = len(joints_name_list)  # number of joints
nx = 2*nq # size of the state variable
kinDyn = KinDynComputations(robot.urdf, joints_name_list)


DO_PLOTS = True
dt = 0.01       # time step for optimal control problem
dt_sim = 0.002  # time step for the simulator
N = 60          # time horizon
q0 = np.zeros(nq)  # initial joint configuration
dq0= np.zeros(nq)  # initial joint velocities
q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
w_v = 1e-4  # velocity weight
w_a = 1e-6  # acceleration weight
w_final = 1e2 # final cost weight

print("Create optimization parameters")
''' The parameters P contain:
    - the initial state (first 12 values)
    - the target configuration (last 6 values)
'''
opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_q_des = opti.parameter(nq)

# CLASSIC MULTI-BODY DYNAMICS MODELING
# M(q) * ddq + h(q, dq) = tau <= INVERSE DYNAMICS (RNEA)
# u = tau
# x = (q, dq)
# dx = f(x, u) 
# ddq = M^-1 (u - h(x)) <= DIRECT DYNAMICS or FORWARD DYNAMICS (ABA)

# ALTERNATIVE MULTI-BODY DYNAMICS MODELING
# x = (q, dq)
# u = ddq
# dx = f(x,u) => Double integrator (Linear dynamical system)
# Torques are no longer a variable of the OCP, so the torque bounds become:
#           tau_min <= M(q) * u + h(q, dq) <= tau_max

# create the dynamics function
q   = cs.SX.sym('q', nq)
dq  = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs    = cs.vertcat(dq, ddq)
# The function f takes as inputs (x, u=ddq) and compute dx=(dq, ddq)
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
lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()

# create the decision variables, the cost and the constraints
X, U = [], []
for i in range(N):
    U.append(opti.variable(nq))

# Compute states as function of control inputs
X.append(param_x_init)
for i in range(N):
    x_next = X[i] + dt * f(X[i], U[i])
    X.append(x_next)

# q = X[:nq]
# dq = X[nq:]
cost = 0.0
for i in range(N):
    cost += w_v * X[i][nq:].T @ X[i][nq:] # ||dq||^2 = dq.T * dq
    cost += w_a * U[i].T @ U[i]           # ||ddq||^2 = ddq.T * ddq

    # joint position & velocity bounds
    opti.subject_to( opti.bounded(lbx, X[i], ubx) )

    # joint torque bounds
    tau = inv_dyn(X[i], U[i])
    opti.subject_to( opti.bounded(tau_min, tau, tau_max) )

# terminal cost: ||q-q_des||^2
cost += w_final * (X[N][:nq] - param_q_des).T @ (X[N][:nq] - param_q_des)
opti.subject_to( X[N][nq:] == 0.0 )
opti.minimize(cost)

print("Create the optimization problem")
opts = {
    "ipopt.print_level": 5,
    "ipopt.tol": 1e-6,
    "ipopt.constr_viol_tol": 1e-6,
    "ipopt.compl_inf_tol": 1e-6,
    "print_time": 0,                # print information about execution time
    "detect_simple_bounds": True
}
opti.solver("ipopt", opts)

print("Start solving the optimization problem")
opti.set_value(param_q_des, q_des)
opti.set_value(param_x_init, np.concatenate([q0, dq0]))
sol = opti.solve()
x_sol = np.array([sol.value(X[k]) for k in range(N+1)]).T
u_sol = np.array([sol.value(U[k]) for k in range(N)]).T
q_sol = x_sol[:nq,:]
dq_sol = x_sol[nq:,:]

print("Total script time:", clock()-time_start)

tau = np.zeros((nq, N))
for i in range(N):
    tau[:,i] = inv_dyn(x_sol[:,i], u_sol[:,i]).toarray().squeeze()
print("q desired:  ", q_des)
print("q final:    ", q_sol[:,N])
print("dq final:   ", dq_sol[:,N])


if(DO_PLOTS):
    # display sparsity patter of constraint Jacobian matrix
    J = sol.value(cs.jacobian(opti.g,opti.x))
    plt.figure()
    plt.spy(J)
    plt.title("Constraint Jacobian")

    # display sparsity patter of cost Hessian matrix
    plt.figure()
    H = sol.value(cs.hessian(opti.f, opti.x)[0])
    plt.spy(H)
    plt.title("Cost Hessian")

    plt.show()

# plot joint trajectories
if(DO_PLOTS):
    time = np.arange(0, (N+1)*dt, dt)
    plt.figure(figsize=(10, 6))
    for i in range(q_sol.shape[0]):
        plt.plot(time, q_sol[i,:].T, label=f'q {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint angle [rad]')
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
    for i in range(u_sol.shape[0]):
        plt.plot(time[:-1], u_sol[i,:].T, label=f'ddq {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint acceleration [rad/s2]')
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