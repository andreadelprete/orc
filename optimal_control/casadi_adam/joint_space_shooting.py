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


USE_MUJOCO_SIMULATOR = 1
DO_PLOTS = 0
dt = 0.01 # time step
dt_sim = 0.002
N = 60  # time horizon
q0 = np.zeros(nq)  # initial joint configuration
dq0= np.zeros(nq)  # initial joint velocities
q_des = np.array([0, -1.57, 0, 0, 0, 0]) # desired joint configuration
w_v = 1e-4  # velocity weight
w_a = 0e-6  # acceleration weight
w_final = 1e2 # final cost weight

if(USE_MUJOCO_SIMULATOR):
    from orc.utils.mujoco_simulator import MujocoSimulator
    print("Creating robot simulator...")
    simu = MujocoSimulator("ur5", dt_sim)
    print("Simulator created")

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
lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()

# create all the decision variables
X, U = [], []
for k in range(N): 
    U += [opti.variable(nq)]

print("Compute state trajectory as function of control inputs and initial state")
X.append(param_x_init)
for k in range(N):
    X.append(X[k] + dt * f(X[k], U[k]))
    opti.subject_to( opti.bounded(lbx, X[-1], ubx) )

for k in range(N):     
    # print("Compute cost function")
    cost += (X[k][:nq] - param_q_des).T @ (X[k][:nq] - param_q_des)
    cost += w_v * X[k][nq:].T @ X[k][nq:]
    cost += w_a * U[k].T @ U[k]

    # print("Add torque constraints")
    opti.subject_to( opti.bounded(tau_min, inv_dyn(X[k], U[k]), tau_max))

# add the final cost
cost += w_final * (X[-1][:nq] - param_q_des).T @ (X[-1][:nq] - param_q_des)
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


  
if(USE_MUJOCO_SIMULATOR):
    print("Display optimized motion")
    simu.display_motion(q_sol.T, dt)
    sleep(1)

    # closed-loop simulation
    print("Simulate robot to track optimized motion")
    kp = 10
    kd = np.sqrt(kp)
    def simulate():
        simu.set_state(q_sol[:,0], dq_sol[:,0])
        for i in range(N):
            simu.step(tau[:,i] + kp*(q_sol[:,i] - simu.data.qpos) + kd*(dq_sol[:,i] - simu.data.qvel), dt)
            sleep(dt)
    simulate()
    print("q final simu:", simu.data.qpos)

    simu.add_sphere(pos=np.array([0.0, 0.0, 0.5]), size=np.ones(3)*0.1, rgba=np.array([1, 0, 0, 1.]))
    print("Red sphere added. Going to simulate again")
    simulate()

    simu.add_sphere(pos=np.array([0.5, 0.0, 0.5]), size=np.ones(3)*0.1, rgba=np.array([0, 0, 1, 1.]))
    print("Blue sphere added. Going to simulate again")
    simulate()
else:
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
