'''
A differential drive mobile robot that has to reach a given state in a fixed time 
while avoiding a set of obstacles. Obstacles can have polygonal or circular shapes.
The robot shape is a circle, and its actuators have limits.
'''

import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
from time import sleep

import orc.utils.plot_utils as plut
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5

time_start = clock()
nu = 2
nx = 3
DO_PLOTS = 1
dt = 0.1           # time step
N = 40             # time horizon
x0 = np.zeros(nx)   # initial state
ray_robot = 0.3     # ray of the robot
x_des = np.array([2, -3, np.pi/2]) # desired state
w_u = 1e-5      # control weight
w_s = 1e0      # safety margin weight
lbu = -2*np.ones(nu)
ubu = 2*np.ones(nu)

POLY_OBS = []
# list of vertices of the polygonal obstacle
obst_vert1 = []
x_off = 0.5
obst_vert1.append(np.array([1.0-x_off, -0.5]))
obst_vert1.append(np.array([1.5-x_off, -1.0]))
obst_vert1.append(np.array([1.0-x_off, -1.5]))
obst_vert1.append(np.array([0.0-x_off, -1.5]))
obst_vert1.append(np.array([0.0-x_off, -0.5]))
POLY_OBS.append(obst_vert1)
obst_vert2 = []
obst_vert2.append(np.array([3.5, -1.0]))
obst_vert2.append(np.array([3.5, -1.5]))
obst_vert2.append(np.array([2.5, -1.5]))
obst_vert2.append(np.array([2.5, -1.0]))
POLY_OBS.append(obst_vert2)

CIRC_OBS = [ [np.array([2.0-0.5, -2.0]), 0.3]]

def plot_map(show=True):
    plt.figure(figsize=(10, 6))
    plt.gca().add_patch(plt.Circle(x0, ray_robot, color='k', fill=0))
    plt.gca().add_patch(plt.Circle(x_des, ray_robot, color='g', fill=1, alpha=0.5))
    plt.grid(True)
    for obs in POLY_OBS:
        patch = plt.Polygon(np.array(obs), color='r')
        plt.gca().add_patch(patch)
    for obs in CIRC_OBS:
        plt.gca().add_patch(plt.Circle(obs[0], obs[1], color='r', fill=1))
    plt.gca().axis('equal')
    if(show): plt.show()

# plot_map()

print("Create optimization parameters")
''' The parameters P contain:
    - the initial state (first nx values)
    - the target configuration (last nx values)
'''
opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_x_des = opti.parameter(nx)
cost = 0

# create the dynamics function
s   = cs.SX.sym('s', nx)  # s = [x, y, theta]
u   = cs.SX.sym('u', nu) # u = [v, w]
rhs = cs.vertcat(cs.cos(s[-1])*u[0], 
                 cs.sin(s[-1])*u[0], 
                 u[1])
f   = cs.Function('f', [s, u], [rhs])

# create all the decision variables for state and control
X, U, S = [], [], []
for k in range(N+1): 
    X += [opti.variable(nx)]
    S += [opti.variable(1)]
    opti.subject_to( opti.bounded(0.0, S[-1], 1.0))
for k in range(N): 
    U += [opti.variable(nu)]
    opti.subject_to( opti.bounded(lbu, U[-1], ubu) )

print("Add initial conditions")
opti.subject_to(X[0] == param_x_init)
A, B = [], []
for k in range(N):     
    cost += w_u * U[k].T @ U[k] - w_s * S[k]
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # Create new variables to model collision avoidance with polygone.
    # Find a separating plane such that all the polygone vertices are
    # separated by the robot position
    #  a^T x => b   AND   a^T v_i <= b
    for obs in POLY_OBS:
        A += [opti.variable(2)]
        B += [opti.variable(1)]
        opti.subject_to(A[-1].T @ X[k][:2] 
                        >= 
                        B[-1] + 
                        (ray_robot + S[k]) * cs.sqrt(1e-10 + A[-1].T @ A[-1]))
        for v in obs:
            opti.subject_to( A[-1].T @ v <= B[-1])
    
    for obs in CIRC_OBS:
        dist_vec = X[k][:2] - obs[0]
        opti.subject_to( dist_vec.T @ dist_vec >= (ray_robot+S[k]+obs[1])**2)

# cost += (X[-1] - param_x_des).T @ (X[-1] - param_x_des)
opti.subject_to( X[-1] == param_x_des)
opti.minimize(cost)

print("Create the optimization problem")
opts = {
    "ipopt.print_level": 1,
    "ipopt.tol": 1e-6,
    "ipopt.constr_viol_tol": 1e-6,
    "ipopt.compl_inf_tol": 1e-6,
    "print_time": 0,                # print information about execution time
    "detect_simple_bounds": True
}
opti.solver("ipopt", opts)

print("Start solving the optimization problem")
opti.set_value(param_x_des, x_des)
opti.set_value(param_x_init, x0)
sol = opti.solve()
x_sol = np.array([sol.value(X[k]) for k in range(N+1)]).T
s_sol = np.array([sol.value(S[k]) for k in range(N+1)]).T
u_sol = np.array([sol.value(U[k]) for k in range(N)]).T

print("Total script time:", clock()-time_start)

print("x desired:  ", x_des)
print("x final:    ", x_sol[:,-1])
print("s mean:     ", np.mean(s_sol))
print("s min:      ", np.min(s_sol))

if(DO_PLOTS):
    time = np.arange(0, (N+1)*dt, dt)
    
    plt.figure(figsize=(10, 6))
    for i in range(x_sol.shape[0]):
        plt.plot(time, x_sol[i,:].T, label=f'x {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)

    plot_map(show=False)
    plt.plot(x_sol[0,:], x_sol[1,:].T, 'x-', label='x', alpha=0.7)
    for i in range(N+1):
        plt.gca().add_patch(plt.Circle((x_sol[0,i], x_sol[1,i]), 
                                       ray_robot, color='k', fill=0))
    plt.show()