import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
from time import sleep
import orc.utils.plot_utils as plut

time_start = clock()
nu = 2
nx = 3
DO_PLOTS = 1
dt = 0.02           # time step
N = 150             # time horizon
x0 = np.zeros(nx)   # initial state
x_des = np.array([2, -3, np.pi/2]) # desired state
x_obst = np.array([1, -1])  # center of the circular obstacle
ray_obst = 0.5              # ray of the circular obstacle
ray_robot = 0.2             # ray of the robot
w_u = 1e-4      # control weight
lbu = -2*np.ones(nu)
ubu = 2*np.ones(nu)


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
s   = cs.SX.sym('s', nx) # s = [x, y, theta]
u   = cs.SX.sym('dq', nu) # u = [v, w]
rhs    = cs.vertcat(cs.cos(s[-1])*u[0], 
                    cs.sin(s[-1])*u[0], 
                    u[1])
f = cs.Function('f', [s, u], [rhs])


# create all the decision variables
X, U = [], []
for k in range(N+1): 
    X += [opti.variable(nx)]
for k in range(N): 
    U += [opti.variable(nu)]
    opti.subject_to( opti.bounded(lbu, U[-1], ubu) )

print("Add initial conditions")
opti.subject_to(X[0] == param_x_init)
for k in range(N):     
    cost += w_u * U[k].T @ U[k]
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
    dist_vec = X[k][:2] - x_obst
    opti.subject_to( dist_vec.T @ dist_vec >= (ray_robot+ray_obst)**2)

cost += (X[-1] - param_x_des).T @ (X[-1] - param_x_des)
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
u_sol = np.array([sol.value(U[k]) for k in range(N)]).T

print("Total script time:", clock()-time_start)

print("x desired:  ", x_des)
print("x final:    ", x_sol[:,-1])

if(DO_PLOTS):
    time = np.arange(0, (N+1)*dt, dt)
    
    plt.figure(figsize=(10, 6))
    for i in range(x_sol.shape[0]):
        plt.plot(time, x_sol[i,:].T, label=f'x {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    for i in range(u_sol.shape[0]):
        plt.plot(time[:-1], u_sol[i,:].T, label=f'u {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Control')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.gca().add_patch(plt.Circle((x_sol[0,i], x_sol[1,i]), 
                                       ray_robot, color='k', fill=0))
    plt.plot(x_sol[0,:], x_sol[1,:].T, 'x-', label='x', alpha=0.7)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid(True)
    circle = plt.Circle((x_obst[0], x_obst[1]), ray_obst, color='r', fill=1)
    plt.gca().add_patch(circle)
    plt.gca().axis('equal')
    plt.show()