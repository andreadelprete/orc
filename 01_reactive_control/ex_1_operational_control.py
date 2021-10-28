import numpy as np
from numpy import nan
from numpy.linalg import inv
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut
import time
from orc.utils.robot_loaders import loadUR
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
import ex_1_conf as conf
import solutions.ex_1_solution as solution

print("".center(conf.LINE_WIDTH,'#'))
print(" Cartesian Space Control - Manipulator ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_JOINT_POS = 0
PLOT_JOINT_VEL = 0
PLOT_JOINT_ACC = 0
PLOT_TORQUES = 0
PLOT_EE_POS = 1

r = loadUR()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot)

# get the ID corresponding to the frame we want to control
assert(robot.model.existFrame(conf.frame_name))
frame_id = robot.model.getFrameId(conf.frame_name)

nx, ndx = 3, 3
N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau     = np.empty((robot.na, N))*nan    # joint torques
tau_c   = np.empty((robot.na, N))*nan    # joint Coulomb torques
q       = np.empty((robot.nq, N+1))*nan  # joint angles
v       = np.empty((robot.nv, N+1))*nan  # joint velocities
dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
x       = np.empty((nx,  N))*nan        # end-effector position
dx      = np.empty((ndx, N))*nan        # end-effector velocity
ddx     = np.empty((ndx, N))*nan        # end effector acceleration
x_ref   = np.empty((nx,  N))*nan        # end-effector reference position
dx_ref  = np.empty((ndx, N))*nan        # end-effector reference velocity
ddx_ref = np.empty((ndx, N))*nan        # end-effector reference acceleration
ddx_des = np.empty((ndx, N))*nan        # end-effector desired acceleration

two_pi_f             = 2*np.pi*conf.freq   # frequency (time 2 PI)
two_pi_f_amp         = two_pi_f * conf.amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp

t = 0.0
kp, kd = conf.kp, conf.kd
PRINT_N = int(conf.PRINT_T/conf.dt)

for i in range(0, N):
    time_start = time.time()
    
    # set reference trajectory
    x_ref[:,i]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
    dx_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
    ddx_ref[:,i] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
    
    # read current state from simulator
    v[:,i] = simu.v
    q[:,i] = simu.q
    
    # compute mass matrix M, bias terms h, gravity terms g
    robot.computeAllTerms(q[:,i], v[:,i])
    M = robot.mass(q[:,i], False)
    h = robot.nle(q[:,i], v[:,i], False)
    g = robot.gravity(q[:,i])
    
    J6 = robot.frameJacobian(q[:,i], frame_id, False)
    J = J6[:3,:]            # take first 3 rows of J6
    H = robot.framePlacement(q[:,i], frame_id, False)
    x[:,i] = H.translation # take the 3d position of the end-effector
    v_frame = robot.frameVelocity(q[:,i], v[:,i], frame_id, False)
    dx[:,i] = v_frame.linear # take linear part of 6d velocity
#    dx[:,i] = J @ v[:,i]
    dJdq = robot.frameAcceleration(q[:,i], v[:,i], None, frame_id, False).linear
    
    # implement your control law here
    ddx_des[:,i] = ddx_ref[:,i] + kp*(x_ref[:,i] - x[:,i]) + kd*(dx_ref[:,i] - dx[:,i])
    tau[:,i] = solution.operational_motion_control(q[:,i], v[:,i], ddx_des[:,i], h, M, J, dJdq, conf)
        
    # send joint torques to simulator
    simu.simulate(tau[:,i], conf.dt, conf.ndt)
    tau_c[:,i] = simu.tau_c
        
    if i%PRINT_N == 0:
        print("Time %.3f"%(t))
    t += conf.dt
        
    time_spent = time.time() - time_start
    if(conf.simulate_real_time and time_spent < conf.dt): 
        time.sleep(conf.dt-time_spent)

# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)

if(PLOT_EE_POS):    
    (f, ax) = plut.create_empty_figure(nx)
    ax = ax.reshape(nx)
    for i in range(nx):
        ax[i].plot(time, x[i,:], label='x')
        ax[i].plot(time, x_ref[i,:], '--', label='x ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'x_'+str(i)+' [m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
    
if(PLOT_JOINT_POS):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, q[i,:-1], label='q')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_VEL):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, v[i,:-1], label='v')
#        ax[i].plot(time, v_ref[i,:], '--', label='v ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'v_'+str(i)+' [rad/s]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_ACC):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, dv[i,:-1], label=r'$\dot{v}$')
#        ax[i].plot(time, dv_ref[i,:], '--', label=r'$\dot{v}$ ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\dot{v}_'+str(i)+'$ [rad/s^2]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
   
if(PLOT_TORQUES):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, tau[i,:], label=r'$\tau$ '+str(i))
        ax[i].plot(time, tau_c[i,:], label=r'$\tau_c$ '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('Torque [Nm]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
plt.show()
