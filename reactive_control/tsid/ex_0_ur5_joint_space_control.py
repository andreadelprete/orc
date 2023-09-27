import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut
from orc.utils.robot_loaders import loadUR, loadUR_urdf
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
import time
import tsid
import ex_0_ur5_conf as conf

print("".center(conf.LINE_WIDTH,'#'))
print(" Joint Space Inverse Dynamics - Manipulator ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_JOINT_POS = 1
PLOT_JOINT_VEL = 1
PLOT_JOINT_ACC = 1
PLOT_TORQUES = 1

urdf, path = loadUR_urdf()
robot = tsid.RobotWrapper(urdf, [path], False)
model = robot.model()

r = loadUR()
robot_simu = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot_simu)

formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
q0 = conf.q0
v0 = np.zeros(robot.nv)
formulation.computeProblemData(0.0, q0, v0)
        
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(conf.kp_posture * np.ones(robot.nv))
postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv))
formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)

trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q0)
postureTask.setReference(trajPosture.computeNext())

v_max = conf.v_max_scaling * model.velocityLimit
v_min = -v_max
jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, conf.dt)
jointBoundsTask.setVelocityBounds(v_min, v_max)
formulation.addMotionTask(jointBoundsTask, conf.w_joint_bounds, 0, 0.0)

solver = tsid.SolverHQuadProgFast("qp solver")
solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

N = conf.N_SIMULATION
tau    = np.empty((robot.na, N))*nan
q      = np.empty((robot.nq, N+1))*nan
v      = np.empty((robot.nv, N+1))*nan
dv     = np.empty((robot.nv, N+1))*nan
q_ref  = np.empty((robot.nq, N))*nan
v_ref  = np.empty((robot.nv, N))*nan
dv_ref = np.empty((robot.nv, N))*nan
dv_des = np.empty((robot.nv, N))*nan
samplePosture = trajPosture.computeNext()

amp                  = conf.amp        # amplitude
phi                  = conf.phase      # phase
two_pi_f             = conf.two_pi_f   # frequency (time 2 PI)
two_pi_f_amp         = two_pi_f * amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp

t = 0.0
dt = conf.dt
q[:,0], v[:,0] = q0, v0

for i in range(0, N):
    time_start = time.time()
    
    # set reference trajectory
    q_ref[:,i]  = q0 +  amp * np.sin(two_pi_f*t + phi)
    v_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + phi)
    dv_ref[:,i] = two_pi_f_squared_amp * -np.sin(two_pi_f*t + phi)
    samplePosture.value(q_ref[:,i])
    samplePosture.derivative(v_ref[:,i])
    samplePosture.second_derivative(dv_ref[:,i])
    postureTask.setReference(samplePosture)

    HQPData = formulation.computeProblemData(t, q[:,i], v[:,i])
    sol = solver.solve(HQPData)
    if(sol.status!=0):
        print("Time %.3f QP problem could not be solved! Error code:"%t, sol.status)
        break
    
    tau[:,i] = formulation.getActuatorForces(sol)
    dv[:,i] = formulation.getAccelerations(sol)
    dv_des[:,i] = postureTask.getDesiredAcceleration

    if i%conf.PRINT_N == 0:
        print("Time %.3f"%(t))
        print("\ttracking err %s: %.3f"%(postureTask.name.ljust(20,'.'), norm(postureTask.position_error, 2)))

    # send torque commands to simulator
    simu.simulate(tau[:,i], dt)
    q[:,i+1] = simu.q
    v[:,i+1] = simu.v
    t += dt
    
    if i%conf.DISPLAY_N == 0: 
        simu.display(q[:,i])

    time_spent = time.time() - time_start
    if(time_spent < dt): time.sleep(dt-time_spent)

# PLOT STUFF
time = np.arange(0.0, N*dt, dt)

if(PLOT_JOINT_POS):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, q[i,:-1], label=r'$q$ ')
        ax[i].plot(time, q_ref[i,:], '--', label=r'$q^{ref}$ ')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_%d$ [rad]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_VEL):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, v[i,:-1], label=r'$\dot{q}$ ')
        ax[i].plot(time, v_ref[i,:], '--', label=r'$\dot{q}^{ref}$ ')
        ax[i].plot([time[0], time[-1]], 2*[v_min[i]], ':')
        ax[i].plot([time[0], time[-1]], 2*[v_max[i]], ':')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\dot{q}_%d$ [rad/s]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_ACC):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, dv[i,:-1], label=r'$\ddot{q}$ ')
        ax[i].plot(time, dv_ref[i,:], '--', label=r'$\ddot{q}^{ref}$ ')
        ax[i].plot(time, dv_des[i,:], ':', label=r'$\ddot{q}^{des}$ ')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\ddot{q}_%d$ [rad/s^2]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
   
if(PLOT_TORQUES):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, tau[i,:], label=r'$\tau$')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\tau_%d$ [Nm]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
plt.show()
