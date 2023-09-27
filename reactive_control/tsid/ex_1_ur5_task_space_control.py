import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut
from orc.utils.robot_loaders import loadUR
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration
import time
from tsid_manipulator import TsidManipulator

import ex_1_ur5_reaching_conf as conf

print(("".center(conf.LINE_WIDTH,'#')))
print((" Task Space Inverse Dynamics - Manipulator ".center(conf.LINE_WIDTH, '#')))
print(("".center(conf.LINE_WIDTH,'#')))

PLOT_EE_POS = 1
PLOT_EE_VEL = 1
PLOT_EE_ACC = 0
PLOT_JOINT_VEL = 1
PLOT_TORQUES = 1

tsid = TsidManipulator(conf)

r = loadUR()
robot_simu = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot_simu)

N = conf.N_SIMULATION
tau    = np.empty((tsid.robot.na, N))*nan
q      = np.empty((tsid.robot.nq, N+1))*nan
v      = np.empty((tsid.robot.nv, N+1))*nan
ee_pos = np.empty((3, N))*nan
ee_vel = np.empty((3, N))*nan
ee_acc = np.empty((3, N))*nan
ee_pos_ref = np.empty((3, N))*nan
ee_vel_ref = np.empty((3, N))*nan
ee_acc_ref = np.empty((3, N))*nan
ee_acc_des = np.empty((3, N))*nan # acc_des = acc_ref - Kp*pos_err - Kd*vel_err

sampleEE = tsid.trajEE.computeNext()
samplePosture = tsid.trajPosture.computeNext()

offset               = sampleEE.value()
offset[:3]          += conf.offset
two_pi_f_amp         = conf.two_pi_f * conf.amp
two_pi_f_squared_amp = conf.two_pi_f * two_pi_f_amp

pEE = offset.copy()
vEE = np.zeros(6)
aEE = np.zeros(6)

addViewerSphere(simu.robot.viz, 'world/ee', conf.SPHERE_RADIUS, conf.EE_SPHERE_COLOR)
addViewerSphere(simu.robot.viz, 'world/ee_ref', conf.REF_SPHERE_RADIUS, conf.EE_REF_SPHERE_COLOR)

t = 0.0
q[:,0], v[:,0] = tsid.q, tsid.v
time_avg = 0

for i in range(0, N):
    time_start = time.time()
    
    pEE[:3] = offset[:3] +  conf.amp * np.sin(conf.two_pi_f*t + conf.phi)
    vEE[:3] = two_pi_f_amp * np.cos(conf.two_pi_f*t + conf.phi)
    aEE[:3] = two_pi_f_squared_amp * -np.sin(conf.two_pi_f*t + conf.phi)
    sampleEE.value(pEE)
    sampleEE.derivative(vEE)
    sampleEE.second_derivative(aEE)
    tsid.eeTask.setReference(sampleEE)

    HQPData = tsid.formulation.computeProblemData(t, q[:,i], v[:,i])
    # if i == 0: HQPData.print_all()

    sol = tsid.solver.solve(HQPData)
    if(sol.status!=0):
        print(("Time %.3f QP problem could not be solved! Error code:"%t, sol.status))
        break
    
    tau[:,i] = tsid.formulation.getActuatorForces(sol)
    dv = tsid.formulation.getAccelerations(sol)
    
    ee_pos[:,i] = tsid.robot.framePosition(tsid.formulation.data(), tsid.EE).translation
    ee_vel[:,i] = tsid.robot.frameVelocityWorldOriented(tsid.formulation.data(), tsid.EE).linear
    ee_acc[:,i] = tsid.eeTask.getAcceleration(dv)[:3]
    ee_pos_ref[:,i] = sampleEE.value()[:3]
    ee_vel_ref[:,i] = sampleEE.derivative()[:3]
    ee_acc_ref[:,i] = sampleEE.second_derivative()[:3]
    ee_acc_des[:,i] = tsid.eeTask.getDesiredAcceleration[:3]

    if i%conf.PRINT_N == 0:
        print(("Time %.3f"%(t)))
        print(("\ttracking err %s: %.3f"%(tsid.eeTask.name.ljust(20,'.'), norm(tsid.eeTask.position_error, 2))))

    # send torque commands to simulator
    simu.simulate(tau[:,i], conf.dt)
    q[:,i+1] = simu.q
    v[:,i+1] = simu.v
    t += conf.dt
    
    if i%conf.DISPLAY_N == 0: 
        applyViewerConfiguration(simu.robot.viz, 'world/ee',     ee_pos[:,i].tolist()+[0,0,0,1.])
        applyViewerConfiguration(simu.robot.viz, 'world/ee_ref', ee_pos_ref[:,i].tolist()+[0,0,0,1.])

    time_spent = time.time() - time_start
    time_avg = (i * time_avg + time_spent) / (i + 1)
    if i % 100 == 0:
        print("Average loop time: %.1f (expected is %.1f)" % (1e3 * time_avg, 1e3 * conf.dt))

    if(time_spent < conf.dt): time.sleep(conf.dt-time_spent)

# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)

if(PLOT_EE_POS):
    (f, ax) = plut.create_empty_figure(3,1)
    for i in range(3):
        ax[i].plot(time, ee_pos[i,:], label=r'$x$')
        ax[i].plot(time, ee_pos_ref[i,:], 'r:', label=r'$x^{ref}$')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$x_%d$ [m]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_EE_VEL):
    (f, ax) = plut.create_empty_figure(3,1)
    for i in range(3):
        ax[i].plot(time, ee_vel[i,:], label=r'$\dot{x}$')
        ax[i].plot(time, ee_vel_ref[i,:], 'r:', label=r'$\dot{x}^{ref}$')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\dot{x}_%d$ [m/s]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_EE_ACC):    
    (f, ax) = plut.create_empty_figure(3,1)
    for i in range(3):
        ax[i].plot(time, ee_acc[i,:], label=r'$\ddot{x}$')
        ax[i].plot(time, ee_acc_ref[i,:], 'r:', label=r'$\ddot{x}^{ref}$')
        ax[i].plot(time, ee_acc_des[i,:], 'g--', label=r'$\ddot{x}^{des}$')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\ddot{x}_%d$ [m/s^2]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_TORQUES):    
    (f, ax) = plut.create_empty_figure(int(tsid.robot.nv/2),2)
    ax = ax.reshape(tsid.robot.nv)
    for i in range(tsid.robot.nv):
        ax[i].plot(time, tau[i,:], label=r'$\tau$')
        ax[i].plot([time[0], time[-1]], 2*[tsid.tau_min[i]], ':', label=r'$\tau^{min}$')
        ax[i].plot([time[0], time[-1]], 2*[tsid.tau_max[i]], ':', label=r'$\tau^{max}$')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\tau_%d$ [Nm]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_JOINT_VEL):    
    (f, ax) = plut.create_empty_figure(int(tsid.robot.nv/2),2)
    ax = ax.reshape(tsid.robot.nv)
    for i in range(tsid.robot.nv):
        ax[i].plot(time, v[i,:-1], label=r'$\dot{q}$')
        ax[i].plot([time[0], time[-1]], 2*[tsid.v_min[i]], ':', label=r'$\dot{q}^{min}$')
        ax[i].plot([time[0], time[-1]], 2*[tsid.v_max[i]], ':', label=r'$\dot{q}^{max}$')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\dot{q}_%d$ [rad/s]'%i)
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
plt.show()
