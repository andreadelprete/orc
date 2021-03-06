# -*- coding: utf-8 -*-
"""
Optimize a trajectory for a robot using a single-shooting optimal control formulation.

@author: student
"""

import numpy as np
from numpy.linalg import norm
from example_robot_data.robots_loader import load

import orc.utils.plot_utils as plut
from orc.utils.robot_loaders import loadUR, loadURlab, loadPendulum
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
import orc.utils.lab_utils as lab

from single_shooting_problem import SingleShootingProblem
import single_shooting_conf as conf
from ode import ODERobot
from numerical_integration import Integrator
from cost_functions import OCPFinalCostState, OCPFinalCostFramePos, OCPFinalCostFrame
from cost_functions import OCPRunningCostQuadraticJointVel, OCPRunningCostQuadraticControl
from inequality_constraints import OCPFinalPlaneCollisionAvoidance, OCPPathPlaneCollisionAvoidance
from inequality_constraints import OCPFinalJointBounds, OCPPathJointBounds
from inequality_constraints import OCPFinalSelfCollisionAvoidance, OCPPathSelfCollisionAvoidance

np.set_printoptions(precision=3, linewidth=200, suppress=True)
    
dt = conf.dt                 # time step
T = conf.T
N = int(T/dt);         # horizon size
PLOT_STUFF = 1
system=conf.system

if(system=='ur'):
    r = loadUR()
elif(system=='ur-lab'):
    r = loadURlab() 
elif(system=='double-pendulum'):
    r = load('double_pendulum')
elif(system=='pendulum'):
    r = loadPendulum()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model, fixed_world_translation=conf.fixed_world_translation)
nq, nv = robot.nq, robot.nv    
n = nq+nv                       # state size
m = robot.na                    # control size

# compute initial guess for control inputs
U = np.zeros((N,m))           
u0 = robot.gravity(conf.q0)
for i in range(N):
    U[i,:] = u0

# create simulator 
simu = RobotSimulator(conf, robot)
lab.display_disi_lab(simu)

# show a blue sphere to display the target end-effector position in the viewer (if any)
if(conf.weight_final_pos>0):
    simu.gui.addSphere('world/target', 0.05, (0., 0., 1., 1.))
    simu.gui.setVisibility('world/target', 'ON')
    robot.applyConfiguration('world/target', conf.p_des.tolist()+[0.,0.,0.,1.])
else:
    simu.gui.setVisibility('world/target', 'OFF')

# add red spheres to display the volumes used for collision avoidance
for (frame, dist) in conf.table_collision_frames:
    simu.add_frame_axes(frame, radius=dist, length=0.0)
for (frame1, frame2, d) in conf.self_collision_frames:
    simu.add_frame_axes(frame1, radius=d, length=0.0)
    simu.add_frame_axes(frame2, radius=d, length=0.0)

# create OCP
ode = ODERobot('ode', robot, conf.B)
problem = SingleShootingProblem('ssp', ode, conf.x0, dt, N, conf.integration_scheme, simu)

# simulate motion with initial guess    
print("Showing initial motion in viewer")
integrator = Integrator('tmp')
X = integrator.integrate(ode, conf.x0, U, 0.0, dt, 1, N, conf.integration_scheme)
simu.display_motion(X[:,:nq], dt)
  
''' Create cost function terms '''
if(conf.weight_final_pos>0):
    final_cost = OCPFinalCostFramePos("final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, 
                                      conf.weight_final_vel)
#    final_cost = OCPFinalCostFrame("final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, conf.R_des, conf.w_des, conf.weight_vel)
    problem.add_final_cost(final_cost, conf.weight_final_pos)

if(conf.weight_final_q>0 or conf.weight_final_dq>0):
    final_cost_state = OCPFinalCostState("final state", robot, conf.q_des, np.zeros(nq), 
                                         conf.weight_final_q, conf.weight_final_dq)
    problem.add_final_cost(final_cost_state)

if(conf.weight_u>0):
    effort_cost = OCPRunningCostQuadraticControl("joint torques", robot, dt)
    problem.add_running_cost(effort_cost, conf.weight_u)

if(conf.weight_dq>0):
    dq_cost = OCPRunningCostQuadraticJointVel("joint vel", robot)
    problem.add_running_cost(dq_cost, conf.weight_dq)    

''' Create constraints '''
q_min = robot.model.lowerPositionLimit
q_max = robot.model.upperPositionLimit
dq_max = robot.model.velocityLimit
dq_min = -dq_max
joint_bounds = OCPPathJointBounds("joint bounds", robot, q_min, q_max, dq_min, dq_max)
problem.add_path_ineq(joint_bounds)

joint_bounds_final = OCPFinalJointBounds("joint bounds", robot, q_min, q_max, dq_min, dq_max)
problem.add_final_ineq(joint_bounds_final)

# inequalities for avoiding collisions with the table
for (frame, dist) in conf.table_collision_frames:
    table_avoidance = OCPPathPlaneCollisionAvoidance("table col "+frame, robot, frame, 
                                                     lab.table_normal, lab.table_pos[2]+dist)
    problem.add_path_ineq(table_avoidance)
    
    table_avoidance = OCPFinalPlaneCollisionAvoidance("table col fin "+frame, robot, frame, 
                                                     lab.table_normal, lab.table_pos[2]+dist)
    problem.add_final_ineq(table_avoidance)

# inequalities for avoiding self-collisions
for (frame1, frame2, min_dist) in conf.self_collision_frames:
    self_coll_avoid = OCPPathSelfCollisionAvoidance("self-col "+frame1+'-'+frame2, robot, 
                                                    frame1, frame2, min_dist)
    problem.add_path_ineq(self_coll_avoid)
    
    self_coll_avoid = OCPFinalSelfCollisionAvoidance("self-col "+frame1+'-'+frame2, robot, 
                                                    frame1, frame2, min_dist)
    problem.add_final_ineq(self_coll_avoid)

#    problem.sanity_check_cost_gradient(5)
#    import os
#    os.exit()

''' Solve OCP '''
#import cProfile
#cProfile.run("problem.solve(y0=U.reshape(N*m), use_finite_diff=conf.use_finite_diff)")
problem.solve(y0=U.reshape(N*m), use_finite_diff=conf.use_finite_diff)

X, U = problem.X, problem.U
print('U norm:', norm(U))
print('X_N\n', X[-1,:].T)

# display final optimized motion
print("Showing final motion in viewer")
simu.display_motion(X[:,:nq], dt, slow_down_factor=3)
print("To display the motion again type:\n simu.display_motion(X[:,:nq], dt)")

# SAVE THE RESULTS
np.savez_compressed(conf.DATA_FILE_NAME, q=X[:,:nq], v=X[:,nv:], u=U)

# PLOT STUFF
if(PLOT_STUFF):    
    time_array = np.arange(0.0, (N+1)*conf.dt, conf.dt)[:N+1]
    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time_array, X[:,i], label='q')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
        
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time_array, X[:,nq+i], label='v')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'v_'+str(i)+' [rad/s]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
        
   
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time_array[:-1], U[:,i], label=r'$\tau$ '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('Torque [Nm]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
