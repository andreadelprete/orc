# -*- coding: utf-8 -*-
"""
Optimize a trajectory for a robot using a single-shooting optimal control formulation.

@author: Andrea Del Prete (andrea.delprete@unitn.it)
"""

import os
import numpy as np
from numpy.linalg import norm
from example_robot_data.robots_loader import load

import orc.utils.plot_utils as plut
from orc.utils.robot_loaders import loadUR, loadURlab
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
import orc.utils.lab_utils as lab
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration

from single_shooting_problem import SingleShootingProblem
import single_shooting_conf as conf
from orc.optimal_control.ode import ODERobot
from orc.optimal_control.numerical_integration import Integrator
from orc.optimal_control.cost_functions import OCPFinalCostState, OCPFinalCostFramePos, OCPFinalCostFrame
from orc.optimal_control.cost_functions import OCPRunningCostQuadraticJointVel, OCPRunningCostQuadraticJointAcc, OCPRunningCostQuadraticControl
from orc.optimal_control.inequality_constraints import OCPFinalPlaneCollisionAvoidance, OCPPathPlaneCollisionAvoidance
from orc.optimal_control.inequality_constraints import OCPFinalJointBounds, OCPPathJointBounds
from orc.optimal_control.inequality_constraints import OCPFinalSelfCollisionAvoidance, OCPPathSelfCollisionAvoidance
from orc.optimal_control.equality_constraints import OCPFinalConstraintState

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
robot = RobotWrapper(r.model, r.collision_model, r.visual_model, 
                     fixed_world_translation=lab.fixed_world_translation)
robot.setJointViscousFriction(conf.B)
nq, nv = robot.nq, robot.nv    
n = nq+nv                       # state size
m = robot.na                    # control size

# compute initial guess for control inputs
U = np.zeros((N,m))           
if(conf.INITIAL_GUESS_FILE is None):
    u0 = robot.gravity(conf.q0)
    for i in range(N):
        U[i,:] = u0
else:
    print("Load initial guess from", conf.INITIAL_GUESS_FILE)
    data = np.load(conf.DATA_FOLDER+conf.INITIAL_GUESS_FILE+'.npz') # , q=X[:,:nq], v=X[:,nv:], u=U
    U = data['u']

# create simulator 
print("Creating robot viewer...")
simu = RobotSimulator(conf, robot)
print("Viewer created.")

if(conf.use_viewer):
    if(conf.system=='ur-lab'):
        lab.display_disi_lab(simu)

    # show a blue sphere to display the target end-effector position in the viewer (if any)
    if(conf.weight_final_ee_pos>0):
        addViewerSphere(robot.viz, 'world/target', 0.05, (0., 0., 1., 1.))
        # simu.gui.addSphere('world/target', 0.05, (0., 0., 1., 1.))
        # simu.gui.setVisibility('world/target', 'ON')
        robot.applyConfiguration('world/target', conf.p_des.tolist()+[0.,0.,0.,1.])
    #else:
    #    simu.gui.setVisibility('world/target', 'OFF')
    
    # add red spheres to display the volumes used for collision avoidance
    for (frame, dist) in conf.table_collision_frames:
        simu.add_frame_axes(frame, radius=dist, length=0.0, color=(1.,0,0,0.2))
    for (frame1, frame2, d) in conf.self_collision_frames:
        simu.add_frame_axes(frame1, radius=d, length=0.0, color=(1.,0,0,0.2))
        simu.add_frame_axes(frame2, radius=d, length=0.0, color=(1.,0,0,0.2))

# create OCP
ode = ODERobot('ode', robot)
problem = SingleShootingProblem('ssp', ode, conf.x0, dt, N, conf.integration_scheme, simu)

# simulate motion with initial guess    
#print("Showing initial motion in viewer")
#integrator = Integrator('tmp')
#X = integrator.integrate(ode, conf.x0, U, 0.0, dt, 1, N, conf.integration_scheme)
#simu.display_motion(X[:,:nq], dt)
  
''' Create cost function terms '''
if(conf.weight_final_ee_pos>0):
    final_cost = OCPFinalCostFramePos("final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, 
                                      conf.weight_final_ee_vel)
#    final_cost = OCPFinalCostFrame("final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, conf.R_des, conf.w_des, conf.weight_vel)
    problem.add_final_cost(final_cost, conf.weight_final_ee_pos)

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

if(conf.weight_ddq>0):
    ddq_cost = OCPRunningCostQuadraticJointAcc("joint acc", robot)
    problem.add_running_cost(ddq_cost, conf.weight_ddq)    

''' Create constraints '''
if(conf.activate_final_state_constraint):
    final_constr_state = OCPFinalConstraintState("final state", robot, conf.q_des, np.zeros(nq))
    problem.add_final_eq(final_constr_state)

if(conf.activate_joint_bounds):
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
    table_avoidance = OCPPathPlaneCollisionAvoidance("col table-"+frame, robot, frame, 
                                                     lab.table_normal, lab.table_pos[2]+0.5*lab.table_size[2]+dist)
    problem.add_path_ineq(table_avoidance)
    
    table_avoidance = OCPFinalPlaneCollisionAvoidance("col fin table-"+frame, robot, frame, 
                                                     lab.table_normal, lab.table_pos[2]+0.5*lab.table_size[2]+dist)
    problem.add_final_ineq(table_avoidance)

# inequalities for avoiding self-collisions
for (frame1, frame2, min_dist) in conf.self_collision_frames:
    self_coll_avoid = OCPPathSelfCollisionAvoidance("col "+frame1+'-'+frame2, robot, 
                                                    frame1, frame2, min_dist)
    problem.add_path_ineq(self_coll_avoid)
    
    self_coll_avoid = OCPFinalSelfCollisionAvoidance("col fin "+frame1+'-'+frame2, robot, 
                                                    frame1, frame2, min_dist)
    problem.add_final_ineq(self_coll_avoid)

#problem.sanity_check_cost_gradient(5)
#os.exit()

''' Solve OCP '''
#import cProfile
#cProfile.run("problem.solve(y0=U.reshape(N*m), use_finite_diff=conf.use_finite_diff, max_iter=conf.max_iter)")
problem.solve(y0=U.reshape(N*m), 
              use_finite_diff=conf.use_finite_diff, 
              max_iter = conf.max_iter)

X, U = problem.X, problem.U
print('U norm:', norm(U))
print('q_N-q_des [deg]\n', (180/np.pi)*(X[-1,:nq]-conf.q_des))
print('dq_N [deg/s]\n', (180/np.pi)*X[-1,nq:])

# display final optimized motion
print("Showing final motion in viewer")
simu.display_motion(X[:,:nq], dt, slow_down_factor=3)
print("To display the motion again type:\n simu.display_motion(X[:,:nq], dt)")

# SAVE THE RESULTS
if(not os.path.exists(conf.DATA_FOLDER)):
    os.mkdir(conf.DATA_FOLDER)
np.savez_compressed(conf.DATA_FOLDER+conf.DATA_FILE_NAME, q=X[:,:nq], v=X[:,nv:], u=U)

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

    (f, ax) = plut.create_empty_figure(1)
    ax.plot(problem.history.cost)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    
    (f, ax) = plut.create_empty_figure(1)
    ax.plot(problem.history.grad)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient norm')
    ax.set_yscale('log')
    