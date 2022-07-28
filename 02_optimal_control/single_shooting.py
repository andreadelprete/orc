# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
import matplotlib.colors as mcolors
import warnings

from ode import ODERobot
from numerical_integration import Integrator

class Empty:
    def __init__(self):
        pass

class SingleShootingProblem:
    
    def __init__(self, name, ode, x0, dt, N, integration_scheme, simu):
        self.name = name
        self.ode = ode
        self.integrator = Integrator('integrator')
        self.x0 = x0
        self.dt = dt
        self.N = N
        self.integration_scheme = integration_scheme
        self.simu = simu
        
        self.nq = int(x0.shape[0]/2)
        self.nx = x0.shape[0]
        self.nu = self.ode.nu
        self.X = np.zeros((N, self.x0.shape[0]))
        self.U = np.zeros((N, self.nu))
        
        self.last_values = Empty()
        self.last_values.cost = 0.0
        self.last_values.running_cost = 0.0
        self.last_values.final_cost = 0.0
    
        self.running_costs = []
        self.final_costs = []
        self.path_ineqs = []
        self.final_ineqs = []
        
    def add_running_cost(self, c, weight=1):
        self.running_costs += [(weight,c)]
        self.last_values.__dict__[c.name] = 0.0
    
    def add_final_cost(self, c, weight=1):
        self.final_costs += [(weight,c)]
        self.last_values.__dict__[c.name] = 0.0
        
    def add_path_ineq(self, c):
        self.path_ineqs += [c]
        self.last_values.__dict__[c.name] = 0.0
    
    def add_final_ineq(self, c):
        self.final_ineqs += [c]
        self.last_values.__dict__[c.name] = 0.0
        
    '''*************************************************'''
    '''                 COST FUNCTIONS                  '''
    '''*************************************************'''
    
    def running_cost(self, X, U):
        ''' Compute the running cost integral '''
        cost = 0.0
        t = 0.0
        
        # reset the variables storing the costs
        for (w,c) in self.running_costs:
            self.last_values.__dict__[c.name] = 0.0
            
        for i in range(U.shape[0]):
            for (w,c) in self.running_costs:
                tmp = w * dt * c.compute(X[i,:], U[i,:], t, recompute=True)
                cost += tmp
                self.last_values.__dict__[c.name] += tmp
            t += self.dt
        return cost
        
    def running_cost_w_gradient(self, X, U, dXdU):
        ''' Compute the running cost integral and its gradient w.r.t. U'''
        cost = 0.0
        grad = np.zeros(self.N*self.nu)
        t = 0.0
        nx, nu = self.nx, self.nu
        
        # reset the variables storing the costs
        for (w,c) in self.running_costs:
            self.last_values.__dict__[c.name] = 0.0
            
        for i in range(U.shape[0]):
            for (w,c) in self.running_costs:
                ci, ci_x, ci_u = c.compute_w_gradient(X[i,:], U[i,:], t, recompute=True)
                dci = ci_x.dot(dXdU[i*nx:(i+1)*nx,:]) 
                dci[i*nu:(i+1)*nu] += ci_u
                
                cost += w * dt * ci
                grad += w * dt * dci
                self.last_values.__dict__[c.name] += w * dt * ci
            t += self.dt
        return (cost, grad)
        
    def final_cost(self, x_N):
        ''' Compute the final cost '''
        cost = 0.0
        for (w,c) in self.final_costs:
            tmp = w * c.compute(x_N, recompute=True)
            cost += tmp
            self.last_values.__dict__[c.name] = tmp
        return cost
        
    def final_cost_w_gradient(self, x_N, dxN_dU):
        ''' Compute the final cost and its gradient w.r.t. U'''
        cost = 0.0
        grad = np.zeros(self.N*self.nu)
        for (w,c) in self.final_costs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU)
            cost += w * ci
            grad += w * dci
            self.last_values.__dict__[c.name] = w * ci
        return (cost, grad)
        
    def compute_cost(self, y):
        ''' Compute cost function '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0, ndt = 0.0, 1
        X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                      self.N, self.integration_scheme)
        
        # compute cost
        run_cost = self.running_cost(X, U)
        fin_cost = self.final_cost(X[-1,:])
        cost = run_cost + fin_cost
        
        # store X, U and cost
        self.X, self.U = X, U
        self.last_values.cost = cost
        self.last_values.running_cost = run_cost
        self.last_values.final_cost = fin_cost
        return cost
        
    def compute_cost_w_gradient_fd(self, y):
        ''' Compute both the cost function and its gradient using finite differences '''
        eps = 1e-8
        y_eps = np.copy(y)
        grad = np.zeros_like(y)
        cost = self.compute_cost(y)
        for i in range(y.shape[0]):
            y_eps[i] += eps
            cost_eps = self.compute_cost(y_eps)
            y_eps[i] = y[i]
            grad[i] = (cost_eps - cost) / eps
        self.last_values.cost = cost
        return (cost, grad)
        
    def compute_cost_w_gradient(self, y):
        ''' Compute cost function and its gradient '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0 = 0.0
        X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                        self.dt, self.N, 
                                                        self.integration_scheme)
        
        # compute cost
        (run_cost, grad_run) = self.running_cost_w_gradient(X, U, dXdU)
        (fin_cost, grad_fin) = self.final_cost_w_gradient(X[-1,:], dXdU[-self.nx:,:])
        cost = run_cost + fin_cost
        grad = grad_run + grad_fin
        
        # store X, U and cost
        self.X, self.U, self.dXdU = X, U, dXdU
        self.last_values.cost = cost  
        self.last_values.running_cost = run_cost
        self.last_values.final_cost = fin_cost
        return (cost, grad)
    
    '''*************************************************'''
    '''                 INEQUALITIES                    '''
    '''*************************************************'''
    
    def path_ineq(self, X, U):
        ''' Compute the path inequalities '''
        ineq = []
        t = 0.0
        for c in self.path_ineqs:
            self.last_values.__dict__[c.name] = []
            
        for i in range(U.shape[0]):
            for c in self.path_ineqs:
                tmp = c.compute(X[i,:], U[i,:], t, recompute=True).tolist()
                ineq.extend(tmp)
                self.last_values.__dict__[c.name].extend(tmp)
            t += self.dt
        return ineq
                
    def final_ineq(self, x_N):
        ''' Compute the final inequalities '''
        ineq = []
        for c in self.final_ineqs:
            tmp = c.compute(x_N, recompute=True).tolist()
            ineq.extend(tmp)
            self.last_values.__dict__[c.name] = tmp
        return ineq
    
    def compute_ineq(self, y):
        ''' Compute all the the inequality constraints '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        if(norm(self.U-U)!=0.0):
            t0, ndt = 0.0, 1
            X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                          self.N, self.integration_scheme)
        else:
            X = self.X
        # compute inequalities
        run_ineq = self.path_ineq(X, U)
        fin_ineq = self.final_ineq(X[-1,:])
        ineq = np.array(run_ineq + fin_ineq) # concatenation
        
        # store X, U and ineq
        self.X, self.U = X, U
        self.last_values.ineq = ineq
        return ineq
    
    
    def path_ineq_w_jac(self, X, U, dXdU):
        ''' Compute the path inequalities '''
        ineq = []
        nx, nu = self.nx, self.nu
        jac = np.zeros((self.N*1000, self.N*nu)) # assume there are <1000 ineq
        index = 0
        t = 0.0
        for c in self.path_ineqs:
            self.last_values.__dict__[c.name] = []
            
        for i in range(U.shape[0]):
            for c in self.path_ineqs:
                ci, ci_x, ci_u = c.compute_w_gradient(X[i,:], U[i,:], t, recompute=True)
                dci = ci_x.dot(dXdU[i*nx:(i+1)*nx,:]) 
                dci[:, i*nu:(i+1)*nu] += ci_u
                
                ineq.extend(ci.tolist())
                jac[index:index+dci.shape[0],:] = dci
                index += dci.shape[0]
                
                self.last_values.__dict__[c.name].extend(ci.tolist())
            t += self.dt
            
        jac = jac[:index,:]
        return (ineq, jac)
    
            
    def final_ineq_w_jac(self, x_N, dxN_dU):
        ''' Compute the final inequalities '''
        ineq = []
        jac = np.zeros((1000, self.N*self.nu)) # assume there are <1000 ineq
        index = 0
        for c in self.final_ineqs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU) 
            
            ineq.extend(ci.tolist())
            jac[index:index+ci.shape[0],:] = dci
            index += dci.shape[0]
            
            self.last_values.__dict__[c.name] = ci.tolist()
            
        jac = jac[:index,:]
        return (ineq, jac)
    
    
    def compute_ineq_jac(self, y):
        ''' Compute all the the inequality constraints '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        if(norm(self.U-U)!=0.0):
            t0 = 0.0
            X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                        self.dt, self.N, 
                                                        self.integration_scheme)
        else:
            X, dXdU = self.X, self.dXdU
            
        # compute inequalities
        (run_ineq, jac_run) = self.path_ineq_w_jac(X, U, dXdU)
        (fin_ineq, jac_fin) = self.final_ineq_w_jac(X[-1,:], dXdU[-self.nx:,:])
        ineq = np.array(run_ineq + fin_ineq) # concatenation
        jac = np.vstack((jac_run, jac_fin))
        
        # store X, U and ineq
        self.X, self.U = X, U
        self.last_values.ineq = ineq
        return jac
        
        
    def solve(self, y0=None, method='SLSQP', use_finite_diff=False, max_iter=500):
        ''' Solve the optimal control problem '''
        # if no initial guess is given => initialize with zeros
        if(y0 is None):
            y0 = np.zeros(self.N*self.nu)
        
        self.iter = 0
        print('Start optimizing')
        if(use_finite_diff):
            cost_func = self.compute_cost_w_gradient_fd
        else:
            cost_func = self.compute_cost_w_gradient
        
        r = minimize(cost_func, y0, jac=True, method=method, 
                     callback=self.clbk, tol=1e-6, options={'maxiter': max_iter, 'disp': True},
                     constraints={'type': 'ineq', 'fun': self.compute_ineq,
                                  'jac': self.compute_ineq_jac})
        return r
        
    def sanity_check_cost_gradient(self, N_TESTS=10):
        ''' Compare the gradient computed with finite differences with the one
            computed by deriving the integrator
        '''
        for i in range(N_TESTS):
            y = np.random.rand(self.N*self.nu)
            (cost, grad_fd) = self.compute_cost_w_gradient_fd(y)
            (cost, grad) = self.compute_cost_w_gradient(y)
            grad_err = np.zeros_like(grad)
            for i in range(grad_err.shape[0]):
                grad_err[i] = np.abs(grad[i]-grad_fd[i])
                if(np.abs(grad_fd[i]) > 1.0): # normalize
                    grad_err[i] = np.abs(grad[i]-grad_fd[i])/grad_fd[i]
                    
            if(np.max(grad_err)>1e-2):
                print('Errors in gradient computations:', np.max(grad_err))
                print('Grad err:\n', grad-grad_fd)
                print('Grad FD:\n', grad_fd)
            else:
                print('Everything is fine', np.max(np.abs(grad_err)))
#                print('Grad FD:\n', 1e3*grad_fd)
#                print('Grad   :\n', 1e3*grad)
        
        
    def clbk(self, xk):
        print('Iter %3d, cost %5f'%(self.iter, self.last_values.cost))
        for (w,c) in self.running_costs:
            print("\t Running cost %40s: %9.3f"%(c.name, self.last_values.__dict__[c.name]))
        for (w,c) in self.final_costs:
            print("\t Final cost   %40s: %9.3f"%(c.name, self.last_values.__dict__[c.name]))
        for c in self.path_ineqs:
            print('\t Path ineq    %40s: %9.3f'%(c.name, np.min(self.last_values.__dict__[c.name])))
        for c in self.final_ineqs:
            print('\t Final ineq   %40s: %9.3f'%(c.name, np.min(self.last_values.__dict__[c.name])))
#        print('\t\tlast u:', self.U.T)
        self.iter += 1
        if(self.iter%100==0):
            self.display_motion()
        return False
        
    
    def display_motion(self, slow_down_factor=1):
        for i in range(0, self.N):
            time_start = time.time()
            q = self.X[i,:self.nq]
            self.simu.display(q)        
            time_spent = time.time() - time_start
            if(time_spent < slow_down_factor*self.dt):
                time.sleep(slow_down_factor*self.dt-time_spent)
        

if __name__=='__main__':
    import orc.utils.plot_utils as plut
    import matplotlib.pyplot as plt
    from orc.utils.robot_loaders import loadUR, loadURlab, loadPendulum
    from example_robot_data.robots_loader import load
    from orc.utils.robot_wrapper import RobotWrapper
    from orc.utils.robot_simulator import RobotSimulator
    import time, sys
    import single_shooting_conf as conf
    from cost_functions import OCPFinalCostState, OCPFinalCostFramePos, OCPFinalCostFrame
    from cost_functions import OCPRunningCostQuadraticJointVel, OCPRunningCostQuadraticControl
    from inequality_constraints import OCPFinalPlaneCollisionAvoidance, OCPPathPlaneCollisionAvoidance
    from inequality_constraints import OCPFinalJointBounds, OCPPathJointBounds
    from inequality_constraints import OCPFinalSelfCollisionAvoidance, OCPPathSelfCollisionAvoidance
    import pinocchio as pin
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
        
    dt = conf.dt                 # time step
    T = conf.T
    N = int(T/dt);         # horizon size
    PLOT_STUFF = 1
    linestyles = ['-*', '--*', ':*', '-.*']
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
    
    ode = ODERobot('ode', robot, conf.B)
    
    # create simulator 
    simu = RobotSimulator(conf, robot)
    
    # display table and stuff
    x_table = conf.table_pos #- conf.fixed_world_translation
    simu.gui.addBox("world/table", conf.table_size[0], conf.table_size[1], conf.table_size[2], (0.5, 0.5, 0.5, 1.))
    robot.applyConfiguration("world/table", (x_table[0], x_table[1], x_table[2], 0, 0, 0, 1))
    
    simu.gui.addLight("world/table_light", "python-pinocchio", 0.1, (1.,1,1,1))
    robot.applyConfiguration("world/table_light", (x_table[0], x_table[1], x_table[2]+1, 0, 0, 0, 1))
    
    simu.gui.addSphere('world/target', 0.05, (0., 0., 1., 1.))
    x_des = conf.p_des #- conf.fixed_world_translation
    robot.applyConfiguration('world/target', x_des.tolist()+[0.,0.,0.,1.])
    
    for frame in conf.table_collision_frames:
        simu.add_frame_axes(frame, radius=conf.safety_margin, length=0.0)
    for (frame1, frame2, d) in conf.self_collision_frames:
        simu.add_frame_axes(frame1, radius=d, length=0.0)
        simu.add_frame_axes(frame2, radius=d, length=0.0)
    
    # create OCP
    problem = SingleShootingProblem('ssp', ode, conf.x0, dt, N, conf.integration_scheme, simu)
    
    # simulate motion with initial guess    
    print("Showing initial motion in viewer")
    nq = robot.model.nq
    integrator = Integrator('tmp')
    X = integrator.integrate(ode, conf.x0, U, 0.0, dt, 1, N, conf.integration_scheme)
    for i in range(0, N):
        time_start = time.time()
        q = X[i,:nq]
        simu.display(q)        
        time_spent = time.time() - time_start
        if(time_spent < dt):
            time.sleep(dt-time_spent)
      
    # create cost function terms
    final_cost = OCPFinalCostFramePos("final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, 
                                      conf.weight_final_vel)
#    final_cost = OCPFinalCostFrame("final e-e pos", robot, conf.frame_name, conf.p_des, conf.dp_des, conf.R_des, conf.w_des, conf.weight_vel)
    problem.add_final_cost(final_cost, conf.weight_final_pos)
    
    final_cost_state = OCPFinalCostState("final dq", robot, conf.q_des, np.zeros(nq), 0.0, conf.weight_final_dq)
    problem.add_final_cost(final_cost_state)
    
    effort_cost = OCPRunningCostQuadraticControl("joint torques", robot, dt)
    problem.add_running_cost(effort_cost, conf.weight_u)
    
    dq_cost = OCPRunningCostQuadraticJointVel("joint vel", robot)
    problem.add_running_cost(dq_cost, conf.weight_dq)    
    
    # create constraints
    q_min = robot.model.lowerPositionLimit
    q_max = robot.model.upperPositionLimit
    dq_max = robot.model.velocityLimit
    dq_min = -dq_max
    joint_bounds = OCPPathJointBounds("joint bounds", robot, q_min, q_max, dq_min, dq_max)
    problem.add_path_ineq(joint_bounds)
    
    joint_bounds_final = OCPFinalJointBounds("joint bounds", robot, q_min, q_max, dq_min, dq_max)
    problem.add_final_ineq(joint_bounds_final)
    
    # inequalities for avoiding collisions with the table
    for frame in conf.table_collision_frames:
        table_avoidance = OCPPathPlaneCollisionAvoidance("table col "+frame, robot, frame, 
                                                         conf.table_normal, conf.table_pos[2]+conf.safety_margin)
        problem.add_path_ineq(table_avoidance)
        
        table_avoidance = OCPFinalPlaneCollisionAvoidance("table col fin "+frame, robot, frame, 
                                                         conf.table_normal, conf.table_pos[2]+conf.safety_margin)
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
    
    # solve OCP
#    import cProfile
#    cProfile.run("problem.solve(y0=U.reshape(N*m), use_finite_diff=conf.use_finite_diff)")
    problem.solve(y0=U.reshape(N*m), use_finite_diff=conf.use_finite_diff)
    
    X, U = problem.X, problem.U
    print('U norm:', norm(U))
    print('X_N\n', X[-1,:].T)
    
    # create simulator 
    print("Showing final motion in viewer")
    problem.display_motion(slow_down_factor=3)
    print("To display the motion again type:\nproblem.display_motion()")
    
    # SAVE THE RESULTS
    np.savez_compressed(conf.DATA_FILE_NAME, q=X[:,:nq], v=X[:,nv:], u=U)
    
    # PLOT STUFF
    time_array = np.arange(0.0, (N+1)*conf.dt, conf.dt)[:N+1]
    
    if(PLOT_STUFF):    
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
