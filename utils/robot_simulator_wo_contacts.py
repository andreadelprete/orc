import pinocchio as se3
import numpy as np
from numpy.linalg import norm
import os
import math
import gepetto.corbaserver
import time
import subprocess


def randomize_robot_model(model_old, sigma):
    ''' Function to randomly perturb the inertial parameters of a robot.
        sigma is the max perturbation allowed for the parameters, expressed
        as a percentage (0, 100)
    '''
    from random import uniform
    model = model_old.copy()
    for (ine, ine_old) in zip(model.inertias, model_old.inertias):
        ine.mass *= 1.0 + uniform(-sigma, sigma)*1e-2       # mass
        ine.lever *= 1.0 + uniform(-sigma, sigma)*1e-2      # center of mass
        ine.inertia *= 1.0 + uniform(-sigma, sigma)*1e-2    # rotational inertia
    return model


class RobotSimulator:

    # Class constructor
    def __init__(self, conf, robot):
        self.conf = conf
        self.robot = robot
        if(conf.randomize_robot_model):
            self.model = randomize_robot_model(robot.model, conf.model_variation)
        else:
            self.model = self.robot.model
        self.data = self.model.createData()
        self.t = 0.0                    # time
        self.nv = nv = self.model.nv    # size of joint velocities vector
        self.na = na = robot.na         # number of actuated joints
        # Matrix S used as filter of vetor of inputs U
        self.S = np.hstack((np.zeros((na, nv-na)), np.eye(na, na)))
        
        self.DISPLAY_T = conf.DISPLAY_T     # refresh period for viewer
        self.display_counter = self.DISPLAY_T
        self.init(conf.q0, None, True)
        
        self.tau_c = np.zeros(na)   # Coulomb friction torque
        self.simulate_coulomb_friction = conf.simulate_coulomb_friction
        self.simulation_type = conf.simulation_type
        if(self.simulate_coulomb_friction):
            self.tau_coulomb_max = 1e-2*conf.tau_coulomb_max*self.model.effortLimit
        else:
            self.tau_coulomb_max = np.zeros(na)
        
        if(norm(self.tau_coulomb_max)==0.0):
            self.simulate_coulomb_friction = False

        # setup gepetto viewer
        self.gui = None
        if(conf.use_viewer):
            try:
                prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(prompt[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
            except:
                pass
            gepetto.corbaserver.Client()
            self.robot.initViewer(loadModel=True)
            self.gui = self.robot.viewer.gui
            if(conf.show_floor):
                self.robot.viewer.gui.createSceneWithFloor('world')
                self.gui.setLightingMode('world/floor', 'OFF')
            self.robot.displayCollisions(False)
            self.robot.displayVisuals(True)
            self.robot.display(self.q)            
            try:  
                self.gui.setCameraTransform("python-pinocchio", conf.CAMERA_TRANSFORM)
            except:
                self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)            

    # Re-initialize the simulator
    def init(self, q0=None, v0=None, reset_contact_positions=False):
        if q0 is not None:
            self.q = q0.copy()
            
        if(v0 is None):
            self.v = np.zeros(self.robot.nv)
        else:
            self.v = v0.copy()
            
        self.dv = np.zeros(self.robot.nv)


    def step(self, u, dt=None):
        if dt is None:
            dt = self.dt

        # compute all quantities needed for simulation
        se3.computeAllTerms(self.model, self.data, self.q, self.v)
        se3.updateFramePlacements(self.model, self.data)
        M = self.data.M         # mass matrix
        h = self.data.nle       # nonlinear effects (gravity, Coriolis, centrifugal)

        if(self.simulate_coulomb_friction and self.simulation_type=='timestepping'):
            # minimize kinetic energy using time stepping
            from quadprog import solve_qp
            '''
            Solve a strictly convex quadratic program
            
            Minimize     1/2 x^T G x - a^T x
            Subject to   C.T x >= b
            
            Input Parameters:
            G : array, shape=(n, n)
            a : array, shape=(n,)
            C : array, shape=(n, m) matrix defining the constraints
            b : array, shape=(m), default=None, vector defining the constraints
            meq : int, default=0
                the first meq constraints are treated as equality constraints,
                all further as inequality constraints
            Output: a tuple, where the first element is the optimal x.
            '''
            # M (v' - v) = dt*S^T*(tau - tau_c) - dt*h + dt*J^T*f
            # M v' = M*v + dt*(S^T*tau - h + J^T*f) - dt*S^T*tau_c
            # M v' = b + B*tau_c
            # v' = Minv*(b + B*tau_c)
            b = M.dot(self.v) + dt*(self.S.T.dot(u) - h)
            B = - dt*self.S.T
            # Minimize kinetic energy:
            # min v'.T * M * v'
            # min  (b+B*tau_c​).T*Minv*(b+B*tau_c​) 
            # min tau_c.T * B.T*Minv*B* tau_C + 2*b.T*Minv*B*tau_c
            Minv = np.linalg.inv(M)
            G = B.T.dot(Minv.dot(B))
            a = -b.T.dot(Minv.dot(B))
            C = np.vstack((np.eye(self.na), -np.eye(self.na)))
            c = np.concatenate((-self.tau_coulomb_max, -self.tau_coulomb_max))
            solution = solve_qp(G, a, C.T, c, 0)
            self.tau_c = solution[0]
            self.v = Minv.dot(b + B.dot(self.tau_c))
            self.q = se3.integrate(self.model, self.q, self.v*dt)
            
        elif(self.simulation_type=='euler' or self.simulate_coulomb_friction==False):
            self.tau_c = self.tau_coulomb_max*np.sign(self.v[-self.na:])
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h)
            v_mean = self.v + 0.5*dt*self.dv
            self.v += self.dv*dt
            self.q = se3.integrate(self.model, self.q, v_mean*dt)
        else:
            print("[ERROR] Unknown simulation type:", self.simulation_type)

        self.t += dt
        return self.q, self.v


    def simulate(self, u, dt=0.001, ndt=1):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
        tau_c_sum = np.zeros_like(self.tau_c)
        sub_dt = dt/ndt
        for i in range(ndt):
            self.q, self.v = self.step(u, sub_dt)
            tau_c_sum += self.tau_c
        self.tau_c = tau_c_sum / ndt # compute average friction during time step

        if(self.conf.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0.0:
                self.display(self.q)

        return self.q, self.v

        
    def display(self, q):
        self.robot.display(q)
        self.display_counter = self.DISPLAY_T
        

        
if __name__=='__main__':
    from numpy import nan
    from example_robot_data.robots_loader import loadUR, getModelPath
    #from arc.utils.robot_loaders import loadUR
    from arc.utils.robot_wrapper import RobotWrapper
    import robot_simulator_conf as conf
    
    print("".center(conf.LINE_WIDTH,'#'))
    print(" Test Simulator ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH,'#'), '\n')
    
    T_SIMULATION = 50       # total simulation time
    dt = 0.01              # time step
    ndt = 100               # number of simulation step per time step
    conf.simulation_type = 'euler' #either 'timestepping' or 'euler'
    conf.tau_coulomb_max = 10*np.ones(6) # expressed as percentage of torque max
    
    r = loadUR()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    #robot = RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ])
    simu = RobotSimulator(conf, robot)
    
    N = int(T_SIMULATION/dt)           # number of time steps
    tau    = np.zeros(robot.na)             # joint torques
    tau_c  = np.empty((robot.na, N))*nan    # joint Coulomb torques
    
    t = 0.0
    for i in range(0, N):
        time_start = time.time()
                
        # send joint torques to simulator
        simu.simulate(tau, dt, ndt)
        tau_c[:,i] = simu.tau_c            
        t += dt
            
        time_spent = time.time() - time_start
        if(time_spent < dt): 
            time.sleep(dt-time_spent)
            
    print("Simulation finished")
    