import pinocchio as se3
import numpy as np
from numpy.linalg import norm
import os
import math
import time
import subprocess

class ContactPoint:
    ''' A point on the robot surface that can make contact with surfaces.
    '''
    def __init__(self, model, data, frame_name):
        self.model = model      # robot model
        self.data = data        # robot data
        self.frame_name = frame_name    # name of reference frame associated to this contact point
        self.frame_id = model.getFrameId(frame_name)    # id of the reference frame
        self.active = False         # True if this contact point is in contact
        
    def get_position(self):
        ''' Get the current position of this contact point 
        '''
        M = self.data.oMf[self.frame_id]
        return M.translation
        
    def get_velocity(self):
        M = self.data.oMf[self.frame_id]
        R = se3.SE3(M.rotation, 0*M.translation)    # same as M but with translation set to zero
        v_local = se3.getFrameVelocity(self.model, self.data, self.frame_id)
        v_world = (R.act(v_local)).linear   # convert velocity from local frame to world frame
        return v_world
        
    def get_jacobian(self):
        J6 = se3.getFrameJacobian(self.model, self.data, self.frame_id, se3.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J6[:3,:]
        
        
class ContactSurface:
    ''' A visco-elastic planar surface
    '''
    def __init__(self, name, pos, normal, K, B, mu):
        self.name = name        # name of this contact surface
        self.x0 = pos           # position of a point of the surface
        self.normal = normal    # direction of the normal to the surface
        self.K = K              # stiffness of the surface material
        self.B = B              # damping of the surface material
        self.mu = mu            # friction coefficient of the surface
        self.bias = self.x0.dot(self.normal)
        
    def check_collision(self, p):
        ''' Check the collision of the given point
            with this contact surface. If the point is not
            inside this surface, then return False.
        '''
        normal_penetration = self.bias - p.dot(self.normal)
        if(normal_penetration < 0.0):
            return False # no penetration
        return True
        
    def compute_force(self, contact_point, anchor_point):
        cp = contact_point
        p0 = anchor_point
        p = cp.get_position()
        v = cp.get_velocity()

        # compute contact force using spring-damper law
        f = self.K.dot(p0 - p) - self.B.dot(v)
        
        # check whether contact force is outside friction cone
        f_N = f.dot(self.normal)   # norm of normal force
        f_T = f - f_N*self.normal  # tangential force (3d)
        f_T_norm = norm(f_T)            # norm of tangential force
        if(f_T_norm > self.mu*f_N):
            # contact is slipping 
            t_dir = f_T / f_T_norm  # direction of tangential force
            # saturate force at the friction cone boundary
            f = f_N*self.normal + self.mu*f_N*t_dir
            
            # update anchor point so that f is inside friction cone
            delta_p0 = (f_T_norm - self.mu*f_N) / self.K[0,0]
            p0 -= t_dir*delta_p0
            
        return f, p0
        


class Contact:
    ''' A contact between a contact-point and a contact-surface
    '''
    def __init__(self, contact_point, contact_surface):
        self.cp = contact_point
        self.cs = contact_surface
        self.reset_contact_position()

    def reset_contact_position(self):
        # Initialize anchor point p0, that is the initial (0-load) position of the spring
        self.p0 = self.cp.get_position()
        self.in_contact = True

    def compute_force(self):
        self.f, self.p0 = self.cs.compute_force(self.cp, self.p0)
        return self.f
        
    def get_jacobian(self):
        return self.cp.get_jacobian()


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
#        print("mass", ine_old.mass, " => ", ine.mass)
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
        self.nv = nv = self.model.nv    # Dimension of joint velocities vector
        self.na = na = robot.na         # number of actuated joints
        # Matrix S used as filter of vetor of inputs U
        self.S = np.hstack((np.zeros((na, nv-na)), np.eye(na, na)))

        self.contacts = []
        self.candidate_contact_points = [] # candidate contact points
        self.contact_surfaces = []
        
        self.frame_axes = [] # list of frames whose axes must be displayed in viewer
        
        self.DISPLAY_T = conf.DISPLAY_T     # refresh period for viewer
        self.display_counter = self.DISPLAY_T
        self.init(conf.q0, None, True)
        
        self.tau_c = np.zeros(na)   # Coulomb friction torque
        self.simulation_type = conf.simulation_type
        self.set_coulomb_friction(conf.tau_coulomb_max)

        # for gepetto viewer
        self.gui = None
        if(conf.use_viewer):
            if(conf.which_viewer=="gepetto"):
                from pinocchio.visualize import GepettoVisualizer
                VISUALIZER = GepettoVisualizer
                import subprocess, os
                try:
                    prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                    if int(prompt[1]) == 0:
                        os.system('gepetto-gui &')
                    time.sleep(1)
                except:
                    pass
            else:
                from pinocchio.visualize import MeshcatVisualizer
                VISUALIZER = MeshcatVisualizer
                # self.viz = MeshcatVisualizer(robot.model, robot.collision_model,robot.visual_model)
                # self.viz.initViewer()
                # import webbrowser
                # webbrowser.open(self.viz.viewer.url())

            self.robot.setVisualizer(VISUALIZER()) 
            self.robot.initViewer(loadModel=True, open=True)
            self.robot.displayCollisions(False)
            self.robot.displayVisuals(True)
            self.robot.display(self.q)

            if(conf.which_viewer=='gepetto'):
                self.gui = self.robot.viewer.gui
                if(conf.show_floor):
                    self.gui.createSceneWithFloor('world')
                    self.gui.setLightingMode('world/floor', 'OFF')

#            LOCOSIM_PATH = "/home/adelprete/devel/src/locosim"
#            success = self.gui.addMesh("world/pinocchio/tavolo", LOCOSIM_PATH+"/ros_impedance_controller/worlds/models/tavolo/mesh/tavolo.stl")
#            if(success):
#                print("Table mesh added with success!")
                
#            try:  
#                self.gui.setCameraTransform("python-pinocchio", conf.CAMERA_TRANSFORM)
#            except:
#                self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)       
            
    def add_frame_axes(self, frame_name, color=(1.,0,0,0.5), radius=0.02, length=0.05):
        self.frame_axes += [frame_name]
        self.gui.addXYZaxis("world/axes-"+frame_name, color, radius, length)

    # Re-initialize the simulator
    def init(self, q0=None, v0=None, reset_contact_positions=False):
        self.first_iter = True

        if q0 is not None:
            self.q = q0.copy()
            
        if(v0 is None):
            self.v = np.zeros(self.robot.nv)
        else:
            self.v = v0.copy()
        self.dv = np.zeros(self.robot.nv)
        self.resize_contact_data(reset_contact_positions)
        
        
    def resize_contact_data(self, reset_contact_positions=False):
        self.nc = len(self.contacts)                    # number of contacts
        self.nk = 3*self.nc                             # size of contact force vector
        self.f = np.zeros(self.nk)                      # contact forces
        self.Jc = np.zeros((self.nk, self.model.nv))    # contact Jacobian

        # reset contact position
        if(reset_contact_positions):
            se3.forwardKinematics(self.model, self.data, self.q)
            se3.updateFramePlacements(self.model, self.data)
            for c in self.contacts:
                c.reset_contact_position()

        self.compute_forces(compute_data=True)
        
    def set_coulomb_friction(self, tau_max):
        self.tau_coulomb_max = 1e-2*tau_max*self.model.effortLimit        
        self.simulate_coulomb_friction = (norm(self.tau_coulomb_max)!=0.0)
        

    def add_candidate_contact_point(self, frame_name):
        self.candidate_contact_points += [ContactPoint(self.model, self.data, frame_name)]
        
    def add_contact_surface(self, name, pos, normal, K, B, mu):
        ''' Add a contact surface (i.e., a wall) located at "pos", with normal 
            outgoing direction "normal", 3d stiffness K, 3d damping B.
        '''
        self.contact_surfaces += [ContactSurface(name, pos, normal, K, B, mu)]
        
        # visualize surface in viewer
        if(self.gui):
            self.gui.addFloor('world/'+name)
            self.gui.setLightingMode('world/'+name, 'OFF')
            z = np.array([0.,0.,1.])
            axis = np.cross(normal, z)
            if(norm(axis)>1e-6):
                angle = math.atan2(np.linalg.norm(axis), normal.dot(z))
                aa = se3.AngleAxis(angle, axis)
                H = se3.SE3(aa.matrix(), pos)
                self.gui.applyConfiguration('world/'+name, se3.se3ToXYZQUATtuple(H))
            else:
                self.gui.applyConfiguration('world/'+name, pos.tolist()+[0.,0.,0.,1.])
    
        
    def collision_detection(self):
        for s in self.contact_surfaces:     # for each contact surface
            for cp in self.candidate_contact_points: # for each candidate contact point
                p = cp.get_position()
                if(s.check_collision(p)):   # check whether the point is colliding with the surface
                    if(not cp.active): # if the contact was not already active
                        print("Collision detected between point", cp.frame_name, " at ", p)
                        cp.active = True
                        cp.contact = Contact(cp, s)
                        self.contacts += [cp.contact]
                        self.resize_contact_data()
                else:
                    if(cp.active):
                        print("Contact lost between point", cp.frame_name, " at ", p)
                        cp.active = False
                        self.contacts.remove(cp.contact)
                        self.resize_contact_data()


    def compute_forces(self, compute_data=True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data:
            se3.forwardKinematics(self.model, self.data, self.q, self.v)
            # Computes the placements of all the operational frames according to the current joint placement stored in data
            se3.updateFramePlacements(self.model, self.data)
            self.collision_detection()
            
        i = 0
        for c in self.contacts:
            self.f[i:i+3] = c.compute_force()
            self.Jc[i:i+3, :] = c.get_jacobian()
            i += 3
        return self.f


    def step(self, u, dt=None):
        if dt is None:
            dt = self.dt

        # (Forces are directly in the world frame, and aba wants them in the end effector frame)
        se3.computeAllTerms(self.model, self.data, self.q, self.v)
        se3.updateFramePlacements(self.model, self.data)
        M = self.data.M
        h = self.data.nle
        self.collision_detection()
        self.compute_forces(False)

        if(self.simulate_coulomb_friction and self.simulation_type=='timestepping'):
            # minimize kinetic energy using time stepping
            from quadprog import solve_qp
            '''
            Solve a strictly convex quadratic program
            
            Minimize     1/2 x^T G x - a^T x
            Subject to   C.T x >= b
            
            Input Parameters
            ----------
            G : array, shape=(n, n)
            a : array, shape=(n,)
            C : array, shape=(n, m) matrix defining the constraints
            b : array, shape=(m), default=None, vector defining the constraints
            meq : int, default=0
                the first meq constraints are treated as equality constraints,
                all further as inequality constraints
            '''
            # M (v' - v) = dt*S^T*(tau - tau_c) - dt*h + dt*J^T*f
            # M v' = M*v + dt*(S^T*tau - h + J^T*f) - dt*S^T*tau_c
            # M v' = b + B*tau_c
            # v' = Minv*(b + B*tau_c)
            b = M.dot(self.v) + dt*(self.S.T.dot(u) - h + self.Jc.T.dot(self.f))
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
            self.dv = np.linalg.solve(M, self.S.T.dot(u-self.tau_c) - h + self.Jc.T.dot(self.f))
            v_mean = self.v + 0.5*dt*self.dv
            self.v += self.dv*dt
            self.q = se3.integrate(self.model, self.q, v_mean*dt)
        else:
            print("[ERROR] Unknown simulation type:", self.simulation_type)

        self.t += dt
        return self.q, self.v

    def reset(self):
        self.first_iter = True

    def simulate(self, u, dt=0.001, ndt=1):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
        tau_c_avg = 0*self.tau_c
        for i in range(ndt):
            self.q, self.v = self.step(u, dt/ndt)
            tau_c_avg += self.tau_c
        self.tau_c = tau_c_avg / ndt

        if(self.conf.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0.0:
                self.robot.display(self.q)
                self.display_counter = self.DISPLAY_T

        return self.q, self.v, self.f
        
    def display(self, q):
        if(self.conf.use_viewer):
            for frame in self.frame_axes:
                frame_id = self.robot.model.getFrameId(frame)
                H = self.robot.framePlacement(q, frame_id)
                self.robot.applyConfiguration("world/axes-"+frame, se3.SE3ToXYZQUATtuple(H))
#                self.gui.applyConfiguration("world/axes-"+frame, se3.SE3ToXYZQUATtuple(H))
                
            self.robot.display(q)
            
    def display_motion(self, Q, dt, slow_down_factor=1):
        for i in range(Q.shape[0]):
            time_start = time.time()
            self.display(Q[i,:])
            time_spent = time.time() - time_start
            if(time_spent < slow_down_factor*dt):
                time.sleep(slow_down_factor*dt-time_spent)