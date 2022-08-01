from pinocchio.robot_wrapper import RobotWrapper as PinocchioRobotWrapper
from pinocchio.deprecation import deprecated
import pinocchio as pin
import pinocchio.utils as utils
from pinocchio.explog import exp
import numpy as np


class RobotWrapper(PinocchioRobotWrapper):
    
    @staticmethod
    def BuildFromURDF(filename, package_dirs=None, root_joint=None, verbose=False, meshLoader=None):
        robot = RobotWrapper()
        robot.initFromURDF(filename, package_dirs, root_joint, verbose, meshLoader)
        return robot
    
    def __init__(self, model = pin.Model(), collision_model = None, visual_model = None, verbose=False, fixed_world_translation=None):
        super().__init__(model, collision_model, visual_model, verbose)
        self.B = np.zeros((model.nv, model.nv))
        if(fixed_world_translation is None):
            self.w2b = np.zeros(3) # world to base fixed translation
        else:
            self.setFixedWorldTranslation(fixed_world_translation)
    
    @property
    def na(self):
        if(self.model.joints[0].nq==7):
            return self.model.nv-6
        return self.model.nv
    
    def setJointViscousFriction(self, friction_coefficients):
        assert(friction_coefficients.shape[0]==self.model.nv)
#        self.model.friction = B # do not work
#        self.model.damping = B  # do not work
        self.B = np.diagflat(friction_coefficients)
    
    def setFixedWorldTranslation(self, translation):
        ''' Set a fixed translation from the world frame to the universe frame of the robot URDF '''
        assert(translation.shape[0]==3)
        self.w2b = translation
        
    def placement(self, q, index, update_kinematics=True):
        H = PinocchioRobotWrapper.placement(self, q, index, update_kinematics)
        H.translation += self.w2b
        return H
    
    def framePlacement(self, q, index, update_kinematics=True):
        H = PinocchioRobotWrapper.framePlacement(self, q, index, update_kinematics)
        H.translation += self.w2b
        return H
    
    def applyConfiguration(self, node_name, configuration):
        c = np.array(configuration)
        c[:3] -= self.w2b
        self.viz.viewer.gui.applyConfiguration(node_name, c.tolist())

    def computeAllTerms(self, q, v):
        ''' pin.computeAllTerms is equivalent to calling:
            pinocchio::forwardKinematics
            pinocchio::crba
            pinocchio::nonLinearEffects
            pinocchio::computeJointJacobians
            pinocchio::centerOfMass
            pinocchio::jacobianCenterOfMass
            pinocchio::kineticEnergy
            pinocchio::potentialEnergy
            This is too much for our needs, so we call only the functions
            we need, including those for the frame kinematics
        '''
#        pin.computeAllTerms(self.model, self.data, q, v);
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        pin.computeJointJacobians(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)
        pin.nonLinearEffects(self.model, self.data, q, v)
        
        
    def forwardKinematics(self, q, v=None, a=None):
        if v is not None:
            if a is not None:
                pin.forwardKinematics(self.model, self.data, q, v, a)
            else:
                pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q)
               
    def frameJacobian(self, q, index, update=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        ''' Call computeFrameJacobian if update is true. If not, user should call computeFrameJacobian first.
            Then call getFrameJacobian and return the Jacobian matrix.
            ref_frame can be: ReferenceFrame.LOCAL, ReferenceFrame.WORLD, ReferenceFrame.LOCAL_WORLD_ALIGNED
        '''
        if(update): 
            pin.computeFrameJacobian(self.model, self.data, q, index)
        return pin.getFrameJacobian(self.model, self.data, index, ref_frame)
    
    
        
    def frameVelocity(self, q, v, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v)
        v_local = pin.getFrameVelocity(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return v_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            v_world = H.act(v_local)
            return v_world
        
        Hr = pin.SE3(H.rotation, np.zeros(3))
        v = Hr.act(v_local)
        return v
            

    def frameAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.SE3(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
        
    def frameClassicAcceleration(self, q, v, a, index, update_kinematics=True, ref_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        v = pin.getFrameVelocity(self.model, self.data, index)
        a_local = pin.getFrameAcceleration(self.model, self.data, index)
        a_local.linear += np.cross(v.angular, v.linear, axis=0)
        if ref_frame==pin.ReferenceFrame.LOCAL:
            return a_local
            
        H = self.data.oMf[index]
        if ref_frame==pin.ReferenceFrame.WORLD:
            a_world = H.act(a_local)
            return a_world
        
        Hr = pin.SE3(H.rotation, np.zeros(3))
        a = Hr.act(a_local)
        return a
    
    def mass(self, q, update=True):
        if(update):
            return pin.crba(self.model, self.data, q)
        return self.data.M

    def nle(self, q, v, update=True):
        if(update):
            pin.nonLinearEffects(self.model, self.data, q, v)
        return self.data.nle - self.B @ v
    
    def aba(self, q, v, u):
        return pin.aba(self.model, self.data, q, v, u) - self.B @ v
    
    def abaDerivatives(self, q, v, u):
        pin.computeABADerivatives(self.model, self.data, q, v, u)
        return self.data.ddq_dq, self.data.ddq_dv - self.B, self.data.Minv
        
    def com(self, q=None, v=None, a=None, update=True):
        if(update==False or q is None):
            return PinocchioRobotWrapper.com(self, q);
        if a is None:
            if v is None:
                return PinocchioRobotWrapper.com(self, q)
            return PinocchioRobotWrapper.com(self, q, v)
        return PinocchioRobotWrapper.com(self, q, v,a)
        
    def Jcom(self, q, update=True):
        if(update):
            return pin.jacobianCenterOfMass(self.model, self.data, q)
        return self.data.Jcom
        
    def momentumJacobian(self, q, v, update=True):
        if(update):
            pin.ccrba(self.model, self.data, q, v);
        return self.data.Ag;
      
    def deactivateCollisionPairs(self, collision_pair_indexes):
        for i in collision_pair_indexes:
            self.collision_data.deactivateCollisionPair(i);
            
    def addAllCollisionPairs(self):
        self.collision_model.addAllCollisionPairs();
        self.collision_data = pin.GeometryData(self.collision_model);
        
    def isInCollision(self, q, stop_at_first_collision=True):
        return pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, np.asmatrix(q).reshape((self.model.nq,1)), stop_at_first_collision);

    def findFirstCollisionPair(self, consider_only_active_collision_pairs=True):
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    return (i, self.collision_model.collisionPairs[i]);
        return None;
        
    def findAllCollisionPairs(self, consider_only_active_collision_pairs=True):
        res = [];
        for i in range(len(self.collision_model.collisionPairs)):
            if(not consider_only_active_collision_pairs or self.collision_data.activeCollisionPairs[i]):
                if(pin.computeCollision(self.collision_model, self.collision_data, i)):
                    res += [(i, self.collision_model.collisionPairs[i])];
        return res;

__all__ = ['RobotWrapper']
