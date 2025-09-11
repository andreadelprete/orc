import numpy as np

DATA_FILE_LIPM = "aliengo_trot.npz"
DATA_FILE_CTRL = "aliengo_trot_ctrl.npz"

# The online URDF of Aliengo has unbounded joints, which are represented using sine and cosine of the joint angle
# therefore they don't match the Mujoco state representation. For this reason I need to use a local version of the
# URDF, but while still downloading the whole model online for the meshes.

# from robot_descriptions.loaders.pinocchio import load_robot_description
# robot = load_robot_description("aliengo_description")

from os import getenv as _getenv
from os import path as _path
from robot_descriptions._cache import clone_to_cache as _clone_to_cache
REPOSITORY_PATH: str = _clone_to_cache("unitree_ros", commit=_getenv("ROBOT_DESCRIPTION_COMMIT", None),)
PACKAGE_PATH: str = _path.join(REPOSITORY_PATH, "robots")
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
robot = RobotWrapper.BuildFromURDF("aliengo.urdf", [PACKAGE_PATH], root_joint=pin.JointModelFreeFlyer())

foot_names = ["FL", "FR", "RL", "RR"]
hip_joint_names = {"FL": "FL_hip_joint", "FR": "FR_hip_joint", "RL": "RL_hip_joint", "RR": "RR_hip_joint"}
hip_joint_ids, hip_pos = {}, {}
for k in hip_joint_names.keys():
    hip_joint_ids[k] = robot.model.getJointId(hip_joint_names[k]) 
    hip_pos[k] = robot.placement(robot.q0, hip_joint_ids[k]).translation[:2]

# configuration for LIPM trajectory optimization
# ----------------------------------------------
wu = 1e1    # CoP error squared cost weight
wc = 0      # CoM position error squared cost weight
wdc = 1e-1  # CoM velocity error squared cost weight
wp = 1e-2   # footstep distance to hip cost weight
h = 0.5     # fixed CoM height
g = 9.81    # norm of the gravity vector
foot_step_0 = ["FL", "RR"]  # initial foot steps on the ground
dt_mpc = 0.1  # sampling time interval
T_step = 0.6  # time needed for every step
step_height = 0.05  # fixed step height
step_length = 0.05
nb_steps = 6  # number of desired walking steps

# configuration for controller
# ----------------------------------------------
dt = 0.002
mujoco_robot_name = "aliengo_mj_description"