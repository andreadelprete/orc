# -*- coding: utf-8 -*-
"""
Created on 3 May  2022

@author: mfocchi
"""

from __future__ import print_function

import os
import rospy as ros
import sys
# messages for topic subscribers
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Wrench, Point
from std_srvs.srv import Trigger, TriggerRequest

# ros utils
import roslaunch
import rosnode
import rosgraph
import rospkg
from rospy import Time

#other utils
from base_controllers.utils.math_tools import *
import numpy as np
from numpy import nan
import pinocchio as pin
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 1000, suppress = True)
from six.moves import input # solves compatibility issue bw pyuthon 2.x and 3 for raw input that does exists in python 3
from termcolor import colored
import matplotlib.pyplot as plt
from base_controllers.utils.common_functions import plotJoint, plotAdmittanceTracking, plotEndeff

import  params as conf
import ur5_traj_track_conf as lab_conf
robotName = "ur5"

# controller manager management
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
from std_msgs.msg import Float64MultiArray

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import actionlib

from base_controllers.base_controller_fixed import BaseControllerFixed

import tf

def resend_robot_program():
    ros.sleep(1.5)
    ros.wait_for_service("/ur5/ur_hardware_interface/resend_robot_program")
    sos_service = ros.ServiceProxy('/ur5/ur_hardware_interface/resend_robot_program', Trigger)
    sos = TriggerRequest()
    result = sos_service(sos)
    # print(result)
    ros.sleep(0.1)


class LabAdmittanceController(BaseControllerFixed):
    
    def __init__(self, robot_name="ur5"):
        super().__init__(robot_name=robot_name)
        self.real_robot = conf.robot_params[self.robot_name]['real_robot']
        self.homing_flag = self.real_robot
        if (conf.robot_params[self.robot_name]['control_type'] == "torque"):
            self.use_torque_control = 1
        else:
            self.use_torque_control = 0
        self.world_name = None

        if self.use_torque_control and self.real_robot:
            print(colored(
                "ERRORS: unfortunately...you cannot use ur5 in torque control mode, talk with your course coordinator to buy a better robot...:))",
                'red'))
            sys.exit()

        print("Initialized L8 admittance  controller---------------------------------------------------------------")

    def startRealRobot(self):
        os.system("killall rviz gzserver gzclient")
        print(colored('------------------------------------------------ROBOT IS REAL!', 'blue'))

        if (not rosgraph.is_master_online()) or (
                "/" + self.robot_name + "/ur_hardware_interface" not in rosnode.get_node_names()):
            print(colored('ERROR: You should first launch the ur driver!', 'red'))
            sys.exit()

        # run rviz
        package = 'rviz'
        executable = 'rviz'
        args = '-d ' + rospkg.RosPack().get_path('ros_impedance_controller') + '/config/operator.rviz'
        node = roslaunch.core.Node(package, executable, args=args)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)

        resend_robot_program()

    def loadModelAndPublishers(self, xacro_path):
        super().loadModelAndPublishers(xacro_path)

        self.sub_ftsensor = ros.Subscriber("/" + self.robot_name + "/wrench", WrenchStamped,
                                           callback=self._receive_ftsensor, queue_size=1, tcp_nodelay=True)
        self.switch_controller_srv = ros.ServiceProxy(
            "/" + self.robot_name + "/controller_manager/switch_controller", SwitchController)
        self.load_controller_srv = ros.ServiceProxy("/" + self.robot_name + "/controller_manager/load_controller",
                                                    LoadController)
        # specific publisher for joint_group_pos_controller that publishes only position
        self.pub_reduced_des_jstate = ros.Publisher("/" + self.robot_name + "/joint_group_pos_controller/command",
                                                    Float64MultiArray, queue_size=10)

        self.zero_sensor = ros.ServiceProxy("/" + self.robot_name + "/ur_hardware_interface/zero_ftsensor", Trigger)

        #  different controllers are available from the real robot and in simulation
        if self.real_robot:
            self.available_controllers = [
                "joint_group_pos_controller",
                "scaled_pos_joint_traj_controller" ]
        else:
            self.available_controllers = ["joint_group_pos_controller",
                                          "pos_joint_traj_controller" ]
        self.active_controller = self.available_controllers[0]

        self.broadcaster = tf.TransformBroadcaster()

    def applyForce(self):
        wrench = Wrench()
        wrench.force.x = 0
        wrench.force.y = 0
        wrench.force.z = 30
        wrench.torque.x = 0
        wrench.torque.y = 0
        wrench.torque.z = 0
        reference_frame = "world" # you can apply forces only in this frame because this service is buggy, it will ignore any other frame
        reference_point = Point(x = 0, y = 0, z = 0)
        try:
            self.apply_body_wrench(body_name="ur5::wrist_3_link", reference_frame=reference_frame, reference_point=reference_point , wrench=wrench, duration=ros.Duration(10))
        except:
            pass

    def _receive_ftsensor(self, msg):
        contactForceTool0 = np.zeros(3)
        contactMomentTool0 = np.zeros(3)
        contactForceTool0[0] = msg.wrench.force.x
        contactForceTool0[1] = msg.wrench.force.y
        contactForceTool0[2] = msg.wrench.force.z
        contactMomentTool0[0] = msg.wrench.torque.x
        contactMomentTool0[1] = msg.wrench.torque.y
        contactMomentTool0[2] = msg.wrench.torque.z
        self.contactForceW = self.w_R_tool0.dot(contactForceTool0)
        self.contactMomentW = self.w_R_tool0.dot(contactMomentTool0)

                                                                                                                                     
    def updateKinematicsDynamics(self):
        # q is continuously updated
        # to compute in the base frame  you should put neutral base
        self.robot.computeAllTerms(self.q, self.qd)
        # joint space inertia matrix
        self.M = self.robot.mass(self.q)
        # bias terms
        self.h = self.robot.nle(self.q, self.qd)
        #gravity terms
        self.g = self.robot.gravity(self.q)
        #compute ee position  in the world frame
        frame_name = conf.robot_params[self.robot_name]['ee_frame']
        # this is expressed in a workdframe with the origin attached to the base frame origin
        self.x_ee = self.robot.framePlacement(self.q, self.robot.model.getFrameId(frame_name)).translation
        self.w_R_tool0 = self.robot.framePlacement(self.q, self.robot.model.getFrameId(frame_name)).rotation
        # compute jacobian of the end effector in the world frame
        self.J6 = self.robot.frameJacobian(self.q, self.robot.model.getFrameId(frame_name), False, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)                    
        # take first 3 rows of J6 cause we have a point contact            
        self.J = self.J6[:3,:] 
        #compute contact forces                        
        self.estimateContactForces()
        # broadcast base world TF
        self.broadcaster.sendTransform(self.base_offset, (0.0, 0.0, 0.0, 1.0), Time.now(), '/base_link', '/world')

    def estimateContactForces(self):  
        # estimate ground reaction forces from torques tau
        if self.use_torque_control:
            self.contactForceW = np.linalg.inv(self.J6.T).dot(self.h-self.tau)[:3]
                                 
    def startupProcedure(self):
        if (self.use_torque_control):
            #set joint pdi gains
            self.pid.setPDjoints( conf.robot_params[self.robot_name]['kp'], conf.robot_params[self.robot_name]['kd'], np.zeros(self.robot.na))
            #only torque loop
            #self.pid.setPDs(0.0, 0.0, 0.0)
        if (self.real_robot):
            self.zero_sensor()
        print(colored("finished startup -- starting controller", "red"))
        
    def initVars(self):
        super().initVars()

        # log variables relative to admittance controller
        self.q_des_adm_log = np.empty((self.robot.na, conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.x_ee_des_adm_log = np.empty((3, conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.EXTERNAL_FORCE = False
        self.payload_weight_avg = 0.0
        self.polynomial_flag = False

        self.Q_ref = []
        for name in lab_conf.traj_file_name:
            data = np.load(name + '.npz')
            self.Q_ref.append(data['q'])


    def logData(self):
        if (conf.robot_params[self.robot_name]['control_type'] == "admittance"):
            self.q_des_adm_log[:, self.log_counter] = self.q_des_adm
            self.x_ee_des_adm_log[:, self.log_counter] = self.x_ee_des_adm
        # I need to do this after because it updates log counter
        super().logData()

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        print('Available controllers: ',self.available_controllers)
        print('Controller manager: loading ', target_controller)

        other_controllers = (self.available_controllers)
        other_controllers.remove(target_controller)
        print('Controller manager:Switching off  :  ',other_controllers)

        srv = LoadControllerRequest()
        srv.name = target_controller

        self.load_controller_srv(srv)  
        
        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers 
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_controller_srv(srv)
        self.active_controller = target_controller

    def send_reduced_des_jstate(self, q_des):     
        msg = Float64MultiArray()
        msg.data = q_des             
        self.pub_reduced_des_jstate.publish(msg) 

    def send_joint_trajectory(self):
        # Creates a trajectory and sends it using the selected action server
        trajectory_client = actionlib.SimpleActionClient("{}/follow_joint_trajectory".format("/" + self.robot_name + "/"+self.active_controller), FollowJointTrajectoryAction)
        # Create and fill trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names

        # The following list are arbitrary positions
        # Change to your own needs if desired q0 [ 0.5, -0.7, 1.0, -1.57, -1.57, 0.5]), #limits([0,pi],   [0, -pi], [-pi/2,pi/2],)
        print(colored("JOINTS ARE: ", 'blue'), self.q.transpose())
        # position_list = [[0.5, -0.7, 1.0, -1.57, -1.57, 0.5]]  # limits([0,-pi], [-pi/2,pi/2],  [0, -pi])
        # position_list.append([0.5, -0.7 - 0.2, 1.0 - 0.1, -1.57, -1.57, 0.5])
        # position_list.append([0.5 + 0.5, -0.7 - 0.3, 1.0 - 0.1, -1.57, -1.57, 0.5])
        # position_list.append([0.5 + 0.5, -0.7 - 0.3, 1.0 , -1., -1.57, 0.5])

        self.q0 = conf.robot_params[p.robot_name]['q_0']
        dq1 = np.array([0.2, 0,0,0,0,0])
        dq2 = np.array([0.2, -0.2, 0, 0, 0, 0])
        dq3 = np.array([0.2, -0.2, 0.4, 0, 0, 0])
        position_list = [self.q0]  # limits([0,-pi], [-pi/2,pi/2],  [0, -pi])
        position_list.append(self.q0 + dq1)
        position_list.append(self.q0 + dq2)
        position_list.append(self.q0 + dq3)
        print(colored("List of targets for joints: ",'blue'))
        print(position_list[0])
        print(position_list[1])
        print(position_list[2])
        print(position_list[3])

        duration_list = [5.0, 10.0, 20.0, 30.0]
        for i, position in enumerate(position_list):
            point = JointTrajectoryPoint()
            point.positions = position
            point.time_from_start = ros.Duration(duration_list[i])
            goal.trajectory.points.append(point)

        self.ask_confirmation(position_list)     
        print("Executing trajectory using the {}".format("pos_joint_traj_controller"))
        trajectory_client.send_goal(goal)
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()
        print("Trajectory execution finished in state {}".format(result.error_code))
        
    def ask_confirmation(self, waypoint_list):
        """Ask the user for confirmation. This function is obviously not necessary, but makes sense
        in a testing script when you know nothing about the user's setup."""
        ros.logwarn("The robot will move to the following waypoints: \n{}".format(waypoint_list))
        confirmed = False
        valid = False
        while not valid:
            input_str = input(
                "Please confirm that the robot path is clear of obstacles.\n"
                "Keep the EM-Stop available at all times. You are executing\n"
                "the motion at your own risk. Please type 'y' to proceed or 'n' to abort: " )
            valid = input_str in ["y", "n"]
            if not valid:
                ros.loginfo("Please confirm by entering 'y' or abort by entering 'n'")
            else:
                if (input_str == "y"):
                    confirmed = True
        if not confirmed:
            ros.loginfo("Exiting as requested by user.")
            sys.exit(0)

    def deregister_node(self):
        super().deregister_node()
        if not self.real_robot:
            os.system(" rosnode kill /"+self.robot_name+"/ros_impedance_controller")
            os.system(" rosnode kill /gzserver /gzclient")

    def plotStuff(self):
        if not (conf.robot_params[p.robot_name]['control_mode'] == "trajectory"):
            plotJoint('position', 0, self.time_log, self.q_log, self.q_des_log, self.qd_log, self.qd_des_log, None, None, self.tau_log,
                      self.tau_ffwd_log, self.joint_names, self.q_des_log)
            plotJoint('torque', 2, self.time_log, self.q_log, self.q_des_log, self.qd_log, self.qd_des_log, None, None, self.tau_log,
                      self.tau_ffwd_log, self.joint_names)
            plotEndeff('force', 1, p.time_log, p.contactForceW_log)
            plt.show(block=True)

    def move_gripper(self, diameter):
        if not self.real_robot:
            return

        import socket
        HOST = "192.168.0.100"  # The UR IP address
        PORT = 30002  # UR secondary client
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.settimeout(0.5)
        try:
            sock.connect((HOST, PORT))
        except:
            raise Exception("Cannot connect to end-effector socket") from None
        sock.settimeout(None)
        scripts_path = rospkg.RosPack().get_path('ur_description') + '/gripper/scripts'

        onrobot_script = scripts_path + "/onrobot_superminimal.script"
        file = open(onrobot_script, "rb")  # Robotiq Gripper
        lines = file.readlines()
        file.close()

        tool_index = 0
        blocking = True
        cmd_string = f"tfg_release({diameter},  tool_index={tool_index}, blocking={blocking})"

        line_number_to_add = 446

        new_lines = lines[0:line_number_to_add]
        new_lines.insert(line_number_to_add + 1, str.encode(cmd_string))
        new_lines += lines[line_number_to_add::]

        offset = 0
        buffer = 2024
        file_to_send = b''.join(new_lines)

        if len(file_to_send) < buffer:
            buffer = len(file_to_send)
        data = file_to_send[0:buffer]
        while data:
            sock.send(data)
            offset += buffer
            if len(file_to_send) < offset + buffer:
                buffer = len(file_to_send) - offset
            data = file_to_send[offset:offset + buffer]
        sock.close()

        print("Gripper moved, now resend robot program")
        resend_robot_program()
        return


def talker(p):
    p.start()
    if p.real_robot:
        p.startRealRobot()
    else:
        p.startSimulator(p.world_name, p.use_torque_control)

    # specify xacro location
    xacro_path = rospkg.RosPack().get_path('ur_description') + '/urdf/' + p.robot_name + '.xacro'
    p.loadModelAndPublishers(xacro_path)
    p.initVars()
    p.startupProcedure()


    p.q_des_q0 = conf.robot_params[p.robot_name]['q_0']
    p.q_des = np.copy(p.q_des_q0)

    #loop frequency
    rate = ros.Rate(1/conf.robot_params[p.robot_name]['dt'])

    if not p.use_torque_control:            
        p.switch_controller("joint_group_pos_controller")
    # reset to actual
    p.updateKinematicsDynamics()
    p.time_poly = None

    ext_traj_counter = 0    # counter for which trajectory is currently tracked
    ext_traj_t = 0          # counter for the time inside a trajectory
    traj_completed = False

    #control loop
    while True:
        # homing procedure
        if p.homing_flag:
            dt = conf.robot_params[p.robot_name]['dt']
            v_des = lab_conf.v_des_homing
            v_ref = 0.0
            print(colored("STARTING HOMING PROCEDURE",'red'))
            q_home = conf.robot_params[p.robot_name]['q_0']
            p.q_des = np.copy(p.q)
            print("Initial joint error = ", np.linalg.norm(p.q_des - q_home))
            print("q = ", p.q.T)
            print("Homing v des", v_des)
            while True:
                e = q_home - p.q_des
                e_norm = np.linalg.norm(e)
                if(e_norm!=0.0):
                    v_ref += 0.005*(v_des-v_ref)
                    p.q_des += dt*v_ref*e/e_norm
                    p.send_reduced_des_jstate(p.q_des)
                rate.sleep()
                if (e_norm<0.001):
                    p.homing_flag = False
                    print(colored("HOMING PROCEDURE ACCOMPLISHED", 'red'))
                    p.move_gripper(30)
                    print(colored("GRIPPER CLOSED", 'red'))
                    break

        #update the kinematics
        p.updateKinematicsDynamics()

        if (int(ext_traj_t) < p.Q_ref[ext_traj_counter].shape[0]): # and p.time>6.0:
            p.q_des = p.Q_ref[ext_traj_counter][int(ext_traj_t),:]
            ext_traj_t += 1.0/lab_conf.traj_slow_down_factor
        else:
            if(ext_traj_counter < len(p.Q_ref)-1):
                print(colored("TRAJECTORY %d COMPLETED"%ext_traj_counter, 'blue'))
                if(ext_traj_counter==0):
                    p.move_gripper(65)
                if (ext_traj_counter == 1):
                    p.move_gripper(30)
                ext_traj_counter += 1
                ext_traj_t = 0
            elif(not traj_completed):
                print(colored("LAST TRAJECTORY COMPLETED", 'red'))
                p.move_gripper(60)
                traj_completed = True
                    
        # controller with gravity coriolis comp
        p.tau_ffwd = p.h + np.zeros(p.robot.na)
        q_to_send = p.q_des

        # send commands to gazebo
        if (p.use_torque_control):
            p.send_des_jstate(q_to_send, p.qd_des, p.tau_ffwd)
        else:
            p.send_reduced_des_jstate(q_to_send)

        p.ros_pub.add_arrow(p.x_ee + p.base_offset, p.contactForceW / (6 * p.robot.robot_mass), "green")

        # log variables
        if (p.time > 1.0):
            p.logData()

        # plot end-effector
        p.ros_pub.add_marker(p.x_ee + p.base_offset)
        p.ros_pub.publishVisual()

        #wait for synconization of the control loop
        rate.sleep()

        p.time = p.time + conf.robot_params[p.robot_name]['dt']
       # stops the while loop if  you prematurely hit CTRL+C
        if ros.is_shutdown():
            p.plotStuff()
            print ("Shutting Down")
            break

    print("Shutting Down")
    ros.signal_shutdown("killed")
    p.deregister_node()


if __name__ == '__main__':
    p = LabAdmittanceController(robotName)
    try:
        talker(p)
    except ros.ROSInterruptException:
        # these plots are for simulated robot
        p.plotStuff()
    