import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import time
import pinocchio as pin
from pinocchio.visualize import (GepettoVisualizer, MeshcatVisualizer)
from example_robot_data.robots_loader import load

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 80
T_SIMULATION = 5             # number of time steps simulated
dt = 0.01                      # controller time step
q0 = np.array([ 0. , -1.0,  0.7,  0. ,  0. ,  0. ]).T  # initial configuration
use_viewer = True
which_viewer = "meshcat"
simulate_real_time = True
PLOT_JOINT_POS = True
PRINT_T = 1                   # print every PRINT_N time steps
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]

print("".center(LINE_WIDTH,'#'))
print(" Test Software Optimization-Based Robot Control ".center(LINE_WIDTH, '#'))
print("".center(LINE_WIDTH,'#'), '\n')

# load the robot model
robot = load('ur5')

# launch the viewer
if(use_viewer):
    if(which_viewer=="gepetto"):
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
        VISUALIZER = MeshcatVisualizer

    robot.setVisualizer(VISUALIZER())    
    robot.initViewer(loadModel=True, open=True)
    robot.display(q0)
    if(which_viewer=="gepetto"):
        robot.viewer.gui.setCameraTransform('python-pinocchio', CAMERA_TRANSFORM)

N  = int(T_SIMULATION/dt)      # number of time steps
q  = np.empty((robot.nq, N+1))*nan  # joint angles
dq = np.zeros(robot.nv)
t  = 0.0
q[:,0] = q0
PRINT_N = int(PRINT_T/dt)

for i in range(0, N):
    time_start = time.time()
        
    # generate random joint accelerations between -1 and 1
    ddq = 2.0*np.random.rand(robot.nv) - 1.0
    # integrate accelerations to get velocities and positions
    q[:,i+1] = pin.integrate(robot.model, q[:,i], dq*dt)
    dq += dt*ddq
    
    # display the new robot configuration in the viewer
    robot.display(q[:,i+1]) 
        
    if i%PRINT_N == 0:
        print("Time %.3f"%(t))
    t += dt
        
    time_spent = time.time() - time_start
    if(simulate_real_time and time_spent < dt): 
        time.sleep(dt-time_spent)

# PLOT the joint angles
if(PLOT_JOINT_POS):    
    (f, ax) = f, ax = plt.subplots(int(robot.nv/2), 2, sharex=True)
    ax = ax.reshape(robot.nv)
    time = np.arange(0.0, N*dt, dt)
    for i in range(robot.nv):
        ax[i].plot(time, q[i,:-1], label='q')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)        
    plt.show()