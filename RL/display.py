# Typical header of a Python script using Pinocchio
from pinocchio.utils import *
import pinocchio as pin
from orc.utils.viz_utils import applyViewerConfiguration, addViewerSphere, addViewerCapsule

# Example of a class Display that connect to the viewer and implement a
# 'place' method to set the position/rotation of a 3D visual object in a scene.
class Display():
    '''
    Class Display: Example of a class implementing a client for the viewer. The main
    method of the class is 'place', that sets the position/rotation of a 3D visual object in a scene.
    '''
    def __init__(self, which_viewer='meshcat', windowName = "pinocchio" ):
        '''
        This function connects with the viewer and opens a window with the given name.
        If the window already exists, it is kept in the current state. Otherwise, the newly-created
        window is set up with a scene named 'world'.
        '''
        self.which_viewer = which_viewer
        if(which_viewer=='meshcat'):
            from pinocchio.visualize import MeshcatVisualizer
            self.viewer = MeshcatVisualizer()
            self.viewer.initViewer()
        elif(which_viewer=="gepetto"):
            # from pinocchio.visualize import GepettoVisualizer
            # VISUALIZER = GepettoVisualizer
            import gepetto.corbaserver
            import subprocess
            import os
            import time
            l = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(l[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(2)
        
            # Create the client and connect it with the display server.
            try:
                self.viewer=gepetto.corbaserver.Client()
            except:
                print("Error while starting the viewer client. ")
                print("Check whether Gepetto-viewer is properly started")

            # Open a window for displaying your model.
            try:
                # If the window already exists, do not do anything.
                windowID = self.viewer.gui.getWindowID (windowName)
                print("Warning: window '"+windowName+"' already created.")
                print("The previously created objects will not be destroyed and do not have to be created again.")
            except:
                # Otherwise, create the empty window.
                windowID = self.viewer.gui.createWindow (windowName)
                # Start a new "scene" in this window, named "world", with just a floor.
                self.viewer.gui.createScene("world")
                self.viewer.gui.addSceneToWindow("world",windowID)

            # Finally, refresh the layout to obtain your first rendering.
            self.viewer.gui.refresh()

    def addSphere(self, name, radius, color):
        addViewerSphere(self.viewer, name, radius, color)

    def addCapsule(self, name, radius, length, color):
        addViewerCapsule(self.viewer, name, radius, length, color)
        
    def refresh(self):
        if isinstance(self.viewer, pin.visualize.GepettoVisualizer):
            self.viewer.viewer.gui.refresh()

    def place(self, objName, M, refresh=True):
        '''
        This function places (ie changes both translation and rotation) of the object
        names "objName" in place given by the SE3 object "M". By default, immediately refresh
        the layout. If multiple objects have to be placed at the same time, do the refresh
        only at the end of the list.
        '''
        applyViewerConfiguration(self.viewer, objName, pin.SE3ToXYZQUATtuple(M))
        if refresh: 
            self.refresh()
