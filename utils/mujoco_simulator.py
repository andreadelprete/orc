import mujoco
from mujoco import viewer
from os.path import join
import time
import numpy as np
from example_robot_data.robots_loader import loader

class MujocoSimulator:

    # Constants
    PIN_2_MJ_POS = [0,1,2,6,3,4,5]
    MJ_2_PIN_POS = [0,1,2,4,5,6,3]
    
    def __init__(self, robot_name, time_step):
        model_loaded = False
        try:
            from robot_descriptions.loaders.mujoco import load_robot_description
            self.model = load_robot_description(robot_name)
            model_loaded = True
        except:
            print("Failed to load robot model with robot_descriptions package")

        if(not model_loaded):
            # read the URDF file of the robot using example_robot_data
            inst = loader(robot_name)
            self.robot_wrapper = inst.robot
            print("urdf file name: ", inst.robot.urdf)
            urdf_file = open(inst.robot.urdf, "r")
            urdf_content = urdf_file.readlines()
            urdf_file.close()

            mesh_dir_found = False
            for mesh_extension in [".stl", ".dae"]:
                for s in urdf_content:
                    # search for a line in the URDF containing the relative path of a mesh (stl) file
                    if("package://" in s and mesh_extension in s):
                        start_ind = s.find("package://")
                        end_ind = s.find(mesh_extension)
                        path = s[start_ind+10:end_ind]
                        meshdir = join(inst.model_path, "../../", path[:path.rfind("/")])
                        # print("Mesh path found:", meshdir)
                        mesh_dir_found = True
                        break
                if(mesh_dir_found):
                    break

            if(not mesh_dir_found):
                print("WARNING! Could not find mesh dir in URDF file.")

            xml = ""
            for s in urdf_content:
                xml += s
                # after the line containing the "robot" tag, add the "mujoco" tag
                if(s.startswith("<robot ")):
                    xml += '<mujoco>\n'+ '<compiler meshdir="' + meshdir + \
                        '" balanceinertia="true" discardvisual="false"/>\n' +\
                        '</mujoco>\n'
            self.xml = xml
            self.create_basic_spec()
            # create the Mujoco model from the augmented URDF file 
            self.model = self.spec.compile()

        # self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.viz = viewer.launch_passive(model=self.model, data=self.data)
        
        self.model.opt.timestep = time_step
        self.viz.cam.distance = 3.0
        # OTHER CAM OPTIONS
        # self.viz.cam.azimuth       self.viz.cam.elevation     self.viz.cam.lookat        self.viz.cam.trackbodyid
        # self.viz.cam.distance      self.viz.cam.fixedcamid    self.viz.cam.orthographic  self.viz.cam.type
        self.sphere_name_to_id = {}

        J_ID = list(range(7, self.model.nq))
        self.xyzw2wxyz_id = MujocoSimulator.PIN_2_MJ_POS + J_ID
        self.wxyz2xyzw_id = MujocoSimulator.MJ_2_PIN_POS + J_ID


    def create_basic_spec(self):
        # alternative way to create model from XML passing through MjSpec, which 
        # should provide API for modifying model (e.g., adding lights)
        # see: https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/specs_test.py
        self.spec = mujoco.MjSpec()
        self.spec.from_string(self.xml)
        
        # modify model if needed
        # b = self.spec.worldbody.first_body()
        # for i in range(7):
        #     g = b.first_geom()
        #     g.name = b.name
        #     i += 1
        #     b = b.first_body()
        
        # Add an ambient light
        light = self.spec.worldbody.add_light()
        light.pos = np.array([0, 0, 4.0])
        light.diffuse = 0.8*np.ones(3)
        light.specular = 0.3*np.ones(3)
        light.cutoff = 30
        light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
        light.targetbody = self.spec.worldbody.first_body().name
        self.light = light


    def set_state(self, q, v):
        self.data.qpos = q
        self.data.qvel = v
        mujoco.mj_step1(self.model, self.data)

    
    def get_state(self):
        """
        Return MuJoCo state in (x, y, z, qx, qy, qz, qw) format, compatible with Pinocchio.
        
        Returns:
            q (NDArray[np.float64]): Joint position
            v (NDArray[np.float64], optional): Joint velocities
        """
        q = self.wxyz2xyzw(self.data.qpos)
        v = self.data.qvel
        return q, v
    

    def wxyz2xyzw(self, q_wxyz):
        """
        Convert MuJoCo to Pinocchio state format.
        Args:
            q_mj (NDArray[np.float64]): State in mj format.
        Returns:
            NDArray[np.float64]: State in pin format.
        """
        q_xyzw = np.take(q_wxyz, self.wxyz2xyzw_id, mode="clip")
        return q_xyzw
    

    def xyzw2wxyz(self, q_xyzw):
        """
        Convert Pinocchio to MuJoCo state format.
        Args:
            q (NDArray[np.float64]): State in pin format.
        Returns:
            NDArray[np.float64]: State in mj format.
        """
        q_wxyz = np.take(q_xyzw, self.xyzw2wxyz_id, mode="clip")
        return q_wxyz


    # Simulate and display video
    def step(self, u, dt=None, update_viewer=True):
        self.data.qfrc_applied = u
        if(dt is None):
            step_iter = 1
        else:
            step_iter = max(1, int(dt / self.model.opt.timestep))

        for i in range(step_iter):
            mujoco.mj_step(self.model, self.data)

        if(update_viewer):
            self.update_viewer()


    def update_viewer(self):
        self.viz.sync()


    def display(self, q):
        self.data.qpos = q
        mujoco.mj_step1(self.model, self.data)
        # mujoco.mjv_updateScene(self.model, self.data, self.viz.opt, self.viz._pert, self.viz.cam, 0, self.viz.user_scn)
        self.update_viewer()


    def display_motion(self, q_traj, dt, slow_down_factor=1):
        for i in range(q_traj.shape[0]):
            time_start = time.time()
            self.display(q_traj[i,:])
            time_spent = time.time() - time_start
            if(time_spent < slow_down_factor*dt):
                time.sleep(slow_down_factor*dt-time_spent)


    def add_visual_sphere(self, name, center, radius, rgba):
        """Adds a visual sphere to the scene."""
        scene = self.viz.user_scn
        if scene.ngeom >= scene.maxgeom:
            print("ERROR: Max number of geom in scene has been reached!")
            return
        self.sphere_name_to_id[name] = scene.ngeom
        scene.ngeom += 1  # increment ngeom
        # initialise a new sphere and add it to the scene
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_SPHERE, radius*np.ones(3),
                            center.astype(np.float32), np.eye(3).flatten(), rgba.astype(np.float32))


    def move_visual_sphere(self, name, pos):
        geom_id = self.sphere_name_to_id[name]
        scene = self.viz.user_scn
        scene.geoms[geom_id].pos = pos

    
    def add_visual_trajectory(self, name, pos_list, width, rgba):
        """Adds a list of geometric lines to the viewer scene."""
        scene = self.viz.user_scn
        for i in range(pos_list.shape[1]-1):
            if scene.ngeom >= scene.maxgeom:
                print("ERROR: Max number of geom in scene has been reached!")
                return
            scene.ngeom += 1  # increment ngeom
            # initialise geom-line and add it to the scene
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(), rgba.astype(np.float32))
            mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                                    mujoco.mjtGeom.mjGEOM_LINE, width,
                                    pos_list[0,i],   pos_list[1,i],   pos_list[2,i],
                                    pos_list[0,i+1], pos_list[1,i+1], pos_list[2,i+1])
        

    def add_visual_capsule(self, point1, point2, radius, rgba):
        """Adds a visual capsule to the scene."""
        scene = self.viz.user_scn
        if scene.ngeom >= scene.maxgeom:
            print("ERROR: Max number of geom in scene has been reached!")
            return
        scene.ngeom += 1  # increment ngeom
        # initialise a new capsule, add it to the scene using mjv_makeConnector
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(), rgba.astype(np.float32))
        mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                                mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                point1[0], point1[1], point1[2],
                                point2[0], point2[1], point2[2])
        

    def add_sphere(self, pos, size, rgba):
        # recreate the spec from XML to reset it
        self.xml = self.spec.to_xml()
        self.spec.from_string(self.xml)
        # add the new sphere
        geom = self.spec.worldbody.add_geom()
        geom.size = size
        geom.rgba = rgba
        geom.pos = pos

        # re-create the Mujoco model and data
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        
        # reload model in viewer
        self.viz._sim().load(self.model, self.data, "")

        # re-launch viewer
        # self.viz.close()
        # viewer_not_launched = True
        # counter = 0
        # while(viewer_not_launched and counter<5):
        #     try:
        #         time.sleep(0.1)
        #         self.viz = viewer.launch_passive(model=self.model, data=self.data)
        #         viewer_not_launched = False
        #     except Exception as e:
        #         counter += 1
        #         if(counter<5):
        #             print("Failed to launch viewer. Gonna try again in a little bit.")
        #         else:
        #             print("Giving up on launching viewer")
        #             raise e