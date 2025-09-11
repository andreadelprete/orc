if __name__=='__main__':
    import aliengo_conf as conf
    import matplotlib.pyplot as plt
    import numpy as np
    from robot_descriptions.loaders.mujoco import load_robot_description
    from orc.utils.mujoco_simulator import MujocoSimulator

    # READ COM-COP TRAJECTORIES COMPUTED WITH LIPM MODEL
    data = np.load(conf.DATA_FILE_CTRL, allow_pickle=True)
    com_ref     = data['com']
    dcom_ref    = data['dcom']
    ddcom_ref   = data['ddcom']
    x_feet_ref   = data['x_feet']
    dx_feet_ref  = data['dx_feet']
    ddx_feet_ref = data['ddx_feet']
    cop_ref      = data['cop']

    simu = MujocoSimulator(conf.mujoco_robot_name, conf.dt)
    simu.set_state(conf.robot.q0, np.zeros(conf.robot.nv))
    simu.update_viewer()

    for t in range(com_ref.shape[1]):
        simu.step(np.zeros(18))
        from time import sleep
        sleep(conf.dt)