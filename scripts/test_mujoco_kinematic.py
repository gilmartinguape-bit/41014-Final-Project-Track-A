from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

from src.model import robot_step


project_root = Path(__file__).resolve().parents[1]

xml_path = (
    project_root
    / "assets"
    / "empty_world.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

data = mujoco.MjData(
    model
)


# ---------------------------------
# FIND JOINT POSITION ADDRESSES
# ---------------------------------

x_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "robot_x"
)

y_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "robot_y"
)

yaw_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "robot_yaw"
)


x_qpos_address = model.jnt_qposadr[
    x_joint_id
]

y_qpos_address = model.jnt_qposadr[
    y_joint_id
]

yaw_qpos_address = model.jnt_qposadr[
    yaw_joint_id
]


print("X qpos address:", x_qpos_address)
print("Y qpos address:", y_qpos_address)
print("Yaw qpos address:", yaw_qpos_address)


# ---------------------------------
# INITIAL ROBOT STATE
# ---------------------------------

state = np.array([
    0.0,
    0.0,
    0.0
])


control = np.array([
    0.18,
    0.30
])


control_dt = 0.05


# ---------------------------------
# VIEWER
# ---------------------------------

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    start_time = time.time()

    while viewer.is_running():

        # Kinematic robot model
        state = robot_step(
            state,
            control,
            control_dt
        )


        # Copy the model state into MuJoCo
        data.qpos[
            x_qpos_address
        ] = state[0]

        data.qpos[
            y_qpos_address
        ] = state[1]

        data.qpos[
            yaw_qpos_address
        ] = state[2]


        # Tell MuJoCo to update geometry positions
        mujoco.mj_forward(
            model,
            data
        )


        viewer.sync()


        time.sleep(
            control_dt
        )


        if time.time() - start_time > 10.0:
            break