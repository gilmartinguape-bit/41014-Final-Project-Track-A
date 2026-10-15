from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

from src.model import robot_step
from src.lidar import cast_lidar_ray


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
# JOINT ADDRESSES
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


x_qpos = model.jnt_qposadr[x_joint_id]
y_qpos = model.jnt_qposadr[y_joint_id]
yaw_qpos = model.jnt_qposadr[yaw_joint_id]


# ---------------------------------
# INITIAL ROBOT STATE
# ---------------------------------

state = np.array([
    0.0,
    0.0,
    0.0
])


control = np.array([
    0.12,   # linear speed
    0.0     # angular speed
])


control_dt = 0.05

lidar_height = 0.16
max_range = 3.5


# ---------------------------------
# VIEWER
# ---------------------------------

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        # Move robot
        state = robot_step(
            state,
            control,
            control_dt
        )


        # Update MuJoCo robot pose
        data.qpos[x_qpos] = state[0]
        data.qpos[y_qpos] = state[1]
        data.qpos[yaw_qpos] = state[2]


        mujoco.mj_forward(
            model,
            data
        )


        # ---------------------------------
        # LIDAR ORIGIN
        # ---------------------------------

        origin = np.array([
            state[0],
            state[1],
            lidar_height
        ])


        left_angle = (
            state[2]
            + np.deg2rad(20.0)
        )

        front_angle = state[2]

        right_angle = (
            state[2]
            - np.deg2rad(20.0)
        )


        # ---------------------------------
        # CAST LIDAR RAYS
        # ---------------------------------

        left_distance, _ = cast_lidar_ray(
            model,
            data,
            origin,
            left_angle,
            max_range
        )

        front_distance, _ = cast_lidar_ray(
            model,
            data,
            origin,
            front_angle,
            max_range
        )

        right_distance, _ = cast_lidar_ray(
            model,
            data,
            origin,
            right_angle,
            max_range
        )


        # Print approximately once per second
        if step % 20 == 0:

            print(
                f"x = {state[0]:.2f} m | "
                f"Left = {left_distance:.2f} m | "
                f"Front = {front_distance:.2f} m | "
                f"Right = {right_distance:.2f} m"
            )


        viewer.sync()

        time.sleep(
            control_dt
        )

        step += 1


        if step * control_dt > 12.0:
            break