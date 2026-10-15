from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

from src.model import robot_step
from src.lidar import cast_lidar_ray
from src.lidar import lidar_safety_override


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
# INITIAL STATE
# ---------------------------------

state = np.array([
    0.0,
    0.0,
    0.0
])


desired_control = np.array([
    0.18,
    0.0
])


control_dt = 0.05


# ---------------------------------
# LIDAR SETTINGS
# ---------------------------------

lidar_height = 0.16

max_range = 3.5

slow_distance = 0.80

stop_distance = 0.35

avoidance_turn_rate = 0.8


# ---------------------------------
# VIEWER
# ---------------------------------

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    step = 0

    while viewer.is_running():

        # ---------------------------------
        # UPDATE ROBOT POSE IN MUJOCO
        # ---------------------------------

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
        # LIDAR READINGS
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


        # ---------------------------------
        # SAFETY OVERRIDE
        # ---------------------------------

        safe_control = lidar_safety_override(
            desired_control[0],
            desired_control[1],
            left_distance,
            front_distance,
            right_distance,
            slow_distance,
            stop_distance,
            avoidance_turn_rate
        )


        # ---------------------------------
        # MOVE ROBOT
        # ---------------------------------

        state = robot_step(
            state,
            safe_control,
            control_dt
        )


        # ---------------------------------
        # TERMINAL OUTPUT
        # ---------------------------------

        if step % 20 == 0:

            print(
                f"x={state[0]:.2f}, "
                f"y={state[1]:.2f}, "
                f"theta={np.degrees(state[2]):.1f} deg | "
                f"L={left_distance:.2f}, "
                f"F={front_distance:.4f}, "
                f"R={right_distance:.2f} | "
                f"v={safe_control[0]:.2f}, "
                f"omega={safe_control[1]:.2f}"
            )


        viewer.sync()

        time.sleep(
            control_dt
        )


        step += 1


        if step * control_dt > 20.0:
            break