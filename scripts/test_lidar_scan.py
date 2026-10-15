from pathlib import Path

import numpy as np
import mujoco

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

def geometry_name(
    model,
    geom_id
):
    if geom_id < 0:
        return "No hit"

    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id
    )

# ---------------------------------
# ROBOT POSE
# ---------------------------------

robot_x = 0.0
robot_y = 0.0
robot_yaw = 0.0


data.qpos[0] = robot_x
data.qpos[1] = robot_y
data.qpos[2] = robot_yaw


mujoco.mj_forward(
    model,
    data
)


# ---------------------------------
# LIDAR SETTINGS
# ---------------------------------

lidar_height = 0.16

max_range = 3.5


origin = np.array([
    robot_x,
    robot_y,
    lidar_height
])


left_angle = (
    robot_yaw
    + np.deg2rad(20.0)
)

front_angle = (
    robot_yaw
)

right_angle = (
    robot_yaw
    - np.deg2rad(20.0)
)


# ---------------------------------
# CAST RAYS
# ---------------------------------

left_distance, left_geom = cast_lidar_ray(
    model,
    data,
    origin,
    left_angle,
    max_range
)


front_distance, front_geom = cast_lidar_ray(
    model,
    data,
    origin,
    front_angle,
    max_range
)


right_distance, right_geom = cast_lidar_ray(
    model,
    data,
    origin,
    right_angle,
    max_range
)


# ---------------------------------
# PRINT RESULTS
# ---------------------------------

print("Left-front LiDAR:")
print(left_distance, "m")
print("Geometry ID:", left_geom)
print("Geometry name:", geometry_name(model, left_geom))

print()

print("Front LiDAR:")
print(front_distance, "m")
print("Geometry ID:", front_geom)
print("Geometry name:", geometry_name(model, front_geom))
print()

print("Right-front LiDAR:")
print(right_distance, "m")
print("Geometry ID:", right_geom)
print("Geometry name:", geometry_name(model, right_geom))