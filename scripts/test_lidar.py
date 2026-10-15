from pathlib import Path
import numpy as np

import mujoco


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


# Put robot at origin, facing +X.
data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.0

mujoco.mj_forward(
    model,
    data
)


origin = np.array([
    0.0,
    0.0,
    0.16
])


direction = np.array([
    1.0,
    0.0,
    0.0
])


geomid = np.array(
    [-1],
    dtype=np.int32
)


distance = mujoco.mj_ray(
    model,
    data,
    origin,
    direction,
    None,
    1,
    -1,
    geomid
)


print("Forward LiDAR distance:")
print(distance, "m")

print()

print("Hit geometry ID:")
print(geomid[0])