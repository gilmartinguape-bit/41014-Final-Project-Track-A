from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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


# ---------------------------------
# PLACE ROBOT
# ---------------------------------

robot_x = 1.0
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
# CREATE OFFSCREEN RENDERER
# ---------------------------------

renderer = mujoco.Renderer(
    model,
    height=480,
    width=640
)


renderer.update_scene(
    data,
    camera="front_camera"
)


image = renderer.render()


print(
    "Rendered image shape:",
    image.shape
)


# ---------------------------------
# SHOW CAMERA IMAGE
# ---------------------------------

plt.figure()

plt.imshow(
    image
)

plt.title(
    "MuJoCo Front Camera"
)

plt.axis(
    "off"
)

plt.show()