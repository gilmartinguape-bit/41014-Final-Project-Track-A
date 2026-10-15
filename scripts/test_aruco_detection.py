from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mujoco


project_root = Path(__file__).resolve().parents[1]

xml_path = (
    project_root
    / "assets"
    / "empty_world.xml"
)


# ---------------------------------
# LOAD MUJOCO MODEL
# ---------------------------------

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
# RENDER CAMERA IMAGE
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

image_rgb = renderer.render()

print("Rendered image shape:", image_rgb.shape)


# ---------------------------------
# CONVERT TO GRAYSCALE
# ---------------------------------

image_gray = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2GRAY
)


# ---------------------------------
# ARUCO DETECTOR SETUP
# ---------------------------------

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

detector_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    detector_parameters
)


# ---------------------------------
# DETECT MARKERS
# ---------------------------------

corners, ids, rejected = aruco_detector.detectMarkers(
    image_gray
)

print()
print("Rejected candidate count:")
print(len(rejected))


# ---------------------------------
# PRINT RESULTS
# ---------------------------------

if ids is None:

    print()
    print("No ArUco markers detected.")

else:

    print()
    print("Detected marker IDs:")
    print(ids.flatten())

    print()
    print("Number of detected markers:")
    print(len(ids))


# ---------------------------------
# DRAW DETECTIONS
# ---------------------------------

image_bgr = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2BGR
)

if ids is not None:

    cv2.aruco.drawDetectedMarkers(
        image_bgr,
        corners,
        ids
    )


image_display = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


plt.figure(figsize=(8, 6))
plt.imshow(image_display)
plt.title("ArUco Detection Result")
plt.axis("off")
plt.show()