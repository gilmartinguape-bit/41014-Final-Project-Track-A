from pathlib import Path

import cv2
import numpy as np
import mujoco

from src.vision import (
    aruco_measurement_to_robot_pose
)


project_root = Path(__file__).resolve().parents[1]

xml_path = (
    project_root
    / "assets"
    / "empty_world.xml"
)


# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

data = mujoco.MjData(
    model
)


# ---------------------------------
# KNOWN TRUE ROBOT POSE
# ---------------------------------

true_robot_pose = np.array([
    1.0,
    0.0,
    0.0
])


data.qpos[0] = true_robot_pose[0]
data.qpos[1] = true_robot_pose[1]
data.qpos[2] = true_robot_pose[2]


mujoco.mj_forward(
    model,
    data
)


# ---------------------------------
# CAMERA IMAGE
# ---------------------------------

image_width = 640
image_height = 480


renderer = mujoco.Renderer(
    model,
    height=image_height,
    width=image_width
)


renderer.update_scene(
    data,
    camera="front_camera"
)


image_rgb = renderer.render()


image_gray = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2GRAY
)


# ---------------------------------
# CAMERA INTRINSICS
# ---------------------------------

fovy = np.deg2rad(
    70.0
)


fy = (
    image_height / 2.0
) / np.tan(
    fovy / 2.0
)


fx = fy

cx = image_width / 2.0
cy = image_height / 2.0


camera_matrix = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])


dist_coeffs = np.zeros(
    (5, 1)
)


# ---------------------------------
# ARUCO DETECTION
# ---------------------------------

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)


parameters = cv2.aruco.DetectorParameters()


detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)


corners, ids, rejected = detector.detectMarkers(
    image_gray
)


if ids is None:

    print(
        "No ArUco marker detected."
    )

    raise SystemExit


ids_flat = np.asarray(
    ids
).reshape(-1)


print("Detected IDs:")
print(ids_flat)


# ---------------------------------
# MARKER GEOMETRY
# ---------------------------------

marker_length = 0.30

half_marker = marker_length / 2.0


object_points = np.array([
    [-half_marker,  half_marker, 0.0],
    [ half_marker,  half_marker, 0.0],
    [ half_marker, -half_marker, 0.0],
    [-half_marker, -half_marker, 0.0]
], dtype=np.float32)


# ---------------------------------
# KNOWN MARKER WORLD POSITION
# ---------------------------------

marker_world_position = np.array([
    1.755,
    0.0,
    0.35
])


# ---------------------------------
# USE MARKER ID 0
# ---------------------------------

marker_index = None


for i in range(
    len(ids_flat)
):

    if ids_flat[i] == 0:

        marker_index = i

        break


if marker_index is None:

    print(
        "Marker ID 0 not detected."
    )

    raise SystemExit


image_points = corners[
    marker_index
].reshape(
    4,
    2
).astype(
    np.float32
)


success, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_IPPE_SQUARE
)


if not success:

    print(
        "solvePnP failed."
    )

    raise SystemExit


rvec = rvec.reshape(3)

tvec = tvec.reshape(3)


# ---------------------------------
# CONVERT TO ROBOT WORLD POSE
# ---------------------------------

estimated_robot_pose = (
    aruco_measurement_to_robot_pose(
        rvec,
        tvec,
        marker_world_position
    )
)


# ---------------------------------
# ERRORS
# ---------------------------------

position_error = np.sqrt(
    (
        estimated_robot_pose[0]
        -
        true_robot_pose[0]
    )**2
    +
    (
        estimated_robot_pose[1]
        -
        true_robot_pose[1]
    )**2
)


heading_error = (
    estimated_robot_pose[2]
    -
    true_robot_pose[2]
)


heading_error = (
    heading_error + np.pi
) % (
    2.0 * np.pi
) - np.pi


# ---------------------------------
# PRINT RESULTS
# ---------------------------------

print()

print(
    "True robot pose [x, y, theta]:"
)

print(
    true_robot_pose
)


print()

print(
    "Vision-estimated robot pose:"
)

print(
    estimated_robot_pose
)


print()

print(
    "Position error:"
)

print(
    position_error,
    "m"
)


print()

print(
    "Heading error:"
)

print(
    np.degrees(
        heading_error
    ),
    "degrees"
)


print()

print(
    "solvePnP tvec:"
)

print(
    tvec
)