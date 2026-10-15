import numpy as np
import cv2

from src.utils import wrap_angle


def make_transform(
    rotation_matrix,
    translation_vector
):
    """
    Create a 4x4 homogeneous transformation matrix.
    """

    T = np.eye(4)

    T[0:3, 0:3] = rotation_matrix

    T[0:3, 3] = translation_vector

    return T


def invert_transform(T):
    """
    Invert a rigid-body homogeneous transformation.
    """

    R = T[0:3, 0:3]
    t = T[0:3, 3]

    T_inverse = np.eye(4)

    T_inverse[0:3, 0:3] = R.T

    T_inverse[0:3, 3] = -R.T @ t

    return T_inverse


def aruco_measurement_to_robot_pose(
    rvec,
    tvec,
    marker_world_position
):
    """
    Convert an OpenCV ArUco pose measurement
    into planar robot world pose [x, y, theta].

    Assumptions:
    - marker is mounted vertically facing toward -world X
    - camera mounting matches the MuJoCo front_camera
    """

    # ---------------------------------
    # 1. MARKER -> CAMERA
    # ---------------------------------

    R_camera_marker, _ = cv2.Rodrigues(
        rvec
    )

    T_camera_marker = make_transform(
        R_camera_marker,
        tvec
    )


    # ---------------------------------
    # 2. MARKER POSE IN WORLD FRAME
    # ---------------------------------
    #
    # Marker axes:
    #
    # +x_marker -> world -Y
    # +y_marker -> world +Z
    # +z_marker -> world -X
    #

    R_world_marker = np.array([
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
        [0.0,  1.0,  0.0]
    ])


    T_world_marker = make_transform(
        R_world_marker,
        marker_world_position
    )


    # ---------------------------------
    # 3. CAMERA POSE IN WORLD
    # ---------------------------------

    T_marker_camera = invert_transform(
        T_camera_marker
    )

    T_world_camera = (
        T_world_marker
        @ T_marker_camera
    )


    # ---------------------------------
    # 4. CAMERA POSE RELATIVE TO ROBOT
    # ---------------------------------
    #
    # OpenCV camera coordinates:
    #
    # +x = image right
    # +y = image down
    # +z = camera forward
    #
    # Relative to robot base:
    #
    # +x_cv -> -Y_robot
    # +y_cv -> -Z_robot
    # +z_cv -> +X_robot
    #

    R_robot_camera = np.array([
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])


    camera_position_robot = np.array([
        0.09,
        0.0,
        0.14
    ])


    T_robot_camera = make_transform(
        R_robot_camera,
        camera_position_robot
    )


    # ---------------------------------
    # 5. ROBOT POSE IN WORLD
    # ---------------------------------

    T_camera_robot = invert_transform(
        T_robot_camera
    )

    T_world_robot = (
        T_world_camera
        @ T_camera_robot
    )


    robot_x = T_world_robot[0, 3]
    robot_y = T_world_robot[1, 3]


    R_world_robot = T_world_robot[
        0:3,
        0:3
    ]


    robot_yaw = np.arctan2(
        R_world_robot[1, 0],
        R_world_robot[0, 0]
    )


    robot_yaw = wrap_angle(
        robot_yaw
    )


    robot_pose = np.array([
        robot_x,
        robot_y,
        robot_yaw
    ])


    return robot_pose