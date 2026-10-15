from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mujoco


from src.config import DT
from src.config import WHEEL_RADIUS
from src.config import WHEEL_BASE

from src.model import robot_step
from src.model import wheel_speeds_to_control

from src.sensors import noisy_wheel_speeds

from src.ekf import PoseEKF

from src.vision import (
    aruco_measurement_to_robot_pose
)

from src.utils import wrap_angle


# ==================================================
# PROJECT / MUJOCO SETUP
# ==================================================

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


# ==================================================
# RANDOM NUMBER GENERATOR
# ==================================================

rng = np.random.default_rng(42)


# ==================================================
# MUJOCO JOINT ADDRESSES
# ==================================================

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


x_qpos = model.jnt_qposadr[
    x_joint_id
]

y_qpos = model.jnt_qposadr[
    y_joint_id
]

yaw_qpos = model.jnt_qposadr[
    yaw_joint_id
]


# ==================================================
# TRUE INITIAL ROBOT STATE
# ==================================================

true_state = np.array([
    0.0,
    0.0,
    0.0
])


# ==================================================
# TRUE ROBOT CONTROL
# ==================================================
#
# For this experiment the robot simply drives
# forward toward the docking station.
#

true_control = np.array([
    0.18,   # linear velocity [m/s]
    0.0     # angular velocity [rad/s]
])


# ==================================================
# EKF INITIALISATION
# ==================================================

initial_estimate = np.array([
    0.0,
    0.0,
    0.0
])


initial_covariance = np.diag([
    0.05**2,
    0.05**2,
    np.deg2rad(5.0)**2
])


process_noise_covariance = np.diag([
    0.002**2,
    0.002**2,
    np.deg2rad(0.5)**2
])


# Visual pose uncertainty.
#
# These are initial modelling values.
# Later we can tune them from repeated
# ArUco validation trials.

measurement_noise_covariance = np.diag([
    0.01**2,
    0.01**2,
    np.deg2rad(1.0)**2
])


ekf = PoseEKF(
    initial_estimate,
    initial_covariance,
    process_noise_covariance
)


# ==================================================
# ENCODER MODEL
# ==================================================

encoder_sigma = 0.15

# Small systematic bias added deliberately
# so that odometry drift becomes visible.

encoder_bias_left = 0.00
encoder_bias_right = 0.03


# ==================================================
# CAMERA RENDERING
# ==================================================

image_width = 640
image_height = 480


renderer = mujoco.Renderer(
    model,
    height=image_height,
    width=image_width
)


# ==================================================
# CAMERA INTRINSICS
# ==================================================
#
# MuJoCo front_camera has:
#
# fovy = 70 degrees
#

fovy_deg = 70.0

fovy_rad = np.deg2rad(
    fovy_deg
)


fy = (
    image_height / 2.0
) / np.tan(
    fovy_rad / 2.0
)

fx = fy

cx = image_width / 2.0
cy = image_height / 2.0


camera_matrix = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])


# Ideal MuJoCo camera:
# assume no lens distortion.

dist_coeffs = np.zeros(
    (5, 1)
)


# ==================================================
# ARUCO SETUP
# ==================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)


aruco_parameters = cv2.aruco.DetectorParameters()


aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


marker_id_required = 0

marker_length = 0.30

half_marker = (
    marker_length / 2.0
)


# Marker corner locations
# in its own coordinate system.

object_points = np.array([
    [-half_marker,  half_marker, 0.0],
    [ half_marker,  half_marker, 0.0],
    [ half_marker, -half_marker, 0.0],
    [-half_marker, -half_marker, 0.0]
], dtype=np.float32)


# Known marker position in MuJoCo world.

marker_world_position = np.array([
    1.755,
    0.0,
    0.35
])


# ==================================================
# SIMULATION SETTINGS
# ==================================================

simulation_time = 8.0

number_of_steps = int(
    simulation_time / DT
)


# Simulate a slower camera than the encoder.
#
# DT = 0.05 s → EKF prediction = 20 Hz
#
# camera every 5 steps → 4 Hz

camera_interval_steps = 5


# ==================================================
# HISTORY STORAGE
# ==================================================

time_history = []


true_x_history = []
true_y_history = []


ekf_x_history = []
ekf_y_history = []


position_error_history = []


camera_update_history = []


sigma_x_history = []
sigma_y_history = []
sigma_theta_history = []


camera_update_count = 0

camera_detection_count = 0


# Save the most recent annotated image
# for visual confirmation.

last_detection_image = None


# ==================================================
# MAIN SIMULATION LOOP
# ==================================================

for step in range(number_of_steps):

    # ==================================================
    # 1. MOVE TRUE ROBOT
    # ==================================================

    true_state = robot_step(
        true_state,
        true_control,
        DT
    )


    # ==================================================
    # 2. COPY TRUE POSE TO MUJOCO
    # ==================================================

    data.qpos[
        x_qpos
    ] = true_state[0]

    data.qpos[
        y_qpos
    ] = true_state[1]

    data.qpos[
        yaw_qpos
    ] = true_state[2]


    mujoco.mj_forward(
        model,
        data
    )


    # ==================================================
    # 3. TRUE CONTROL → TRUE WHEEL SPEEDS
    # ==================================================

    v_true = true_control[0]
    omega_true = true_control[1]


    omega_right_true = (
        v_true
        +
        (
            WHEEL_BASE / 2.0
        ) * omega_true
    ) / WHEEL_RADIUS


    omega_left_true = (
        v_true
        -
        (
            WHEEL_BASE / 2.0
        ) * omega_true
    ) / WHEEL_RADIUS


    # ==================================================
    # 4. SIMULATE NOISY ENCODERS
    # ==================================================

    measured_left, measured_right = (
        noisy_wheel_speeds(
            omega_left_true,
            omega_right_true,
            encoder_sigma,
            encoder_sigma,
            rng,
            encoder_bias_left,
            encoder_bias_right
        )
    )


    measured_control = (
        wheel_speeds_to_control(
            measured_left,
            measured_right,
            WHEEL_RADIUS,
            WHEEL_BASE
        )
    )


    # ==================================================
    # 5. EKF PREDICTION
    # ==================================================

    ekf.predict(
        measured_control,
        DT
    )


    # ==================================================
    # 6. CAMERA / ARUCO UPDATE
    # ==================================================

    camera_updated_this_step = False


    if step % camera_interval_steps == 0:

        # ---------------------------------
        # RENDER MUJOCO CAMERA
        # ---------------------------------

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
        # DETECT ARUCO
        # ---------------------------------

        corners, ids, rejected = (
            aruco_detector.detectMarkers(
                image_gray
            )
        )


        if ids is not None:

            ids_flat = np.asarray(
                ids
            ).reshape(-1)


            marker_index = None


            for i in range(
                len(ids_flat)
            ):

                if int(
                    ids_flat[i]
                ) == marker_id_required:

                    marker_index = i

                    break


            if marker_index is not None:

                camera_detection_count += 1


                image_points = corners[
                    marker_index
                ].reshape(
                    4,
                    2
                ).astype(
                    np.float32
                )


                # ---------------------------------
                # SOLVE PNP
                # ---------------------------------

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )


                if success:

                    rvec = rvec.reshape(
                        3
                    )

                    tvec = tvec.reshape(
                        3
                    )


                    # ---------------------------------
                    # CONVERT VISUAL MEASUREMENT
                    # INTO WORLD ROBOT POSE
                    # ---------------------------------

                    visual_robot_pose = (
                        aruco_measurement_to_robot_pose(
                            rvec,
                            tvec,
                            marker_world_position
                        )
                    )


                    # ---------------------------------
                    # EKF CAMERA CORRECTION
                    # ---------------------------------

                    ekf.update_pose(
                        visual_robot_pose,
                        measurement_noise_covariance
                    )


                    camera_update_count += 1

                    camera_updated_this_step = True


                    # ---------------------------------
                    # SAVE ANNOTATED IMAGE
                    # ---------------------------------

                    image_bgr = cv2.cvtColor(
                        image_rgb,
                        cv2.COLOR_RGB2BGR
                    )


                    ids_for_drawing = np.asarray(
                        ids
                    ).reshape(
                        -1,
                        1
                    )


                    cv2.aruco.drawDetectedMarkers(
                        image_bgr,
                        corners,
                        ids_for_drawing
                    )


                    cv2.drawFrameAxes(
                        image_bgr,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        0.10
                    )


                    last_detection_image = cv2.cvtColor(
                        image_bgr,
                        cv2.COLOR_BGR2RGB
                    )


    # ==================================================
    # 7. SAVE RESULTS
    # ==================================================

    current_time = (
        step + 1
    ) * DT


    time_history.append(
        current_time
    )


    true_x_history.append(
        true_state[0]
    )

    true_y_history.append(
        true_state[1]
    )


    ekf_x_history.append(
        ekf.state[0]
    )

    ekf_y_history.append(
        ekf.state[1]
    )


    position_error = np.sqrt(
        (
            ekf.state[0]
            -
            true_state[0]
        )**2
        +
        (
            ekf.state[1]
            -
            true_state[1]
        )**2
    )


    position_error_history.append(
        position_error
    )


    camera_update_history.append(
        int(
            camera_updated_this_step
        )
    )


    sigma_x_history.append(
        np.sqrt(
            ekf.P[0, 0]
        )
    )

    sigma_y_history.append(
        np.sqrt(
            ekf.P[1, 1]
        )
    )

    sigma_theta_history.append(
        np.sqrt(
            ekf.P[2, 2]
        )
    )


# ==================================================
# CONVERT HISTORIES TO ARRAYS
# ==================================================

time_history = np.array(
    time_history
)


position_error_history = np.array(
    position_error_history
)


camera_update_history = np.array(
    camera_update_history
)


# ==================================================
# METRICS
# ==================================================

position_rmse = np.sqrt(
    np.mean(
        position_error_history**2
    )
)


final_position_error = np.sqrt(
    (
        ekf.state[0]
        -
        true_state[0]
    )**2
    +
    (
        ekf.state[1]
        -
        true_state[1]
    )**2
)


final_heading_error = wrap_angle(
    ekf.state[2]
    -
    true_state[2]
)


# ==================================================
# PRINT RESULTS
# ==================================================

print("Camera detection count:")
print(camera_detection_count)

print()

print("Successful EKF visual updates:")
print(camera_update_count)

print()

print("Final true state:")
print(true_state)

print()

print("Final EKF state:")
print(ekf.state)

print()

print("Final position error:")
print(
    final_position_error,
    "m"
)

print()

print("Final heading error:")
print(
    np.degrees(
        final_heading_error
    ),
    "degrees"
)

print()

print("Position RMSE:")
print(
    position_rmse,
    "m"
)


# ==================================================
# FIGURE 1 — TRAJECTORY
# ==================================================

plt.figure()

plt.plot(
    true_x_history,
    true_y_history,
    label="True trajectory"
)

plt.plot(
    ekf_x_history,
    ekf_y_history,
    "--",
    label="EKF estimate"
)

plt.scatter(
    marker_world_position[0],
    marker_world_position[1],
    marker="s",
    label="ArUco marker"
)

plt.xlabel(
    "X position [m]"
)

plt.ylabel(
    "Y position [m]"
)

plt.title(
    "MuJoCo Vision-Aided EKF"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()


# ==================================================
# FIGURE 2 — POSITION ERROR
# ==================================================

plt.figure()

plt.plot(
    time_history,
    position_error_history
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Position error [m]"
)

plt.title(
    "Vision-Aided EKF Position Error"
)

plt.grid()


# ==================================================
# FIGURE 3 — VISUAL UPDATE EVENTS
# ==================================================

plt.figure()

plt.step(
    time_history,
    camera_update_history,
    where="post"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Visual EKF update"
)

plt.title(
    "ArUco Visual Update Events"
)

plt.yticks(
    [0, 1],
    ["No", "Yes"]
)

plt.grid()


# ==================================================
# FIGURE 4 — MOST RECENT ARUCO IMAGE
# ==================================================

if last_detection_image is not None:

    plt.figure()

    plt.imshow(
        last_detection_image
    )

    plt.title(
        "Latest MuJoCo ArUco Detection"
    )

    plt.axis(
        "off"
    )


plt.show()