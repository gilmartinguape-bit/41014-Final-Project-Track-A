import time

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer


from src.config import DT
from src.config import WHEEL_RADIUS
from src.config import WHEEL_BASE

from src.model import robot_step
from src.model import wheel_speeds_to_control

from src.sensors import noisy_wheel_speeds

from src.ekf import PoseEKF

from src.controller import design_lqr
from src.controller import lqr_control
from src.controller import docking_forward_speed
from src.controller import docking_pose_control

from src.lidar import cast_lidar_ray
from src.lidar import lidar_safety_state_machine

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
    / "obstacle_world.xml"
)


model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

data = mujoco.MjData(
    model
)


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
    0.25,
    np.deg2rad(8.0)
])


# ==================================================
# EKF INITIAL STATE
# ==================================================

initial_estimate = np.array([
    0.0,
    0.20,
    np.deg2rad(6.0)
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
# LQR
# ==================================================

max_forward_speed = 0.18

q_y = 8.0
q_heading = 4.0
r_omega = 1.2

omega_limit = 1.5


A, B, K = design_lqr(
    max_forward_speed,
    q_y,
    q_heading,
    r_omega
)


# ==================================================
# DOCKING TARGET
# ==================================================

target_pose = np.array([
    1.20,
    0.0,
    0.0
])

target_x = target_pose[0]

final_docking_start_x = 0.95

slow_distance = 0.50

stop_tolerance = 0.03


# ==================================================
# ENCODER MODEL
# ==================================================

encoder_sigma = 0.15

encoder_bias_left = 0.00
encoder_bias_right = 0.02


# ==================================================
# CAMERA
# ==================================================

image_width = 640
image_height = 480


renderer = mujoco.Renderer(
    model,
    height=image_height,
    width=image_width
)


fovy_deg = 90.0

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


dist_coeffs = np.zeros(
    (5, 1)
)


# ==================================================
# ARUCO
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


object_points = np.array([
    [-half_marker,  half_marker, 0.0],
    [ half_marker,  half_marker, 0.0],
    [ half_marker, -half_marker, 0.0],
    [-half_marker, -half_marker, 0.0]
], dtype=np.float32)


marker_world_position = np.array([
    1.755,
    0.0,
    0.35
])


camera_interval_steps = 5


# ==================================================
# LIDAR
# ==================================================

lidar_height = 0.16

lidar_max_range = 3.5

slow_obstacle_distance = 0.65

avoidance_enter_distance = 0.30

avoidance_exit_distance = 0.60

avoidance_turn_rate = 0.8


recovery_speed = 0.10

recovery_steps = int(
    4.5 / DT
)


# ==================================================
# SIMULATION
# ==================================================

simulation_time = 20.0

number_of_steps = int(
    simulation_time / DT
)

# ==================================================
# VIEWER PLAYBACK SETTINGS
# ==================================================

# 1.0 = approximately real time
# 0.5 = half speed
# 2.0 = twice real time

playback_speed = 0.5

viewer_step_time = (
    DT / playback_speed
)


# ==================================================
# HISTORIES
# ==================================================

time_history = []

true_x_history = []
true_y_history = []

ekf_x_history = []
ekf_y_history = []

estimated_lateral_error_history = []
true_lateral_error_history = []

estimated_heading_error_history = []
true_heading_error_history = []

position_error_history = []

command_v_history = []
command_omega_history = []

front_lidar_history = []

camera_update_history = []

camera_update_count = 0

last_detection_image = None


# ==================================================
# INITIAL COMMAND
# ==================================================

control_command = np.array([
    0.0,
    0.0
])

avoidance_mode = "NORMAL"

avoidance_turn_direction = 0.0

recovery_steps_remaining = 0

avoidance_steps_remaining = 0

minimum_avoidance_steps = int(
    0.75 / DT
)


# ==================================================
# VIEWER
# ==================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    previous_avoidance_mode = (
    avoidance_mode
    )

    # ==================================================
    # MAIN LOOP
    # ==================================================

    for step in range(number_of_steps):

        step_start_time = time.time()

        if not viewer.is_running():
            break


        # ==============================================
        # 1. MOVE TRUE ROBOT USING PREVIOUS COMMAND
        # ==============================================

        true_state = robot_step(
            true_state,
            control_command,
            DT
        )


        # ==============================================
        # 2. COPY TRUE STATE INTO MUJOCO
        # ==============================================

        data.qpos[x_qpos] = true_state[0]
        data.qpos[y_qpos] = true_state[1]
        data.qpos[yaw_qpos] = true_state[2]


        mujoco.mj_forward(
            model,
            data
        )


        # ==============================================
        # 3. TRUE COMMAND → WHEEL SPEEDS
        # ==============================================

        v_true = control_command[0]
        omega_true = control_command[1]


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


        # ==============================================
        # 4. NOISY ENCODER MEASUREMENTS
        # ==============================================

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


        measured_control = wheel_speeds_to_control(
            measured_left,
            measured_right,
            WHEEL_RADIUS,
            WHEEL_BASE
        )


        # ==============================================
        # 5. EKF PREDICTION
        # ==============================================

        ekf.predict(
            measured_control,
            DT
        )


        # ==============================================
        # 6. CAMERA / ARUCO UPDATE
        # ==============================================

        camera_updated = False


        if step % camera_interval_steps == 0:

            renderer.update_scene(
                data,
                camera="front_camera"
            )


            image_rgb = renderer.render()


            image_gray = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2GRAY
            )


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


                    if success:

                        rvec = rvec.reshape(3)
                        tvec = tvec.reshape(3)


                        visual_robot_pose = (
                            aruco_measurement_to_robot_pose(
                                rvec,
                                tvec,
                                marker_world_position
                            )
                        )


                        ekf.update_pose(
                            visual_robot_pose,
                            measurement_noise_covariance
                        )


                        camera_update_count += 1

                        camera_updated = True


                        image_bgr = cv2.cvtColor(
                            image_rgb,
                            cv2.COLOR_RGB2BGR
                        )


                        ids_draw = np.asarray(
                            ids
                        ).reshape(
                            -1,
                            1
                        )


                        cv2.aruco.drawDetectedMarkers(
                            image_bgr,
                            corners,
                            ids_draw
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


        # ==============================================
        # 7. CONTROLLER
        #
        # Controller uses EKF estimate only.
        # ==============================================

        if ekf.state[0] < final_docking_start_x:

            # ------------------------------------------
            # PHASE 1:
            # LQR APPROACH
            # ------------------------------------------

            forward_speed_command = (
                docking_forward_speed(
                    ekf.state[0],
                    target_x,
                    max_forward_speed,
                    slow_distance,
                    stop_tolerance
                )
            )


            lqr_command, estimated_lateral_error, estimated_heading_error = (
                lqr_control(
                    ekf.state,
                    K,
                    forward_speed_command,
                    omega_limit
                )
            )


            nominal_control_command = (
                lqr_command
            )

            controller_mode = "LQR_APPROACH"


        else:

            # ------------------------------------------
            # PHASE 2:
            # FINAL POSE DOCKING
            # ------------------------------------------

            (
                pose_command,
                docking_distance_error,
                docking_heading_error
            ) = docking_pose_control(
                ekf.state,
                target_pose
            )


            nominal_control_command = (
                pose_command
            )


            estimated_lateral_error = (
                ekf.state[1]
                - target_pose[1]
            )


            estimated_heading_error = wrap_angle(
                ekf.state[2]
                - target_pose[2]
            )


            controller_mode = "FINAL_DOCK"


        # ==============================================
        # 8. LIDAR
        # ==============================================

        lidar_origin = np.array([
            true_state[0],
            true_state[1],
            lidar_height
        ])


        left_angle = (
            true_state[2]
            +
            np.deg2rad(20.0)
        )


        front_angle = true_state[2]


        right_angle = (
            true_state[2]
            -
            np.deg2rad(20.0)
        )


        left_distance, _ = cast_lidar_ray(
            model,
            data,
            lidar_origin,
            left_angle,
            lidar_max_range
        )


        front_distance, _ = cast_lidar_ray(
            model,
            data,
            lidar_origin,
            front_angle,
            lidar_max_range
        )


        right_distance, _ = cast_lidar_ray(
            model,
            data,
            lidar_origin,
            right_angle,
            lidar_max_range
        )


        # ==============================================
        # 9. LIDAR SAFETY OVERRIDE
        # ==============================================

        (
            control_command,
            avoidance_mode,
            avoidance_turn_direction,
            recovery_steps_remaining,
            avoidance_steps_remaining
        ) = lidar_safety_state_machine(
            nominal_control_command[0],
            nominal_control_command[1],
            left_distance,
            front_distance,
            right_distance,
            avoidance_mode,
            avoidance_turn_direction,
            recovery_steps_remaining,
            avoidance_steps_remaining,
            slow_obstacle_distance,
            avoidance_enter_distance,
            avoidance_exit_distance,
            avoidance_turn_rate,
            recovery_speed,
            recovery_steps,
            minimum_avoidance_steps
        )


        if avoidance_mode != previous_avoidance_mode:

            print(
                f"AVOIDANCE MODE: "
                f"{previous_avoidance_mode}"
                f" -> "
                f"{avoidance_mode}"
        )

        previous_avoidance_mode = (
            avoidance_mode
        )


        # ==============================================
        # 10. TRUE EVALUATION ERRORS
        # ==============================================

        true_lateral_error = (
            true_state[1]
        )


        true_heading_error = wrap_angle(
            true_state[2]
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


        # ==============================================
        # 11. SAVE RESULTS
        # ==============================================

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


        estimated_lateral_error_history.append(
            estimated_lateral_error
        )

        true_lateral_error_history.append(
            true_lateral_error
        )


        estimated_heading_error_history.append(
            estimated_heading_error
        )

        true_heading_error_history.append(
            true_heading_error
        )


        position_error_history.append(
            position_error
        )


        command_v_history.append(
            control_command[0]
        )

        command_omega_history.append(
            control_command[1]
        )


        front_lidar_history.append(
            front_distance
        )


        camera_update_history.append(
            int(
                camera_updated
            )
        )

        if step % int(
            1.0 / DT
        ) == 0:

            print(
                f"Simulation time: "
                f"{step * DT:.1f} s | "
                f"x={true_state[0]:.2f} m | "
                f"y={true_state[1]:.2f} m | "
                f"heading="
                f"{np.degrees(true_state[2]):.1f} deg | "
                f"mode={avoidance_mode} | "
                f"controller={controller_mode} | "
                f"front={front_distance:.2f} m"
            )


        # ==============================================
        # 12. VIEWER
        # ==============================================

        viewer.sync()


        # ==============================================
        # REAL-TIME / SLOW-MOTION PLAYBACK
        # ==============================================

        processing_time = (
            time.time()
            - step_start_time
        )


        sleep_time = (
            viewer_step_time
            -
            processing_time
        )


        if sleep_time > 0.0:

            time.sleep(
                sleep_time
        )


    # ==================================================
    # HOLD FINAL ROBOT POSE FOR VIEWING
    # ==================================================

    print()
    print("Simulation complete.")
    print("Holding final MuJoCo view for 5 seconds...")


    hold_start = time.time()


    while (
        viewer.is_running()
        and
        time.time() - hold_start < 5.0
    ):

        viewer.sync()

        time.sleep(
            0.05
        )

# ==================================================
# CONVERT ARRAYS
# ==================================================

time_history = np.array(
    time_history
)

position_error_history = np.array(
    position_error_history
)

estimated_lateral_error_history = np.array(
    estimated_lateral_error_history
)

true_lateral_error_history = np.array(
    true_lateral_error_history
)

estimated_heading_error_history = np.array(
    estimated_heading_error_history
)

true_heading_error_history = np.array(
    true_heading_error_history
)


# ==================================================
# METRICS
# ==================================================

position_rmse = np.sqrt(
    np.mean(
        position_error_history**2
    )
)


true_lateral_rmse = np.sqrt(
    np.mean(
        true_lateral_error_history**2
    )
)


true_heading_rmse = np.sqrt(
    np.mean(
        true_heading_error_history**2
    )
)


final_position_error = position_error_history[
    -1
]


final_true_lateral_error = (
    true_state[1]
)


final_true_heading_error = wrap_angle(
    true_state[2]
)

final_x_docking_error = (
    true_state[0]
    - target_x
)

final_y_docking_error = (
    true_state[1]
)

final_planar_docking_error = np.sqrt(
    final_x_docking_error**2
    +
    final_y_docking_error**2
)


# ==================================================
# PRINT
# ==================================================

print("Camera update count:")
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

print("Position RMSE:")
print(
    position_rmse,
    "m"
)

print()

print("True lateral RMSE:")
print(
    true_lateral_rmse,
    "m"
)

print()

print("True heading RMSE:")
print(
    np.degrees(
        true_heading_rmse
    ),
    "degrees"
)

print()

print("Final true lateral error:")
print(
    final_true_lateral_error,
    "m"
)

print()

print("Final true heading error:")
print(
    np.degrees(
        final_true_heading_error
    ),
    "degrees"
)

print()

print("Final X docking error:")
print(
    final_x_docking_error,
    "m"
)

print()

print("Final Y docking error:")
print(
    final_y_docking_error,
    "m"
)

print()

print("Final planar docking error:")
print(
    final_planar_docking_error,
    "m"
)

# ==================================================
# PLOTS
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

plt.axhline(
    y=0.0,
    linestyle=":",
    label="Desired docking line"
)

plt.scatter(
    target_x,
    0.0,
    marker="x",
    s=100,
    label="Dock target"
)

plt.xlabel(
    "X position [m]"
)

plt.ylabel(
    "Y position [m]"
)

plt.title(
    "Full MuJoCo EKF + LQR + Pose-Control Docking"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()


# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    estimated_lateral_error_history,
    label="Estimated lateral error"
)

plt.plot(
    time_history,
    true_lateral_error_history,
    "--",
    label="True lateral error"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Lateral error [m]"
)

plt.title(
    "Estimated vs True Lateral Error"
)

plt.grid()

plt.legend()


# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    np.degrees(
        estimated_heading_error_history
    ),
    label="Estimated heading error"
)

plt.plot(
    time_history,
    np.degrees(
        true_heading_error_history
    ),
    "--",
    label="True heading error"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Heading error [deg]"
)

plt.title(
    "Estimated vs True Heading Error"
)

plt.grid()

plt.legend()


# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    command_v_history,
    label="Linear velocity"
)

plt.plot(
    time_history,
    command_omega_history,
    label="Angular velocity"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Command"
)

plt.title(
    "Robot Control Commands"
)

plt.grid()

plt.legend()


# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    front_lidar_history
)

plt.axhline(
    y=slow_obstacle_distance,
    linestyle="--",
    label="Slow threshold"
)

plt.axhline(
    y=avoidance_enter_distance,
    linestyle=":",
    label="Avoidance enter threshold"
)

plt.axhline(
    y=avoidance_exit_distance,
    linestyle="-.",
    label="Avoidance exit threshold"
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Front LiDAR distance [m]"
)

plt.title(
    "Front LiDAR Clearance"
)

plt.grid()

plt.legend()


# --------------------------------------------------

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
    "ArUco Update Events"
)

plt.yticks(
    [0, 1],
    ["No", "Yes"]
)

plt.grid()


# --------------------------------------------------

if last_detection_image is not None:

    plt.figure()

    plt.imshow(
        last_detection_image
    )

    plt.title(
        "Latest ArUco Detection"
    )

    plt.axis(
        "off"
    )


plt.show()