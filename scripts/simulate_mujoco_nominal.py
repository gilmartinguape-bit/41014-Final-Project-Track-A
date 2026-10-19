import csv

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

from src.vision import (
    aruco_measurement_to_robot_pose
)

from src.utils import wrap_angle


# ==================================================
# PROJECT / MUJOCO SETUP
# ==================================================

project_root = Path(__file__).resolve().parents[1]

scenario_name = "nominal"

results_dir = (
    project_root
    / "results"
    / scenario_name
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

xml_path = (
    project_root
    / "assets"
    / "nominal_world.xml"
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
# VIDEO RECORDING
# ==================================================

video_width = 640
video_height = 480

video_renderer = mujoco.Renderer(
    model,
    height=video_height,
    width=video_width
)

video_path = (
    results_dir
    / "simulation.mp4"
)

video_fps = int(
    round(
        1.0 / DT
    )
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    str(video_path),
    fourcc,
    video_fps,
    (
        video_width,
        video_height
    )
)

if not video_writer.isOpened():
    raise RuntimeError(
        f"Could not open overview video writer for {video_path}"
    )

# ==================================================
# ROBOT FRONT-CAMERA VIDEO
# ==================================================

robot_camera_video_path = (
    results_dir
    / "robot_camera.mp4"
)

robot_camera_writer = cv2.VideoWriter(
    str(robot_camera_video_path),
    fourcc,
    video_fps,
    (
        image_width,
        image_height
    )
)

if not robot_camera_writer.isOpened():
    raise RuntimeError(
        f"Could not open robot-camera video writer for "
        f"{robot_camera_video_path}"
    )

robot_camera_renderer = mujoco.Renderer(
    model,
    height=image_height,
    width=image_width
)

# ==================================================
# BIRD'S-EYE PERSPECTIVE VIDEO
# ==================================================

birdseye_video_path = (
    results_dir
    / "birdseye_perspective.mp4"
)

birdseye_writer = cv2.VideoWriter(
    str(birdseye_video_path),
    fourcc,
    video_fps,
    (
        video_width,
        video_height
    )
)

if not birdseye_writer.isOpened():
    raise RuntimeError(
        f"Could not open bird's-eye video writer for "
        f"{birdseye_video_path}"
    )


# ==================================================
# TOP-VIEW VIDEO
# ==================================================

top_view_video_path = (
    results_dir
    / "top_view.mp4"
)

top_view_writer = cv2.VideoWriter(
    str(top_view_video_path),
    fourcc,
    video_fps,
    (
        video_width,
        video_height
    )
)

if not top_view_writer.isOpened():
    raise RuntimeError(
        f"Could not open top-view video writer for "
        f"{top_view_video_path}"
    )

birdseye_renderer = mujoco.Renderer(
    model,
    height=video_height,
    width=video_width
)

top_view_renderer = mujoco.Renderer(
    model,
    height=video_height,
    width=video_width
)

# ==================================================
# FOUR-VIEW GRID VIDEO
# ==================================================

grid_video_path = (
    results_dir
    / "four_view_grid.mp4"
)

status_bar_height = 100

grid_width = (
    video_width * 2
)

grid_height = (
    video_height * 2
    + status_bar_height
)

grid_writer = cv2.VideoWriter(
    str(grid_video_path),
    fourcc,
    video_fps,
    (
        grid_width,
        grid_height
    )
)

if not grid_writer.isOpened():
    raise RuntimeError(
        f"Could not open four-view grid video writer for "
        f"{grid_video_path}"
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
# OVERVIEW CAMERA
# ==================================================

overview_camera = mujoco.MjvCamera()

overview_camera.type = (
    mujoco.mjtCamera.mjCAMERA_FREE
)

overview_camera.lookat[:] = [
    0.65,
    0.0,
    0.0
]

overview_camera.distance = 2.7
overview_camera.azimuth = 90
overview_camera.elevation = -55


# ==================================================
# BIRD'S-EYE PERSPECTIVE CAMERA
# ==================================================

birdseye_camera = mujoco.MjvCamera()

birdseye_camera.type = (
    mujoco.mjtCamera.mjCAMERA_FREE
)

birdseye_camera.lookat[:] = [
    0.70,
    0.05,
    0.0
]

birdseye_camera.distance = 3.0
birdseye_camera.azimuth = 135
birdseye_camera.elevation = -65


# ==================================================
# TOP-DOWN CAMERA
# ==================================================

top_view_camera = mujoco.MjvCamera()

top_view_camera.type = (
    mujoco.mjtCamera.mjCAMERA_FREE
)

top_view_camera.lookat[:] = [
    0.70,
    0.05,
    0.0
]

top_view_camera.distance = 3.0
top_view_camera.azimuth = 90
top_view_camera.elevation = -89.0

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

aruco_visible = False

controller_mode_history = []


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

def add_video_label(
    frame,
    text
):

    labelled_frame = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_width = text_size[0]
    text_height = text_size[1]

    x = 15
    y = 35

    cv2.rectangle(
        labelled_frame,
        (
            x - 8,
            y - text_height - 8
        ),
        (
            x + text_width + 8,
            y + baseline + 8
        ),
        (
            0,
            0,
            0
        ),
        -1
    )

    cv2.putText(
        labelled_frame,
        text,
        (
            x,
            y
        ),
        font,
        font_scale,
        (
            255,
            255,
            255
        ),
        thickness,
        cv2.LINE_AA
    )

    return labelled_frame

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
        # OVERVIEW
        # ==============================================
        
        video_renderer.update_scene(
            data,
            camera=overview_camera
        )

        video_frame_rgb = (
            video_renderer.render()
        )

        video_frame_bgr = cv2.cvtColor(
            video_frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        video_writer.write(
            video_frame_bgr
        )

        # ==============================================
        # ROBOT CAMERA
        # ==============================================

        robot_camera_renderer.update_scene(
            data,
            camera="front_camera"
        )

        robot_camera_frame_rgb = (
            robot_camera_renderer.render()
        )

        robot_camera_frame_bgr = cv2.cvtColor(
            robot_camera_frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        robot_camera_writer.write(
            robot_camera_frame_bgr
        )

        # ==============================================
        # BIRD'S EYE VIEW
        # ==============================================

        birdseye_renderer.update_scene(
            data,
            camera=birdseye_camera
        )

        birdseye_frame_rgb = (
            birdseye_renderer.render()
        )

        birdseye_frame_bgr = cv2.cvtColor(
            birdseye_frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        birdseye_writer.write(
            birdseye_frame_bgr
        )

        # ==============================================
        # TOP VIEW
        # ==============================================

        top_view_renderer.update_scene(
            data,
            camera=top_view_camera
        )

        top_view_frame_rgb = (
            top_view_renderer.render()
        )

        top_view_frame_bgr = cv2.cvtColor(
            top_view_frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        top_view_writer.write(
            top_view_frame_bgr
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

            aruco_visible = False

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

                    aruco_visible = True

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
        # 9. NOMINAL CONTROL COMMAND
        # ==============================================

        control_command = (
            nominal_control_command
        )

        avoidance_mode = "NOT_APPLICABLE"


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


        controller_mode_history.append(
            controller_mode
        )

        overview_grid_frame = add_video_label(
            video_frame_bgr,
            "OVERVIEW"
        )

        robot_grid_frame = add_video_label(
            robot_camera_frame_bgr,
            "ROBOT CAMERA"
        )

        birdseye_grid_frame = add_video_label(
            birdseye_frame_bgr,
            "BIRD'S-EYE PERSPECTIVE"
        )

        top_grid_frame = add_video_label(
            top_view_frame_bgr,
            "TOP VIEW"
        )

        top_row = cv2.hconcat([
            overview_grid_frame,
            robot_grid_frame
        ])

        bottom_row = cv2.hconcat([
            birdseye_grid_frame,
            top_grid_frame
        ])

        views_grid = cv2.vconcat([
            top_row,
            bottom_row
        ])

        status_bar = np.zeros(
            (
                status_bar_height,
                grid_width,
                3
            ),
            dtype=np.uint8
        )

        display_time = (
            step + 1
        ) * DT

        aruco_visible_text = (
            "YES"
            if aruco_visible
            else "NO"
        )

        ekf_update_text = (
            "YES"
            if camera_updated
            else "NO"
        )

        status_line_1 = (
            f"Time: {display_time:5.2f} s"
            f"    |    Controller: {controller_mode}"
            f"    |    Scenario: NOMINAL"
        )

        status_line_2 = (
            f"ArUco Visible: {aruco_visible_text}"
            f"    |    EKF Visual Update: {ekf_update_text}"
        )

        cv2.putText(
            status_bar,
            status_line_1,
            (
                20,
                38
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            status_bar,
            status_line_2,
            (
                20,
                78
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        grid_frame = cv2.vconcat([
            status_bar,
            views_grid
        ])

        grid_writer.write(
            grid_frame
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

metrics = {
    "scenario": scenario_name,
    "camera_updates": camera_update_count,
    "final_true_x_m": true_state[0],
    "final_true_y_m": true_state[1],
    "final_true_heading_deg": np.degrees(
        final_true_heading_error
    ),
    "final_ekf_x_m": ekf.state[0],
    "final_ekf_y_m": ekf.state[1],
    "final_ekf_heading_deg": np.degrees(
        ekf.state[2]
    ),
    "final_localisation_error_m": final_position_error,
    "position_rmse_m": position_rmse,
    "true_lateral_rmse_m": true_lateral_rmse,
    "true_heading_rmse_deg": np.degrees(
        true_heading_rmse
    ),
    "final_x_docking_error_m": final_x_docking_error,
    "final_y_docking_error_m": final_y_docking_error,
    "final_planar_docking_error_m": (
        final_planar_docking_error
    ),
}

metrics["obstacle_avoidance"] = (
    "not_applicable"
)

# ==================================================
# SAVE METRICS CSV
# ==================================================

metrics_path = (
    results_dir
    / "metrics.csv"
)

with open(
    metrics_path,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(
        csvfile
    )

    writer.writerow([
        "metric",
        "value"
    ])

    for key, value in metrics.items():

        writer.writerow([
            key,
            value
        ])

print()

print(
    f"Saved metrics to: "
    f"{metrics_path}"
)

video_writer.release()
robot_camera_writer.release()
birdseye_writer.release()
top_view_writer.release()
grid_writer.release()

print()

print(
    f"Saved overview video to: "
    f"{video_path}"
)

print(
    f"Saved robot-camera video to: "
    f"{robot_camera_video_path}"
)

print(
    f"Saved bird's-eye video to: "
    f"{birdseye_video_path}"
)

print(
    f"Saved top-view video to: "
    f"{top_view_video_path}"
)

print(
    f"Saved four-view grid video to: "
    f"{grid_video_path}"
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
    "Full MuJoCo EKF + LQR Docking"
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

# ==================================================
# SAVE FIGURES
# ==================================================

figure_names = [
    "trajectory",
    "lateral_error",
    "heading_error",
    "control_commands",
    "lidar_clearance",
    "aruco_updates"
]

if last_detection_image is not None:

    figure_names.append(
        "latest_aruco_detection"
    )

figure_numbers = (
    plt.get_fignums()
)

for figure_number, figure_name in zip(
    figure_numbers,
    figure_names
):

    figure = plt.figure(
        figure_number
    )

    output_path = (
        results_dir
        / f"{figure_name}.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

print()

print(
    f"Saved figures to: "
    f"{results_dir}"
)

plt.show()
