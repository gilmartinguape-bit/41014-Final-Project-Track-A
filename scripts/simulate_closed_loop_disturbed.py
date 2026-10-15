import numpy as np
import matplotlib.pyplot as plt

from src.config import DT
from src.config import WHEEL_RADIUS
from src.config import WHEEL_BASE

from src.model import robot_step
from src.model import wheel_speeds_to_control

from src.sensors import noisy_wheel_speeds
from src.sensors import noisy_camera_pose
from src.sensors import camera_can_see_marker

from src.ekf import PoseEKF

from src.controller import design_lqr
from src.controller import lqr_control


rng = np.random.default_rng(42)


# --------------------------------------------------
# INITIAL TRUE STATE
# --------------------------------------------------

true_state = np.array([
    0.0,                    # x [m]
    0.30,                   # y [m]
    np.deg2rad(10.0)        # heading [rad]
])


# --------------------------------------------------
# INITIAL EKF STATE
# --------------------------------------------------

initial_estimate = np.array([
    0.0,
    0.25,
    np.deg2rad(8.0)
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
    0.02**2,
    0.02**2,
    np.deg2rad(2.0)**2
])


ekf = PoseEKF(
    initial_estimate,
    initial_covariance,
    process_noise_covariance
)


# --------------------------------------------------
# LQR DESIGN
# --------------------------------------------------

forward_speed = 0.18

q_y = 8.0
q_heading = 4.0
r_omega = 1.2

omega_limit = 1.5


A, B, K = design_lqr(
    forward_speed,
    q_y,
    q_heading,
    r_omega
)


# --------------------------------------------------
# SENSOR PARAMETERS
# --------------------------------------------------

encoder_sigma = 0.15

encoder_bias_left = 0.00
encoder_bias_right = 0.05


marker_position = np.array([
    1.0,
    0.0
])


camera_max_range = 2.0

camera_field_of_view = np.deg2rad(
    70.0
)


# --------------------------------------------------
# SIMULATION SETTINGS
# --------------------------------------------------

simulation_time = 20.0

number_of_steps = int(
    simulation_time / DT
)


# --------------------------------------------------
# HISTORIES
# --------------------------------------------------

time_history = []

true_x_history = []
true_y_history = []

ekf_x_history = []
ekf_y_history = []

lateral_error_history = []
heading_error_history = []

omega_command_history = []

camera_visible_history = []

position_error_history = []

sigma_x_history = []
sigma_y_history = []
sigma_theta_history = []

true_lateral_error_history = []
true_heading_error_history = []

camera_update_count = 0


# --------------------------------------------------
# INITIAL CONTROL
# --------------------------------------------------

control_command = np.array([
    forward_speed,
    0.0
])


# --------------------------------------------------
# MAIN CLOSED-LOOP SIMULATION
# --------------------------------------------------

for step in range(number_of_steps):

    # ==================================================
    # 1. TRUE ROBOT MOTION
    # ==================================================

    true_state = robot_step(
        true_state,
        control_command,
        DT
    )

    true_lateral_error = true_state[1]

    true_heading_error = (
        true_state[2] + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


    # ==================================================
    # 2. CONVERT TRUE BODY CONTROL TO TRUE WHEEL SPEEDS
    # ==================================================

    v_true = control_command[0]
    omega_true = control_command[1]


    omega_right_true = (
        v_true
        +
        (WHEEL_BASE / 2.0) * omega_true
    ) / WHEEL_RADIUS


    omega_left_true = (
        v_true
        -
        (WHEEL_BASE / 2.0) * omega_true
    ) / WHEEL_RADIUS


    # ==================================================
    # 3. SIMULATE ENCODER MEASUREMENTS
    # ==================================================

    measured_left, measured_right = noisy_wheel_speeds(
        omega_left_true,
        omega_right_true,
        encoder_sigma,
        encoder_sigma,
        rng,
        encoder_bias_left,
        encoder_bias_right
    )


    odometry_control = wheel_speeds_to_control(
        measured_left,
        measured_right,
        WHEEL_RADIUS,
        WHEEL_BASE
    )


    # ==================================================
    # 4. EKF PREDICTION
    # ==================================================

    ekf.predict(
        odometry_control,
        DT
    )


    # ==================================================
    # 5. CAMERA VISIBILITY
    # ==================================================

    visible = camera_can_see_marker(
        true_state,
        marker_position,
        camera_max_range,
        camera_field_of_view
    )


    camera_visible_history.append(
        visible
    )


    # ==================================================
    # 6. CAMERA UPDATE
    # ==================================================

    if visible:

        camera_update_count += 1

        camera_measurement = noisy_camera_pose(
            true_state,
            0.02,
            0.02,
            np.deg2rad(2.0),
            rng
        )

        ekf.update_pose(
            camera_measurement,
            measurement_noise_covariance
        )


    # ==================================================
    # 7. LQR CONTROL
    #
    # IMPORTANT:
    # CONTROLLER USES ekf.state
    # NOT true_state
    # ==================================================

    control_command, lateral_error, heading_error = lqr_control(
        ekf.state,
        K,
        forward_speed,
        omega_limit
    )


    # ==================================================
    # 8. SAVE RESULTS
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

    true_lateral_error_history.append(
        true_lateral_error
    )

    true_heading_error_history.append(
        true_heading_error
    )

    lateral_error_history.append(
        lateral_error
    )

    heading_error_history.append(
        heading_error
    )


    omega_command_history.append(
        control_command[1]
    )


    position_error = np.sqrt(
        (ekf.state[0] - true_state[0])**2
        +
        (ekf.state[1] - true_state[1])**2
    )

    position_error_history.append(
        position_error
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


# --------------------------------------------------
# CONVERT HISTORIES TO ARRAYS
# --------------------------------------------------

time_history = np.array(
    time_history
)

true_lateral_error_history = np.array(
    true_lateral_error_history
)

true_heading_error_history = np.array(
    true_heading_error_history
)

lateral_error_history = np.array(
    lateral_error_history
)

heading_error_history = np.array(
    heading_error_history
)

omega_command_history = np.array(
    omega_command_history
)

position_error_history = np.array(
    position_error_history
)

camera_visible_history = np.array(
    camera_visible_history,
    dtype=int
)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

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

lateral_rmse = np.sqrt(
    np.mean(
        lateral_error_history**2
    )
)


heading_rmse = np.sqrt(
    np.mean(
        heading_error_history**2
    )
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("Camera update count:")
print(camera_update_count)

print()

print("Final true state:")
print(true_state)

print()

print("Final EKF state:")
print(ekf.state)

print()

print("Position RMSE:")
print(
    position_rmse,
    "m"
)

print()

print("Lateral tracking RMSE:")
print(
    lateral_rmse,
    "m"
)

print()

print("Heading tracking RMSE:")
print(
    np.degrees(
        heading_rmse
    ),
    "degrees"
)

print()

print("True lateral tracking RMSE:")
print(
    true_lateral_rmse,
    "m"
)

print()

print("True heading tracking RMSE:")
print(
    np.degrees(
        true_heading_rmse
    ),
    "degrees"
)

# --------------------------------------------------
# FIGURE 1 — TRAJECTORY
# --------------------------------------------------

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
    label="Desired path"
)

plt.scatter(
    marker_position[0],
    marker_position[1],
    marker="s",
    label="ArUco marker"
)

plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")

plt.title(
    "Closed-Loop LQR Path Tracking"
)

plt.axis("equal")
plt.grid()
plt.legend()


# --------------------------------------------------
# FIGURE 2 — LATERAL ERROR
# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    lateral_error_history,
    label="Estimated lateral error"
)

plt.plot(
    time_history,
    true_lateral_error_history,
    "--",
    label="True lateral error"
)

plt.axhline(
    y=0.0,
    linestyle=":"
)

plt.xlabel("Time [s]")
plt.ylabel("Lateral error [m]")

plt.title(
    "Estimated vs True Lateral Tracking Error"
)

plt.grid()
plt.legend()


# --------------------------------------------------
# FIGURE 3 — HEADING ERROR
# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    np.degrees(
        heading_error_history
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

plt.axhline(
    y=0.0,
    linestyle=":"
)

plt.xlabel("Time [s]")
plt.ylabel("Heading error [deg]")

plt.title(
    "Estimated vs True Heading Tracking Error"
)

plt.grid()
plt.legend()


# --------------------------------------------------
# FIGURE 4 — CONTROL EFFORT
# --------------------------------------------------

plt.figure()

plt.plot(
    time_history,
    omega_command_history
)

plt.xlabel("Time [s]")
plt.ylabel("Angular velocity command [rad/s]")

plt.title(
    "LQR Control Effort"
)

plt.grid()


# --------------------------------------------------
# FIGURE 5 — CAMERA VISIBILITY
# --------------------------------------------------

plt.figure()

plt.step(
    time_history,
    camera_visible_history,
    where="post"
)

plt.xlabel("Time [s]")
plt.ylabel("Marker visible")

plt.title(
    "ArUco Marker Visibility"
)

plt.yticks(
    [0, 1],
    ["No", "Yes"]
)

plt.grid()


plt.show()