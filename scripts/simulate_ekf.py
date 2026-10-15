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


rng = np.random.default_rng(42)


# -----------------------------
# INITIAL STATES
# -----------------------------

true_state = np.array([
    0.0,
    0.0,
    0.0
])

odometry_state = np.array([
    0.0,
    0.0,
    0.0
])


# -----------------------------
# EKF SETUP
# -----------------------------

initial_estimate = np.array([
    0.0,
    0.0,
    0.0
])

initial_covariance = np.diag([
    0.01,
    0.01,
    np.deg2rad(5.0)**2
])

process_noise_covariance = np.diag([
    0.001**2,
    0.001**2,
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


# -----------------------------
# SENSOR / MOTION SETTINGS
# -----------------------------

omega_left_true = 4.0
omega_right_true = 5.0

encoder_sigma = 0.15

encoder_bias_left = 0.00
encoder_bias_right = 0.05


marker_position = np.array([
    1.0,
    0.2
])

camera_max_range = 2.0

camera_field_of_view = np.deg2rad(
    70.0
)


# -----------------------------
# SIMULATION SETTINGS
# -----------------------------

simulation_time = 20.0

number_of_steps = int(
    simulation_time / DT
)


# -----------------------------
# HISTORY
# -----------------------------

true_x_history = []
true_y_history = []

odom_x_history = []
odom_y_history = []

ekf_x_history = []
ekf_y_history = []

ekf_position_error_history = []

time_history = []

camera_update_count = 0

camera_visible_history = []

sigma_x_history = []
sigma_y_history = []
sigma_theta_history = []

x_error_history = []
y_error_history = []
theta_error_history = []

# -----------------------------
# MAIN LOOP
# -----------------------------

for step in range(number_of_steps):

    # TRUE MOTION

    true_control = wheel_speeds_to_control(
        omega_left_true,
        omega_right_true,
        WHEEL_RADIUS,
        WHEEL_BASE
    )

    true_state = robot_step(
        true_state,
        true_control,
        DT
    )


    # NOISY ENCODERS

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


    # ODOMETRY-ONLY STATE

    odometry_state = robot_step(
        odometry_state,
        odometry_control,
        DT
    )


    # EKF PREDICTION

    ekf.predict(
        odometry_control,
        DT
    )

    

    # CAMERA VISIBILITY

    visible = camera_can_see_marker(
        true_state,
        marker_position,
        camera_max_range,
        camera_field_of_view
    )

    camera_visible_history.append(
        visible
    )


    # CAMERA UPDATE

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


    # SAVE RESULTS

    sigma_x_history.append(
        np.sqrt(ekf.P[0, 0])
    )
    
    sigma_y_history.append(
        np.sqrt(ekf.P[1, 1])
    )
    
    sigma_theta_history.append(
        np.sqrt(ekf.P[2, 2])
    )
    

    true_x_history.append(
        true_state[0]
    )

    true_y_history.append(
        true_state[1]
    )

    odom_x_history.append(
        odometry_state[0]
    )

    odom_y_history.append(
        odometry_state[1]
    )

    ekf_x_history.append(
        ekf.state[0]
    )

    ekf_y_history.append(
        ekf.state[1]
    )


    ekf_position_error = np.sqrt(
        (ekf.state[0] - true_state[0])**2
        +
        (ekf.state[1] - true_state[1])**2
    )

    ekf_position_error_history.append(
        ekf_position_error
    )


    time_history.append(
        (step + 1) * DT
    )

    x_error = ekf.state[0] - true_state[0]
    y_error = ekf.state[1] - true_state[1]

    theta_error = (
        ekf.state[2] - true_state[2]
    )

    theta_error = (
        theta_error + np.pi
    ) % (
        2.0 * np.pi
     ) - np.pi


    x_error_history.append(
        x_error
    )

    y_error_history.append(
        y_error
    )


    theta_error_history.append(
        theta_error
    )


# -----------------------------
# RMSE
# -----------------------------

ekf_position_error_history = np.array(
    ekf_position_error_history
)

ekf_position_rmse = np.sqrt(
    np.mean(
        ekf_position_error_history**2
    )
)


# -----------------------------
# FINAL RESULTS
# -----------------------------

final_ekf_position_error = np.sqrt(
    (ekf.state[0] - true_state[0])**2
    +
    (ekf.state[1] - true_state[1])**2
)

camera_visible_history = np.array(
    camera_visible_history,
    dtype=int
)

sigma_x_history = np.array(
    sigma_x_history
)

sigma_y_history = np.array(
    sigma_y_history
)

sigma_theta_history = np.array(
    sigma_theta_history
)

x_error_history = np.array(
    x_error_history
)

y_error_history = np.array(
    y_error_history
)

theta_error_history = np.array(
    theta_error_history
)

print()

print("Camera update count:")
print(camera_update_count)

print("Final true state:")
print(true_state)

print()

print("Final odometry state:")
print(odometry_state)

print()

print("Final EKF state:")
print(ekf.state)

print()

print("Final EKF position error:")
print(
    final_ekf_position_error,
    "m"
)

print()

print("EKF position RMSE:")
print(
    ekf_position_rmse,
    "m"
)


# -----------------------------
# TRAJECTORY PLOT
# -----------------------------

plt.figure()

plt.plot(
    true_x_history,
    true_y_history,
    label="True trajectory"
)

plt.plot(
    odom_x_history,
    odom_y_history,
    "--",
    label="Wheel odometry"
)

plt.plot(
    ekf_x_history,
    ekf_y_history,
    label="EKF estimate"
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
    "Truth vs Odometry vs EKF"
)

plt.axis("equal")
plt.grid()
plt.legend()


# -----------------------------
# EKF ERROR PLOT
# -----------------------------

plt.figure()

plt.plot(
    time_history,
    ekf_position_error_history
)

plt.xlabel("Time [s]")
plt.ylabel("EKF position error [m]")

plt.title(
    "EKF Position Error"
)

plt.grid()

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

plt.figure()

plt.plot(
    time_history,
    x_error_history,
    label="X estimation error"
)

plt.plot(
    time_history,
    2.0 * sigma_x_history,
    "--",
    label="+2 sigma"
)

plt.plot(
    time_history,
    -2.0 * sigma_x_history,
    "--",
    label="-2 sigma"
)

plt.xlabel("Time [s]")
plt.ylabel("X error [m]")

plt.title(
    "EKF X Error and Uncertainty"
)

plt.grid()
plt.legend()

plt.figure()

plt.plot(
    time_history,
    y_error_history,
    label="Y estimation error"
)

plt.plot(
    time_history,
    2.0 * sigma_y_history,
    "--",
    label="+2 sigma"
)

plt.plot(
    time_history,
    -2.0 * sigma_y_history,
    "--",
    label="-2 sigma"
)

plt.xlabel("Time [s]")
plt.ylabel("Y error [m]")

plt.title(
    "EKF Y Error and Uncertainty"
)

plt.grid()
plt.legend()

plt.figure()

plt.plot(
    time_history,
    np.degrees(theta_error_history),
    label="Heading estimation error"
)

plt.plot(
    time_history,
    np.degrees(2.0 * sigma_theta_history),
    "--",
    label="+2 sigma"
)

plt.plot(
    time_history,
    np.degrees(-2.0 * sigma_theta_history),
    "--",
    label="-2 sigma"
)

plt.xlabel("Time [s]")
plt.ylabel("Heading error [deg]")

plt.title(
    "EKF Heading Error and Uncertainty"
)

plt.grid()
plt.legend()

plt.show()