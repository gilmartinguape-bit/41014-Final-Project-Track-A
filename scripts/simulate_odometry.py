import numpy as np
import matplotlib.pyplot as plt

from src.config import DT
from src.config import WHEEL_RADIUS
from src.config import WHEEL_BASE

from src.model import robot_step
from src.model import wheel_speeds_to_control

from src.sensors import noisy_wheel_speeds
from src.utils import wrap_angle


# ---------------------------------
# RANDOM NUMBER GENERATOR
# ---------------------------------

rng = np.random.default_rng(42)


# ---------------------------------
# INITIAL STATES
# ---------------------------------

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


# ---------------------------------
# TRUE WHEEL SPEEDS
# ---------------------------------

omega_left_true = 4.0
omega_right_true = 5.0


# ---------------------------------
# ENCODER NOISE
# ---------------------------------

encoder_sigma = 0.15
encoder_bias_left = 0.00
encoder_bias_right = 0.05


# ---------------------------------
# SIMULATION SETTINGS
# ---------------------------------

simulation_time = 20.0

number_of_steps = int(
    simulation_time / DT
)


# ---------------------------------
# RESULT STORAGE
# ---------------------------------

true_x_history = []
true_y_history = []

odom_x_history = []
odom_y_history = []

position_error_history = []
time_history = []


# ---------------------------------
# SIMULATION LOOP
# ---------------------------------

for step in range(number_of_steps):

    # ---------------------------------
    # TRUE ROBOT MOTION
    # ---------------------------------

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


    # ---------------------------------
    # SIMULATED ENCODER MEASUREMENTS
    # ---------------------------------

    measured_left, measured_right = noisy_wheel_speeds(
        omega_left_true,
        omega_right_true,
        encoder_sigma,
        encoder_sigma,
        rng,
        encoder_bias_left,
        encoder_bias_right
    )


    # ---------------------------------
    # ODOMETRY CALCULATION
    # ---------------------------------

    odometry_control = wheel_speeds_to_control(
        measured_left,
        measured_right,
        WHEEL_RADIUS,
        WHEEL_BASE
    )

    odometry_state = robot_step(
        odometry_state,
        odometry_control,
        DT
    )


    # ---------------------------------
    # SAVE TRAJECTORY RESULTS
    # ---------------------------------

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


    # ---------------------------------
    # CALCULATE POSITION ERROR
    # ---------------------------------

    position_error = np.sqrt(
        (odometry_state[0] - true_state[0])**2
        +
        (odometry_state[1] - true_state[1])**2
    )

    position_error_history.append(
        position_error
    )


    # ---------------------------------
    # SAVE TIME
    # ---------------------------------

    current_time = (step + 1) * DT

    time_history.append(
        current_time
    )


# ---------------------------------
# CONVERT HISTORIES TO NUMPY ARRAYS
# ---------------------------------

position_error_history = np.array(
    position_error_history
)

time_history = np.array(
    time_history
)


# ---------------------------------
# CALCULATE RMSE
# ---------------------------------

position_rmse = np.sqrt(
    np.mean(
        position_error_history**2
    )
)


# ---------------------------------
# FINAL ERRORS
# ---------------------------------

final_position_error = np.sqrt(
    (odometry_state[0] - true_state[0])**2
    +
    (odometry_state[1] - true_state[1])**2
)

heading_error = wrap_angle(
    odometry_state[2]
    -
    true_state[2]
)


# ---------------------------------
# PRINT RESULTS
# ---------------------------------

print("Final true state:")
print(true_state)

print()

print("Final odometry state:")
print(odometry_state)

print()

print("Final position error:")
print(
    final_position_error,
    "m"
)

print()

print("Final heading error:")
print(
    np.degrees(heading_error),
    "degrees"
)

print()

print("Position RMSE:")
print(
    position_rmse,
    "m"
)


# ---------------------------------
# FIGURE 1:
# TRUE VS ODOMETRY TRAJECTORY
# ---------------------------------

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

plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")

plt.title(
    "True Trajectory vs Wheel Odometry"
)

plt.axis("equal")
plt.grid()
plt.legend()


# ---------------------------------
# FIGURE 2:
# POSITION ERROR VS TIME
# ---------------------------------

plt.figure()

plt.plot(
    time_history,
    position_error_history
)

plt.xlabel("Time [s]")
plt.ylabel("Position error [m]")

plt.title(
    "Wheel Odometry Position Error"
)

plt.grid()


# ---------------------------------
# SHOW ALL FIGURES
# ---------------------------------

plt.show()