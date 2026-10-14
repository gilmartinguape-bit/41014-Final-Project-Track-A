import numpy as np


def robot_step(state, control, dt):
    """
    Propagate the robot state forward by one time step.

    state = [x, y, theta]
    control = [v, omega]
    """

    x = state[0]
    y = state[1]
    theta = state[2]

    v = control[0]
    omega = control[1]

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt

    next_state = np.array([
        x_next,
        y_next,
        theta_next
    ])

    return next_state

def wheel_speeds_to_control(
    omega_left,
    omega_right,
    wheel_radius,
    wheel_base
):
    """
    Convert left and right wheel angular velocities
    into robot linear and angular velocity.

    omega_left  = left wheel angular velocity [rad/s]
    omega_right = right wheel angular velocity [rad/s]
    """

    v_left = wheel_radius * omega_left
    v_right = wheel_radius * omega_right

    v = (v_right + v_left) / 2.0

    omega = (
        v_right - v_left
    ) / wheel_base

    control = np.array([
        v,
        omega
    ])

    return control