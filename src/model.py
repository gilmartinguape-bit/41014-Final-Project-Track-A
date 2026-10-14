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