import numpy as np

from src.utils import wrap_angle

def noisy_wheel_speeds(
    omega_left,
    omega_right,
    sigma_left,
    sigma_right,
    rng,
    bias_left=0.0,
    bias_right=0.0
):
    """
    Simulate noisy wheel encoder measurements.

    Parameters
    ----------
    omega_left : float
        True left wheel angular velocity [rad/s]

    omega_right : float
        True right wheel angular velocity [rad/s]

    sigma_left : float
        Standard deviation of left wheel measurement noise

    sigma_right : float
        Standard deviation of right wheel measurement noise

    rng :
        NumPy random number generator
    """

    noise_left = rng.normal(
        0.0,
        sigma_left
    )

    noise_right = rng.normal(
        0.0,
        sigma_right
    )

    measured_left = (
        omega_left
        + bias_left
        + noise_left
    )

    measured_right = (
        omega_right
        + bias_right
        + noise_right
    )

    return measured_left, measured_right


def noisy_camera_pose(
    true_state,
    sigma_x,
    sigma_y,
    sigma_theta,
    rng
):
    """
    Simulate a noisy camera/ArUco pose measurement.

    true_state = [x, y, theta]
    """

    noise_x = rng.normal(
        0.0,
        sigma_x
    )

    noise_y = rng.normal(
        0.0,
        sigma_y
    )

    noise_theta = rng.normal(
        0.0,
        sigma_theta
    )

    measured_state = np.array([
        true_state[0] + noise_x,
        true_state[1] + noise_y,
        true_state[2] + noise_theta
    ])

    return measured_state

def camera_can_see_marker(
    robot_state,
    marker_position,
    max_range,
    field_of_view
):
    """
    Check whether the ArUco marker is within
    camera range and field of view.

    robot_state = [x, y, theta]

    marker_position = [marker_x, marker_y]

    max_range is in metres.

    field_of_view is in radians.
    """

    robot_x = robot_state[0]
    robot_y = robot_state[1]
    robot_theta = robot_state[2]

    marker_x = marker_position[0]
    marker_y = marker_position[1]


    dx = marker_x - robot_x
    dy = marker_y - robot_y


    distance = np.sqrt(
        dx**2 + dy**2
    )


    angle_to_marker = np.arctan2(
        dy,
        dx
    )


    relative_bearing = wrap_angle(
        angle_to_marker
        -
        robot_theta
    )


    within_range = (
        distance <= max_range
    )

    within_field_of_view = (
        abs(relative_bearing)
        <= field_of_view / 2.0
    )


    visible = (
        within_range
        and
        within_field_of_view
    )


    return visible