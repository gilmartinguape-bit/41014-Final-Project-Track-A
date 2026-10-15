import numpy as np

from scipy.linalg import solve_continuous_are

from src.utils import wrap_angle


def design_lqr(
    forward_speed,
    q_y,
    q_heading,
    r_omega
):
    """
    Design an LQR controller for lateral and heading error.

    Error state:
        [e_y, e_heading]

    Control:
        omega
    """

    A = np.array([
        [
            0.0,
            forward_speed
        ],
        [
            0.0,
            0.0
        ]
    ])

    B = np.array([
        [
            0.0
        ],
        [
            1.0
        ]
    ])


    Q = np.diag([
        q_y,
        q_heading
    ])

    R = np.array([
        [
            r_omega
        ]
    ])


    P = solve_continuous_are(
        A,
        B,
        Q,
        R
    )


    K = (
        np.linalg.inv(R)
        @ B.T
        @ P
    )


    return A, B, K


def lqr_control(
    estimated_state,
    gain,
    forward_speed,
    omega_limit
):
    """
    Calculate robot control using estimated state.

    IMPORTANT:
    estimated_state must come from the EKF,
    not simulator ground truth.
    """

    estimated_y = estimated_state[1]
    estimated_heading = estimated_state[2]


    lateral_error = estimated_y

    heading_error = wrap_angle(
        estimated_heading
    )


    error_state = np.array([
        lateral_error,
        heading_error
    ])


    omega_command = -(
        gain
        @ error_state
    )[0]


    omega_command = np.clip(
        omega_command,
        -omega_limit,
        omega_limit
    )


    control = np.array([
        forward_speed,
        omega_command
    ])


    return (
        control,
        lateral_error,
        heading_error
    )

def docking_forward_speed(
    estimated_x,
    target_x,
    max_speed,
    slow_distance,
    stop_tolerance
):
    """
    Generate forward speed for approach to docking target.

    Uses estimated state only.
    """

    distance_remaining = (
        target_x - estimated_x
    )


    if distance_remaining <= stop_tolerance:

        return 0.0


    if distance_remaining <= slow_distance:

        scale = (
            distance_remaining
            /
            slow_distance
        )

        scale = np.clip(
            scale,
            0.15,
            1.0
        )

        return (
            max_speed * scale
        )


    return max_speed