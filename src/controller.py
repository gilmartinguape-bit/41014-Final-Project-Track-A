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


def docking_pose_control(
    estimated_state,
    target_pose,
    max_linear_speed=0.12,
    max_angular_speed=1.0,
    k_rho=0.8,
    k_alpha=2.0,
    k_beta=-0.6,
    position_tolerance=0.025,
    heading_tolerance=np.deg2rad(3.0)
):
    """
    Final pose controller for differential-drive docking.

    Controls both position and orientation using
    the EKF-estimated robot pose.
    """

    x = estimated_state[0]
    y = estimated_state[1]
    theta = estimated_state[2]

    target_x = target_pose[0]
    target_y = target_pose[1]
    target_theta = target_pose[2]


    dx = target_x - x
    dy = target_y - y


    rho = np.sqrt(
        dx**2 + dy**2
    )


    # Direction from robot to docking target.
    target_bearing = np.arctan2(
        dy,
        dx
    )


    alpha = wrap_angle(
        target_bearing - theta
    )


    beta = wrap_angle(
        target_theta
        - theta
        - alpha
    )


    # ==============================================
    # POSITION NOT YET REACHED
    # ==============================================

    if rho > position_tolerance:

        v = (
            k_rho
            * rho
        )

        omega = (
            k_alpha
            * alpha
            +
            k_beta
            * beta
        )


        v = np.clip(
            v,
            0.0,
            max_linear_speed
        )


        omega = np.clip(
            omega,
            -max_angular_speed,
            max_angular_speed
        )


    # ==============================================
    # POSITION REACHED — ALIGN HEADING
    # ==============================================

    else:

        v = 0.0

        heading_error = wrap_angle(
            target_theta
            - theta
        )


        if (
            abs(heading_error)
            > heading_tolerance
        ):

            omega = np.clip(
                1.5 * heading_error,
                -max_angular_speed,
                max_angular_speed
            )

        else:

            omega = 0.0


    return (
        np.array([
            v,
            omega
        ]),
        rho,
        wrap_angle(
            target_theta - theta
        )
    )