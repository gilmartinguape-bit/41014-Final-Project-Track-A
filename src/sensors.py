import numpy as np


def noisy_wheel_speeds(
    omega_left,
    omega_right,
    sigma_left,
    sigma_right,
    rng
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
        omega_left + noise_left
    )

    measured_right = (
        omega_right + noise_right
    )

    return measured_left, measured_right