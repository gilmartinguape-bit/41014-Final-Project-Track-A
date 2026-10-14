import numpy as np

from src.model import robot_step
from src.model import motion_jacobian


class PoseEKF:

    def __init__(
        self,
        initial_state,
        initial_covariance,
        process_noise_covariance
    ):

        self.state = initial_state.copy()

        self.P = initial_covariance.copy()

        self.Q = process_noise_covariance.copy()

    def predict(
        self,
        control,
        dt
    ):
        """
        EKF prediction using wheel odometry.
        """

        F = motion_jacobian(
            self.state,
            control,
            dt
        )

        self.state = robot_step(
            self.state,
            control,
            dt
        )

        self.P = (
            F
            @ self.P
            @ F.T
            +
            self.Q
        )

    