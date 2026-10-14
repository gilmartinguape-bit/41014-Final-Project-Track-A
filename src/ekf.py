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

    def update_pose(
        self,
        measurement,
        measurement_noise_covariance
    ):
        """
        EKF correction using camera/ArUco pose measurement.
        """

        H = np.eye(3)

        innovation = (
            measurement
            -
            self.state
        )

        # Wrap heading innovation to [-pi, pi)
        innovation[2] = (
            innovation[2] + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


        S = (
            H
            @ self.P
            @ H.T
            +
            measurement_noise_covariance
        )


        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )


        self.state = (
            self.state
            +
            K @ innovation
        )


        I = np.eye(3)

        self.P = (
            (I - K @ H)
            @ self.P
        )

    