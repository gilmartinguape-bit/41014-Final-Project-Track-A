import numpy as np

from src.config import DT
from src.ekf import PoseEKF


initial_state = np.array([
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
    initial_state,
    initial_covariance,
    process_noise_covariance
)


control = np.array([
    0.18,
    0.0
])


ekf.predict(
    control,
    DT
)


print("State after prediction:")
print(ekf.state)

print()

print("Covariance after prediction:")
print(ekf.P)


camera_measurement = np.array([
    0.012,
    -0.004,
    np.deg2rad(1.0)
])


ekf.update_pose(
    camera_measurement,
    measurement_noise_covariance
)


print()

print("Camera measurement:")
print(camera_measurement)

print()

print("State after camera update:")
print(ekf.state)

print()

print("Covariance after camera update:")
print(ekf.P)