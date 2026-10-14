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


ekf = PoseEKF(
    initial_state,
    initial_covariance,
    process_noise_covariance
)


control = np.array([
    0.18,
    0.0
])


print("Initial EKF state:")
print(ekf.state)

print()

print("Initial covariance:")
print(ekf.P)


ekf.predict(
    control,
    DT
)


print()

print("Predicted EKF state:")
print(ekf.state)

print()

print("Predicted covariance:")
print(ekf.P)