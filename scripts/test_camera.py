import numpy as np

from src.sensors import noisy_camera_pose
from src.sensors import camera_can_see_marker


rng = np.random.default_rng(42)


true_state = np.array([
    1.0,
    2.0,
    np.deg2rad(30.0)
])


sigma_x = 0.02
sigma_y = 0.02
sigma_theta = np.deg2rad(2.0)


camera_measurement = noisy_camera_pose(
    true_state,
    sigma_x,
    sigma_y,
    sigma_theta,
    rng
)


print("True state:")
print(true_state)

print()

print("Camera measurement:")
print(camera_measurement)

print()

print("True heading [deg]:")
print(
    np.degrees(
        true_state[2]
    )
)

print()

print("Measured heading [deg]:")
print(
    np.degrees(
        camera_measurement[2]
    )
)

print()
print("----- CAMERA VISIBILITY TEST -----")


marker_position = np.array([
    3.0,
    2.0
])


max_range = 3.0

field_of_view = np.deg2rad(
    70.0
)


visible = camera_can_see_marker(
    true_state,
    marker_position,
    max_range,
    field_of_view
)


print("Marker position:")
print(marker_position)

print()

print("Marker visible:")
print(visible)