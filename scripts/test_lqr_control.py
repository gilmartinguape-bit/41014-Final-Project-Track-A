import numpy as np

from src.controller import design_lqr
from src.controller import lqr_control


forward_speed = 0.18

q_y = 8.0
q_heading = 4.0
r_omega = 1.2

omega_limit = 1.5


A, B, K = design_lqr(
    forward_speed,
    q_y,
    q_heading,
    r_omega
)


# ---------------------------------
# TEST 1:
# ROBOT ABOVE THE DESIRED PATH
# ---------------------------------

estimated_state = np.array([
    0.0,
    0.30,
    0.0
])


control, lateral_error, heading_error = lqr_control(
    estimated_state,
    K,
    forward_speed,
    omega_limit
)


print("----- TEST 1: ABOVE PATH -----")

print("Estimated state:")
print(estimated_state)

print("Lateral error:")
print(lateral_error)

print("Heading error [deg]:")
print(
    np.degrees(heading_error)
)

print("Control [v, omega]:")
print(control)


# ---------------------------------
# TEST 2:
# ROBOT BELOW THE DESIRED PATH
# ---------------------------------

estimated_state = np.array([
    0.0,
    -0.30,
    0.0
])


control, lateral_error, heading_error = lqr_control(
    estimated_state,
    K,
    forward_speed,
    omega_limit
)


print()
print("----- TEST 2: BELOW PATH -----")

print("Estimated state:")
print(estimated_state)

print("Control [v, omega]:")
print(control)


# ---------------------------------
# TEST 3:
# ROBOT POINTING TOO FAR LEFT
# ---------------------------------

estimated_state = np.array([
    0.0,
    0.0,
    np.deg2rad(15.0)
])


control, lateral_error, heading_error = lqr_control(
    estimated_state,
    K,
    forward_speed,
    omega_limit
)


print()
print("----- TEST 3: POSITIVE HEADING ERROR -----")

print("Estimated heading [deg]:")
print(
    np.degrees(
        estimated_state[2]
    )
)

print("Control [v, omega]:")
print(control)