import numpy as np

from src.config import DT
from src.model import robot_step
from src.model import wheel_speeds_to_control


state = np.array([
    0.0,   # x position [m]
    0.0,   # y position [m]
    0.0    # heading [rad]
])

control = np.array([
    0.18,  # linear velocity v [m/s]
    0.0    # angular velocity omega [rad/s]
])


next_state = robot_step(
    state,
    control,
    DT
)


print("Initial state:")
print(state)

print()

print("Control input:")
print(control)

print()

print("State after one time step:")
print(next_state)

print()
print("----- ONE SECOND TEST -----")

state = np.array([
    0.0,
    0.0,
    0.0
])

control = np.array([
    0.18,
    0.0
])


number_of_steps = 20


for step in range(number_of_steps):

    state = robot_step(
        state,
        control,
        DT
    )


print("State after 1 second:")
print(state)

print()
print("----- 90 DEGREE HEADING TEST -----")

state = np.array([
    0.0,
    0.0,
    np.pi / 2
])

control = np.array([
    0.18,
    0.0
])


for step in range(20):

    state = robot_step(
        state,
        control,
        DT
    )


print("State after travelling for 1 second:")
print(state)

print()
print("----- ROTATION TEST -----")

state = np.array([
    0.0,
    0.0,
    0.0
])

control = np.array([
    0.0,
    np.pi / 4
])


for step in range(20):

    state = robot_step(
        state,
        control,
        DT
    )


print("State after rotating for 1 second:")
print(state)

print()
print("----- WHEEL SPEED TEST: STRAIGHT -----")

control = wheel_speeds_to_control(
    omega_left=5.0,
    omega_right=5.0,
    wheel_radius=0.033,
    wheel_base=0.160
)

print("Control from equal wheel speeds:")
print(control)

print()
print("----- WHEEL SPEED TEST: TURNING -----")

control = wheel_speeds_to_control(
    omega_left=3.0,
    omega_right=5.0,
    wheel_radius=0.033,
    wheel_base=0.160
)

print("Control from unequal wheel speeds:")
print(control)