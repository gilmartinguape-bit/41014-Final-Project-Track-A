from src.lidar import (
    lidar_safety_state_machine
)


mode = "NORMAL"

turn_direction = 0.0

recovery_steps_remaining = 0


slow_distance = 0.65
enter_distance = 0.30
exit_distance = 0.60

avoidance_turn_rate = 0.8

recovery_speed = 0.08

recovery_steps = 20


linear_velocity = 0.18
angular_velocity = 0.0


test_ranges = [
    # clear
    (1.0, 1.0, 1.0),

    # slowing
    (0.8, 0.50, 0.9),

    # enter avoidance
    (0.35, 0.25, 0.80),

    # still avoiding
    (0.40, 0.35, 0.90),

    # still avoiding
    (0.50, 0.50, 1.00),

    # sufficiently clear → recovery
    (0.80, 0.65, 1.10),

    # recovery
    (0.90, 0.80, 1.20),
]


for step, (
    left_distance,
    front_distance,
    right_distance
) in enumerate(test_ranges):


    (
        safe_control,
        mode,
        turn_direction,
        recovery_steps_remaining
    ) = lidar_safety_state_machine(
        linear_velocity,
        angular_velocity,
        left_distance,
        front_distance,
        right_distance,
        mode,
        turn_direction,
        recovery_steps_remaining,
        slow_distance,
        enter_distance,
        exit_distance,
        avoidance_turn_rate,
        recovery_speed,
        recovery_steps
    )


    print(
        f"Step {step}: "
        f"Front={front_distance:.2f} m | "
        f"Mode={mode} | "
        f"v={safe_control[0]:.2f} | "
        f"omega={safe_control[1]:.2f}"
    )