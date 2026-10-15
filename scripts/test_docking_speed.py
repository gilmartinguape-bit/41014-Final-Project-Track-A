from src.controller import docking_forward_speed


target_x = 1.20
max_speed = 0.18
slow_distance = 0.50
stop_tolerance = 0.03


test_positions = [
    0.0,
    0.6,
    0.8,
    1.0,
    1.15,
    1.18,
    1.20
]


for x in test_positions:

    v = docking_forward_speed(
        x,
        target_x,
        max_speed,
        slow_distance,
        stop_tolerance
    )

    print(
        f"x = {x:.2f} m"
        f" → v = {v:.3f} m/s"
    )