from src.lidar import lidar_safety_override


linear_velocity = 0.18
angular_velocity = 0.0

slow_distance = 0.80
stop_distance = 0.35

avoidance_turn_rate = 0.8


# ---------------------------------
# TEST 1 — PATH CLEAR
# ---------------------------------

safe_control = lidar_safety_override(
    linear_velocity,
    angular_velocity,
    left_distance=1.5,
    front_distance=1.5,
    right_distance=1.5,
    slow_distance=slow_distance,
    stop_distance=stop_distance,
    avoidance_turn_rate=avoidance_turn_rate
)

print("TEST 1 — PATH CLEAR")
print(safe_control)


# ---------------------------------
# TEST 2 — OBSTACLE APPROACHING
# ---------------------------------

safe_control = lidar_safety_override(
    linear_velocity,
    angular_velocity,
    left_distance=1.0,
    front_distance=0.60,
    right_distance=1.0,
    slow_distance=slow_distance,
    stop_distance=stop_distance,
    avoidance_turn_rate=avoidance_turn_rate
)

print()
print("TEST 2 — SLOW DOWN")
print(safe_control)


# ---------------------------------
# TEST 3 — OBSTACLE VERY CLOSE
# MORE SPACE ON LEFT
# ---------------------------------

safe_control = lidar_safety_override(
    linear_velocity,
    angular_velocity,
    left_distance=1.0,
    front_distance=0.25,
    right_distance=0.40,
    slow_distance=slow_distance,
    stop_distance=stop_distance,
    avoidance_turn_rate=avoidance_turn_rate
)

print()
print("TEST 3 — STOP AND TURN LEFT")
print(safe_control)


# ---------------------------------
# TEST 4 — OBSTACLE VERY CLOSE
# MORE SPACE ON RIGHT
# ---------------------------------

safe_control = lidar_safety_override(
    linear_velocity,
    angular_velocity,
    left_distance=0.30,
    front_distance=0.25,
    right_distance=1.0,
    slow_distance=slow_distance,
    stop_distance=stop_distance,
    avoidance_turn_rate=avoidance_turn_rate
)

print()
print("TEST 4 — STOP AND TURN RIGHT")
print(safe_control)