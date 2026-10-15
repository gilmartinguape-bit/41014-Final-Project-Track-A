import numpy as np
import mujoco


def cast_lidar_ray(
    model,
    data,
    origin,
    angle,
    max_range
):
    """
    Cast one horizontal LiDAR ray in MuJoCo.

    Parameters
    ----------
    model :
        MuJoCo model

    data :
        MuJoCo simulation data

    origin :
        [x, y, z] ray starting position

    angle :
        Ray direction in world frame [rad]

    max_range :
        Maximum LiDAR range [m]

    Returns
    -------
    distance :
        Measured range [m]

    geom_id :
        ID of geometry hit
    """

    direction = np.array([
        np.cos(angle),
        np.sin(angle),
        0.0
    ])


    geom_id = np.array(
        [-1],
        dtype=np.int32
    )


    distance = mujoco.mj_ray(
        model,
        data,
        origin,
        direction,
        None,
        1,
        -1,
        geom_id
    )


    # MuJoCo returns a negative value
    # when no geometry is hit.
    if distance < 0.0:

        distance = max_range

        geom_id[0] = -1


    # Limit reading to sensor maximum range.
    distance = min(
        distance,
        max_range
    )


    return (
        distance,
        geom_id[0]
    )

def lidar_safety_override(
    linear_velocity,
    angular_velocity,
    left_distance,
    front_distance,
    right_distance,
    slow_distance,
    stop_distance,
    avoidance_turn_rate
):
    """
    Modify the commanded robot velocity when
    an obstacle is too close.

    Positive angular velocity = turn left.
    Negative angular velocity = turn right.
    """

    safe_v = linear_velocity
    safe_omega = angular_velocity


    # ---------------------------------
    # CASE 1:
    # OBSTACLE VERY CLOSE IN FRONT
    # ---------------------------------

    if front_distance <= stop_distance:

        safe_v = 0.0

        # Turn toward the side with
        # greater free space.

        if left_distance > right_distance:

            safe_omega = avoidance_turn_rate

        else:

            safe_omega = -avoidance_turn_rate


    # ---------------------------------
    # CASE 2:
    # OBSTACLE APPROACHING
    # ---------------------------------

    elif front_distance <= slow_distance:

        # Reduce speed progressively.

        scale = (
            front_distance
            - stop_distance
        ) / (
            slow_distance
            - stop_distance
        )


        scale = np.clip(
            scale,
            0.15,
            1.0
        )


        safe_v = (
            linear_velocity
            * scale
        )


    return np.array([
        safe_v,
        safe_omega
    ])

def lidar_safety_state_machine(
    linear_velocity,
    angular_velocity,
    left_distance,
    front_distance,
    right_distance,
    mode,
    turn_direction,
    recovery_steps_remaining,
    avoidance_steps_remaining,
    slow_distance,
    enter_distance,
    exit_distance,
    avoidance_turn_rate,
    recovery_speed,
    recovery_steps,
    minimum_avoidance_steps
):
    """
    Stateful LiDAR obstacle-avoidance layer.

    Modes:
        NORMAL
        AVOIDING
        RECOVERY

    Returns:
        safe_control
        new_mode
        new_turn_direction
        new_recovery_steps_remaining
        new_avoidance_steps_remaining
    """

    safe_v = linear_velocity
    safe_omega = angular_velocity


    # ==================================================
    # MODE 1 — NORMAL
    # ==================================================

    if mode == "NORMAL":

        # Obstacle is dangerously close.
        if front_distance <= enter_distance:

            mode = "AVOIDING"

            avoidance_steps_remaining = (
                minimum_avoidance_steps
            )

            # Lock the turn direction based on
            # whichever side has more clearance.
            if left_distance > right_distance:

                turn_direction = 1.0

            else:

                turn_direction = -1.0


            safe_v = 0.0

            safe_omega = (
                turn_direction
                * avoidance_turn_rate
            )


        # Obstacle is approaching,
        # but not yet inside the avoidance threshold.
        elif front_distance <= slow_distance:

            scale = (
                front_distance
                - enter_distance
            ) / (
                slow_distance
                - enter_distance
            )


            scale = np.clip(
                scale,
                0.20,
                1.0
            )


            safe_v = (
                linear_velocity
                * scale
            )


    # ==================================================
    # MODE 2 — AVOIDING
    # ==================================================

    elif mode == "AVOIDING":

        if avoidance_steps_remaining > 0:

            avoidance_steps_remaining -= 1

        # Stay committed to the chosen turn.
        safe_v = 0.0

        safe_omega = (
            turn_direction
            * avoidance_turn_rate
        )


        # Do NOT leave avoidance as soon as
        # the obstacle is barely outside enter_distance.
        #
        # Require the larger exit_distance.
        if (
            front_distance >= exit_distance
            and
            avoidance_steps_remaining <= 0
        ):

            mode = "RECOVERY"

            recovery_steps_remaining = (
                recovery_steps
            )


            safe_v = recovery_speed

            # Drive straight along the cleared avoidance heading.
            safe_omega = 0.0


    # ==================================================
    # MODE 3 — RECOVERY
    # ==================================================

    elif mode == "RECOVERY":

        # If the obstacle becomes dangerous again,
        # immediately return to avoidance.
        if front_distance <= enter_distance:

            mode = "AVOIDING"

            avoidance_steps_remaining = (
                minimum_avoidance_steps
            )

            safe_v = 0.0

            safe_omega = (
                turn_direction
                * avoidance_turn_rate
            )


        elif recovery_steps_remaining > 0:

            recovery_steps_remaining -= 1

            safe_v = recovery_speed

            safe_omega = 0.0


        else:

            # Recovery complete.
            # Give control back to LQR.
            mode = "NORMAL"

            turn_direction = 0.0

            safe_v = linear_velocity
            safe_omega = angular_velocity


    return (
        np.array([
            safe_v,
            safe_omega
        ]),
        mode,
        turn_direction,
        recovery_steps_remaining,
        avoidance_steps_remaining
    )