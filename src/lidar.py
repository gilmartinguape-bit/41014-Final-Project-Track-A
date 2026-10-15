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