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