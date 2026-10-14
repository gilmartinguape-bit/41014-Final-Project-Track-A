import numpy as np


def wrap_angle(angle):
    """
    Wrap an angle to the interval [-pi, pi).
    """

    wrapped_angle = (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    return wrapped_angle