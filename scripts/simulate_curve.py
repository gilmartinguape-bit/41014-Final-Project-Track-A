import numpy as np
import matplotlib.pyplot as plt

from src.config import DT
from src.model import robot_step


state = np.array([
    0.0,
    0.0,
    0.0
])

control = np.array([
    0.18,
    0.30
])


x_history = []
y_history = []


simulation_time = 10.0

number_of_steps = int(
    simulation_time / DT
)


for step in range(number_of_steps):

    x_history.append(state[0])
    y_history.append(state[1])

    state = robot_step(
        state,
        control,
        DT
    )


print("Final robot state:")
print(state)


plt.figure()

plt.plot(
    x_history,
    y_history
)

plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")

plt.title(
    "Differential-Drive Robot Curved Motion"
)

plt.axis("equal")
plt.grid()

plt.show()