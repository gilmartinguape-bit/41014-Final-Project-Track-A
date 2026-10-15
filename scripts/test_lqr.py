import numpy as np

from src.controller import design_lqr


forward_speed = 0.18

q_y = 8.0
q_heading = 4.0

r_omega = 1.2


A, B, K = design_lqr(
    forward_speed,
    q_y,
    q_heading,
    r_omega
)


print("A matrix:")
print(A)

print()

print("B matrix:")
print(B)

print()

print("LQR gain K:")
print(K)


controllability_matrix = np.hstack([
    B,
    A @ B
])


controllability_rank = np.linalg.matrix_rank(
    controllability_matrix
)


print()

print("Controllability matrix:")
print(controllability_matrix)

print()

print("Controllability rank:")
print(controllability_rank)


closed_loop_matrix = (
    A - B @ K
)


closed_loop_poles = np.linalg.eigvals(
    closed_loop_matrix
)


print()

print("Closed-loop poles:")
print(closed_loop_poles)