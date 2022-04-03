#!/usr/bin/python3

from simulation import *

# Simulate some exact data
anchors = np.array(
    [
        [10, -1620, -170],
        [1800 * np.cos(np.pi / 4), 1800 * np.sin(np.pi / 4), -150],
        [-1620 * np.cos(np.pi / 6), 1620 * np.sin(np.pi / 6), -150],
        [20, 30, 2350],
    ]
)
pos = positions(3, 200, 0)
how_many = 11
pos = pos[:how_many]
print("pos=\n", pos)
radii = np.array([75.0, 75.0, 75.0, 75.0])
samp = motor_pos_samples_with_spool_buildup_compensation(anchors, pos, constant_spool_buildup_factor, radii)
line_lengths_origin = np.linalg.norm(anchors, 2, 1)
solution = solve(samp, pos, line_lengths_origin, "SLSQP", True)

np.set_printoptions(precision=12)
np.set_printoptions(suppress=True)  # No scientific notation
print("Anchor errors:")
print(anchorsvec2matrix(solution[:12]) - anchors)
print("Spool radii errors:")
print(solution[12:] - radii)
