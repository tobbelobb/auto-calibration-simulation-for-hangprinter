#!/usr/bin/python3

from simulation import *

# Simulate some exact data
anchors = np.array(
    [
        [10, -1620, -170],
        [1800, 0, -150],
        [0, 1620, -150],
        [-1800, 0, -150],
        [20, 30, 2350],
    ]
)

pos = positions(4, 500, 50)[:6]
how_many_pos = 3
# print("pos=\n", pos)
radii = np.array([75.0, 75.0, 75.0, 75.0, 75.0])
use_flex = True
samp = pos_to_motor_pos_samples(anchors, pos, 20, use_flex, constant_spool_buildup_factor, radii)

line_lengths_origin = np.linalg.norm(anchors, 2, 1)
use_line_lengths = True
debug = True
solution = solve(samp, pos[:how_many_pos], line_lengths_origin, use_flex, use_line_lengths, debug)

np.set_printoptions(precision=12)
np.set_printoptions(suppress=True)  # No scientific notation
print("Anchor errors:")
print(anchorsvec2matrix(solution[:params_anch]) - anchors)
print("Spool radii errors:")
print("[", solution[-5 - use_flex] - radii[0], ", ", solution[-4 - use_flex] - radii[4], "]")
print("Full solution:")
print(solution)
