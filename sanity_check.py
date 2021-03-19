from simulation import *

# Simulate some exact data
anchors = np.array(
                      [
                          [0, -1620, -150],
                          [1800 * np.cos(np.pi / 4), 1800 * np.sin(np.pi / 4), -150],
                          [-1620 * np.cos(np.pi / 6), 1620 * np.sin(np.pi / 6), -150],
                          [0, 0, 2350],
                      ]
                  )
pos = positions(3, 200, 0)
how_many = 6
pos = pos[:how_many]
print("pos=\n",pos)
radii = np.array([65.0, 65.0, 65.0, 65.0])
samp = motor_pos_samples_with_spool_buildup_compensation(anchors, pos, 0.008, radii)
solution = solve(samp, pos, "SLSQP")

np.set_printoptions(precision=12)
np.set_printoptions(suppress=True)  # No scientific notation
print("Anchor errors:")
print(anchorsvec2matrix(solution[:12]) - anchors)
print("Buildup factor error:")
print(np.array([solution[12] - 0.008]))
print("Spool radii errors:")
print(solution[13:] - radii)
