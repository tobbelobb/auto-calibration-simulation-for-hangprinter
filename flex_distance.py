import numpy as np


def unit_vectors_along_force(anch_to_pos, distances):
    four_direction_vectors_for_each_position = np.ones(
        (np.size(anch_to_pos, 0), np.size(anch_to_pos, 2), np.size(anch_to_pos, 1))
    )
    if (abs(distances) > 0).all():
        four_direction_vectors_for_each_position = np.transpose(
            np.divide(anch_to_pos, distances[:, :, np.newaxis]), (0, 2, 1)
        )
    return four_direction_vectors_for_each_position


def forces_gravity_and_pretension(abc_axis_max_force, abc_axis_target_force, anch_to_pos, distances, mover_weight):
    mg = 9.81 * mover_weight

    force_directions = unit_vectors_along_force(anch_to_pos, distances)

    D_mg = mg / force_directions[:, 2, 3]
    ABC_matrices = force_directions[:, :, 0:3]

    # Scale the d-direction vectors by the target_force
    D_pre = abc_axis_target_force * force_directions[:, :, 3]
    D_grav = np.c_[force_directions[:, :3, 3] * D_mg[:, np.newaxis]]
    grav = np.array([0, 0, -mg])

    # Find the ABC forces needed to cancel out target_force in D-direction
    ABC_forces_pre = np.ones((np.size(distances, 0), 3))
    ABC_forces_grav = np.ones((np.size(distances, 0), 3))
    try:
        ABC_forces_pre = np.linalg.solve(ABC_matrices, -D_pre)
        ABC_forces_grav = np.linalg.solve(ABC_matrices, -D_grav - grav)
    except:
        pass

    forces = [ABC_forces_pre, np.linalg.norm(D_pre, 2, 1), ABC_forces_grav, np.linalg.norm(D_grav, 2, 1)]

    return forces


def forces_gravity_and_pretension_scaled(
    abc_axis_max_force, abc_axis_target_force, anch_to_pos, distances, mover_weight
):

    [ABC_forces_pre, D_forces_pre, ABC_forces_grav, D_forces_grav] = forces_gravity_and_pretension(
        abc_axis_max_force, abc_axis_target_force, anch_to_pos, distances, mover_weight
    )

    # Make ABC_axes pull with exactly max force, or less
    scale_it = np.min(
        np.c_[
            np.max(
                abs(
                    (abc_axis_target_force - np.c_[ABC_forces_grav, D_forces_grav])
                    / np.c_[ABC_forces_pre, D_forces_pre]
                ),
                1,
            ),
            np.min(
                abs((abc_axis_max_force - np.c_[ABC_forces_grav, D_forces_grav]) / np.c_[ABC_forces_pre, D_forces_pre]),
                1,
            ),
        ],
        1,
    )

    ABC_forces_pre = ABC_forces_pre * scale_it[:, np.newaxis]
    D_forces_pre = D_forces_pre * scale_it

    forces = np.c_[ABC_forces_pre + ABC_forces_grav, D_forces_pre + D_forces_grav]
    # forces = np.c_[ABC_forces_grav, D_forces_grav]
    # forces = np.c_[ABC_forces_pre, D_forces_pre]
    forces = np.clip(forces, 0, np.max(forces))

    return forces


def flex_distance(
    abc_axis_max_force, abc_axis_target_force, anchors, pos, mechanical_advantage, springKPerUnitLength, mover_weight
):
    guyWireLengths = np.array(
        [
            np.linalg.norm(anchors[0] - anchors[3]),
            np.linalg.norm(anchors[1] - anchors[3]),
            np.linalg.norm(anchors[2] - anchors[3]),
            0.0,
        ]
    )
    # Insert the origin as the first position always
    # It will be used for computing relative effects of flex later
    pos_w_origin = np.r_[[[0.0, 0.0, 0.0]], pos]
    anch_to_pos = anchors - pos_w_origin[:, np.newaxis, :]
    distances = np.linalg.norm(anch_to_pos, 2, 2)

    forces = forces_gravity_and_pretension_scaled(
        abc_axis_max_force, abc_axis_target_force, anch_to_pos, distances, mover_weight
    )

    springKs = springKPerUnitLength / (distances * mechanical_advantage + guyWireLengths)
    distances_with_relaxed_springs = distances - forces / (springKs * mechanical_advantage)

    line_pos = distances_with_relaxed_springs - distances_with_relaxed_springs[0]

    distance_differences = distances - distances[0]
    impact_of_spring_model = line_pos - distance_differences

    return impact_of_spring_model[1:]
