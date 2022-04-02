import numpy as np

def flex_distance(abc_axis_max_force, anchors, pos, mechanical_advantage, springKPerUnitLength, mover_weight):
    mg = 9.81*mover_weight
    guyWireLengths = np.array([
                        np.linalg.norm(anchors[0] - anchors[3]),
                        np.linalg.norm(anchors[1] - anchors[3]),
                        np.linalg.norm(anchors[2] - anchors[3]),
                        0.0])
    # Insert the origin as the first position always
    # It will be used for computing relative effects of flex later
    pos_w_origin = np.r_[[[0., 0., 0.]], pos]
    anch_to_pos = anchors - pos_w_origin[:, np.newaxis, :]
    distances = np.linalg.norm(anch_to_pos, 2, 2)

    four_direction_vectors_for_each_position = np.transpose(np.divide(anch_to_pos, distances[:,:,np.newaxis]), (0, 2, 1))

    D_mg = mg/four_direction_vectors_for_each_position[:,2,3]

    ABC_matrices = four_direction_vectors_for_each_position[:,:,0:3]

    # Scale the d-direction vectors by the target_force
    D_pre = -abc_axis_max_force * four_direction_vectors_for_each_position[:,:,3]
    D_grav = -np.c_[four_direction_vectors_for_each_position[:,:2,3]*D_mg[:,np.newaxis], np.zeros((np.size(D_mg,0)))]

    # Find the ABC forces needed to cancel out target_force in D-direction
    ABC_forces_pre = np.ones(pos_w_origin.shape)
    ABC_forces_grav = np.ones(pos_w_origin.shape)
    try:
        ABC_forces_pre = np.linalg.solve(ABC_matrices, D_pre)
        ABC_forces_grav = np.linalg.solve(ABC_matrices, D_grav)
    except:
        pass

    # Make ABC_axes pull with max force, or less
    scale_it =  abc_axis_max_force/np.max(ABC_forces_pre, 1)[:,np.newaxis]

    ABC_forces_pre = ABC_forces_pre * scale_it
    D_forces_pre = np.linalg.norm(D_pre, 2, 1) * scale_it[:,0]
    forces = np.c_[ABC_forces_pre + ABC_forces_grav, D_forces_pre + np.linalg.norm(D_grav,2,1)]

    springKs = springKPerUnitLength/(distances*mechanical_advantage + guyWireLengths)
    distances_with_relaxed_springs = distances - forces/(springKs * mechanical_advantage)

    line_pos = distances_with_relaxed_springs - distances_with_relaxed_springs[0]

    distance_differences = distances - distances[0]
    impact_of_spring_model = line_pos - distance_differences

    return impact_of_spring_model[1:]
