"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division  # Always want 3/2 = 1.5
import numpy as np
import scipy.optimize
import argparse
import timeit
import sys

# Axes indexing
A = 0
B = 1
C = 2
D = 3
X = 0
Y = 1
Z = 2
params_anch = 9
params_buildup = 5  # four spool radii, one spool buildup factor
A_bx = 2
A_cx = 5


def symmetric_anchors(l, az=-120.0, bz=-120.0, cz=-120.0):
    anchors = np.array(np.zeros((4, 3)))
    anchors[A, Y] = -l
    anchors[A, Z] = az
    anchors[B, X] = l * np.cos(np.pi / 6)
    anchors[B, Y] = l * np.sin(np.pi / 6)
    anchors[B, Z] = bz
    anchors[C, X] = -l * np.cos(np.pi / 6)
    anchors[C, Y] = l * np.sin(np.pi / 6)
    anchors[C, Z] = cz
    anchors[D, Z] = l
    return anchors


def centered_rand(l):
    """Sample from U(-l, l)"""
    return l * (2.0 * np.random.rand() - 1.0)


def irregular_anchors(l, fuzz_percentage=0.2, az=-120.0, bz=-120.0, cz=-120.0):
    """Realistic exact positions of anchors.

    Each dimension of each anchor is treated separately to
    resemble the use case.
    Six anchor coordinates must be constant and known
    for the coordinate system to be uniquely defined by them.
    A 3d coordinate system, like a rigid body, has six degrees of freedom.

    Parameters
    ---------
    l : The line length to create the symmetric anchors first
    fuzz_percentage : Percentage of l that line lenghts are allowed to differ
                      (except Z-difference of B- and C-anchors)
    """
    fuzz = np.array(np.zeros((4, 3)))
    fuzz[A, Y] = centered_rand(l * fuzz_percentage)
    # fuzz[A, Z] = 0 # Fixated
    fuzz[B, X] = centered_rand(l * fuzz_percentage * np.cos(np.pi / 6))
    fuzz[B, Y] = centered_rand(l * fuzz_percentage * np.sin(np.pi / 6))
    # fuzz[B, Z] = 0 # Fixated
    fuzz[C, X] = centered_rand(l * fuzz_percentage * np.cos(np.pi / 6))
    fuzz[C, Y] = centered_rand(l * fuzz_percentage * np.sin(np.pi / 6))
    # fuzz[C, Z] = 0 # Fixated
    # fuzz[D, X] = 0 # Fixated
    # fuzz[D, Y] = 0 # Fixated
    fuzz[D, Z] = l * fuzz_percentage * np.random.rand()  # usually higher than A is long
    return symmetric_anchors(l, az, bz, cz) + fuzz


def positions(n, l, fuzz=10):
    """Return (n^3)x3 matrix of positions in fuzzed grid of side length 2*l

    Move to u=n^3 positions in an fuzzed grid of side length 2*l
    centered around (0, 0, l).

    Parameters
    ----------
    n : Number of positions of which to sample along each axis
    l : Max length from origin along each axis to sample
    fuzz: How much each measurement point can differ from the regular grid
    """
    from itertools import product

    pos = (
        np.array(list(product(np.linspace(-l, l, n), repeat=3)))
        + 2.0 * fuzz * (np.random.rand(n ** 3, 3) - 0.5)
        + [0, 0, 1 * l]
    )
    index_closest_to_origin = np.int(np.shape(pos)[0] / 2) - int(n / 2)
    # Make pos[0] a point fairly close to origin
    tmp = pos[0].copy()
    pos[0] = pos[index_closest_to_origin]
    pos[index_closest_to_origin] = tmp
    return pos


def samples(anchors, pos, fuzz=1):
    """Possible relative line length measurments according to anchors and position.

    Parameters
    ----------
    anchors : 4x3 matrix of anhcor positions in mm
    pos : ux3 matrix of positions
    fuzz: Maximum measurment error per motor in mm
    """
    # pos[:,np.newaxis,:]: ux1x3
    # Broadcasting happens u times and we get ux4x3 output before norm operation
    line_lengths = np.linalg.norm(anchors - pos[:, np.newaxis, :], 2, 2)
    return (
        line_lengths
        - line_lengths[0]
        + 2.0 * fuzz * (np.random.rand(np.shape(pos)[0], 1) - 0.5)
    )


def samples_relative_to_origin(anchors, pos, fuzz=1):
    """Possible relative line length measurments according to anchors and position.

    Parameters
    ----------
    anchors : 4x3 matrix of anhcor positions in mm
    pos : ux3 matrix of positions
    fuzz: Maximum measurment error per motor in mm
    """
    # pos[:,np.newaxis,:]: ux1x3
    # Broadcasting happens u times and we get ux4x3 output before norm operation
    line_lengths = np.linalg.norm(anchors - pos[:, np.newaxis, :], 2, 2)
    return (
        line_lengths
        - np.linalg.norm(anchors, 2, 1)
        + 2.0 * fuzz * (np.random.rand(np.shape(pos)[0], 1) - 0.5)
    )


def samples_relative_to_origin_no_fuzz(anchors, pos):
    """Possible relative line length measurments according to anchors and position.

    Parameters
    ----------
    anchors : 4x3 matrix of anhcor positions in mm
    pos : ux3 matrix of positions
    fuzz: Maximum measurment error per motor in mm
    """
    # pos[:,np.newaxis,:]: ux1x3
    # Broadcasting happens u times and we get ux4x3 output before norm operation
    line_lengths = np.linalg.norm(anchors - pos[:, np.newaxis, :], 2, 2)
    return line_lengths - np.linalg.norm(anchors, 2, 1)


def motor_pos_samples_with_spool_buildup_compensation(
    anchors,
    pos,
    spool_buildup_factor=0.008,  # Qualified first guess for 0.5 mm line
    spool_r_in_origin=np.array([65.0, 65.0, 65.0, 65.0]),
    spool_to_motor_gearing_factor=12.75,
    mech_adv=np.array([2.0, 2.0, 2.0, 2.0]),
    number_of_lines_per_spool=np.array([1.0, 1.0, 1.0, 1.0]),
):
    """What motor positions (in degrees) motors would be at
    """
    # Assure np.array type
    spool_r_in_origin = np.array(spool_r_in_origin)
    mech_adv = np.array(mech_adv)
    number_of_lines_per_spool = np.array(number_of_lines_per_spool)

    spool_r_in_origin_sq = spool_r_in_origin * spool_r_in_origin

    # Buildup per line times lines. Minus sign because more line in air means less line on spool
    k2 = -mech_adv * number_of_lines_per_spool * spool_buildup_factor

    # we now want to use degrees instead of steps as unit of rotation
    # so setting 360 where steps per motor rotation is in firmware buildup compensation algorithms
    degrees_per_unit_times_r = (spool_to_motor_gearing_factor * mech_adv * 360.0) / (
        2.0 * np.pi
    )
    k0 = 2.0 * degrees_per_unit_times_r / k2

    line_lengths_origin = np.linalg.norm(anchors - np.array([[[0, 0, 0]]]), 2, 2)

    relative_line_lengths = samples_relative_to_origin_no_fuzz(anchors, pos)
    motor_positions = k0 * (
        np.sqrt(spool_r_in_origin_sq + relative_line_lengths * k2) - spool_r_in_origin
    )

    return motor_positions


def motor_pos_samples_to_line_length_with_buildup_compensation(
    motor_samps,
    spool_buildup_factor=0.008,  # Qualified first guess for 0.5 mm line
    spool_r=np.array([65.0, 65.0, 65.0, 65.0]),
    spool_to_motor_gearing_factor=12.75,  # HP4 default (255/20)
    mech_adv=np.array([2.0, 2.0, 2.0, 2.0]),  # HP4 default
    number_of_lines_per_spool=np.array([1.0, 1.0, 1.0, 1.0]),  # HP4 default
):
    # Buildup per line times lines. Minus sign because more line in air means less line on spool
    c1 = -mech_adv * number_of_lines_per_spool * spool_buildup_factor

    # we now want to use degrees instead of steps as unit of rotation
    # so setting 360 where steps per motor rotation is in firmware buildup compensation algorithms
    degrees_per_unit_times_r = (spool_to_motor_gearing_factor * mech_adv * 360.0) / (
        2.0 * np.pi
    )
    k0 = 2.0 * degrees_per_unit_times_r / c1

    return (((motor_samps / k0) + spool_r) ** 2 - spool_r * spool_r) / c1


def cost(anchors, pos, samp):
    """If all positions and samples correspond perfectly, this returns 0.

    This is the systems of equations:
    sum for i from 1 to u
      sum for k from a to d
    |sqrt(sum for s from x to z (A_ks-s_i)^2) - sqrt(sum for s from x to z (A_ks-s_0)^2) - t_ik|

    or...
    sum for i from 1 to u
    |sqrt((A_ax-x_i)^2 + (A_ay-y_i)^2 + (A_az-z_i)^2) - sqrt((A_ax-x_0)^2 + (A_ay-y_0)^2 + (A_az-z_0)^2) - t_ia| +
    |sqrt((A_bx-x_i)^2 + (A_by-y_i)^2 + (A_bz-z_i)^2) - sqrt((A_bx-x_0)^2 + (A_by-y_0)^2 + (A_bz-z_0)^2) - t_ib| +
    |sqrt((A_cx-x_i)^2 + (A_cy-y_i)^2 + (A_cz-z_i)^2) - sqrt((A_cx-x_0)^2 + (A_cy-y_0)^2 + (A_cz-z_0)^2) - t_ic| +
    |sqrt((A_dx-x_i)^2 + (A_dy-y_i)^2 + (A_dz-z_i)^2) - sqrt((A_dx-x_0)^2 + (A_dy-y_0)^2 + (A_dz-z_0)^2) - t_id|

    Parameters
    ---------
    anchors : 4x3 matrix of anchor positions
    pos: ux3 matrix of positions
    samp : ux4 matrix of corresponding samples, starting with [0., 0., 0., 0.]
    """
    return np.sum(np.abs(samples_relative_to_origin_no_fuzz(anchors, pos) - samp))


def cost_sq(anchors, pos, samp):
    """
    For all samples sum
    (Sample value if anchor position A and cartesian position x were guessed   - actual sample)^2

    (sqrt((A_ax-x_i)^2 + (A_ay-y_i)^2 + (A_az-z_i)^2) - sqrt(A_ax^2 + A_ay^2 + A_az^2) - t_ia)^2 +
    (sqrt((A_bx-x_i)^2 + (A_by-y_i)^2 + (A_bz-z_i)^2) - sqrt(A_bx^2 + A_by^2 + A_bz^2) - t_ib)^2 +
    (sqrt((A_cx-x_i)^2 + (A_cy-y_i)^2 + (A_cz-z_i)^2) - sqrt(A_cx^2 + A_cy^2 + A_cz^2) - t_ic)^2 +
    (sqrt((A_dx-x_i)^2 + (A_dy-y_i)^2 + (A_dz-z_i)^2) - sqrt(A_dx^2 + A_dy^2 + A_dz^2) - t_id)^2
    """
    return np.sum(pow((samples_relative_to_origin_no_fuzz(anchors, pos) - samp), 2))


def cost_for_pos_samp(anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r):
    """
    For all samples sum
    (Sample value if anchor position A and cartesian position x were guessed   - actual sample)^2

    (sqrt((A_ax-x_i)^2 + (A_ay-y_i)^2 + (A_az-z_i)^2) - sqrt(A_ax^2 + A_ay^2 + A_az^2) - motor_pos_to_samp(t_ia))^2 +
    (sqrt((A_bx-x_i)^2 + (A_by-y_i)^2 + (A_bz-z_i)^2) - sqrt(A_bx^2 + A_by^2 + A_bz^2) - motor_pos_to_samp(t_ib))^2 +
    (sqrt((A_cx-x_i)^2 + (A_cy-y_i)^2 + (A_cz-z_i)^2) - sqrt(A_cx^2 + A_cy^2 + A_cz^2) - motor_pos_to_samp(t_ic))^2 +
    (sqrt((A_dx-x_i)^2 + (A_dy-y_i)^2 + (A_dz-z_i)^2) - sqrt(A_dx^2 + A_dy^2 + A_dz^2) - motor_pos_to_samp(t_id))^2
    """
    return np.sum(
        np.abs(
            samples_relative_to_origin_no_fuzz(anchors, pos)
            - motor_pos_samples_to_line_length_with_buildup_compensation(
                motor_pos_samp, spool_buildup_factor, spool_r
            )
        )
    )


def cost_sq_for_pos_samp(anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r):
    """
    Sum of squares

    For all samples sum
    (Sample value if anchor position A and cartesian position x were guessed   - actual sample)^2

    (sqrt((A_ax-x_i)^2 + (A_ay-y_i)^2 + (A_az-z_i)^2) - sqrt(A_ax^2 + A_ay^2 + A_az^2) - motor_pos_to_samp(t_ia))^2 +
    (sqrt((A_bx-x_i)^2 + (A_by-y_i)^2 + (A_bz-z_i)^2) - sqrt(A_bx^2 + A_by^2 + A_bz^2) - motor_pos_to_samp(t_ib))^2 +
    (sqrt((A_cx-x_i)^2 + (A_cy-y_i)^2 + (A_cz-z_i)^2) - sqrt(A_cx^2 + A_cy^2 + A_cz^2) - motor_pos_to_samp(t_ic))^2 +
    (sqrt((A_dx-x_i)^2 + (A_dy-y_i)^2 + (A_dz-z_i)^2) - sqrt(A_dx^2 + A_dy^2 + A_dz^2) - motor_pos_to_samp(t_id))^2
    """
    return np.sum(
        pow(
            samples_relative_to_origin_no_fuzz(anchors, pos)
            - motor_pos_samples_to_line_length_with_buildup_compensation(
                motor_pos_samp, spool_buildup_factor, spool_r
            ),
            2,
        )
    )


def cost_sqsq_for_pos_samp(anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r):
    """Sum of squared squares"""
    return np.sum(
        pow(
            samples_relative_to_origin_no_fuzz(anchors, pos)
            - motor_pos_samples_to_line_length_with_buildup_compensation(
                motor_pos_samp, spool_buildup_factor, spool_r
            ),
            4,
        )
    )


def cost_sqsqsq_for_pos_samp(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r
):
    """Sum of squared squared squares"""
    return np.sum(
        pow(
            samples_relative_to_origin_no_fuzz(anchors, pos)
            - motor_pos_samples_to_line_length_with_buildup_compensation(
                motor_pos_samp, spool_buildup_factor, spool_r
            ),
            8,
        )
    )


def anchorsvec2matrix(anchorsvec):
    """ Create a 4x3 anchors matrix from 6 element anchors vector.
    """
    anchors = np.array(np.zeros((4, 3)))
    anchors[A, Y] = anchorsvec[0]
    anchors[A, Z] = anchorsvec[1]
    anchors[B, X] = anchorsvec[2]
    anchors[B, Y] = anchorsvec[3]
    anchors[B, Z] = anchorsvec[4]
    anchors[C, X] = anchorsvec[5]
    anchors[C, Y] = anchorsvec[6]
    anchors[C, Z] = anchorsvec[7]
    anchors[D, Z] = anchorsvec[8]
    return anchors


def anchorsmatrix2vec(a):
    return [
        a[A, Y],
        a[A, Z],
        a[B, X],
        a[B, Y],
        a[B, Z],
        a[C, X],
        a[C, Y],
        a[C, Z],
        a[D, Z],
    ]


def posvec2matrix(v, u):
    return np.reshape(v, (u, 3))


def posmatrix2vec(m):
    return np.reshape(m, np.shape(m)[0] * 3)


def pre_list(l, num):
    return np.append(np.append(l[0:params_anch], l[params_anch:params_anch + 3*num]), l[-params_buildup :])

def solve(motor_pos_samp, xyz_of_samp, method, cx_is_positive=False):
    """Find reasonable positions and anchors given a set of samples.
    """

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    number_of_params_pos = 3 * (u - ux)

    def costx(_cost, posvec, anchvec, spool_buildup_factor, spool_r, u):
        """Identical to cost, except the shape of inputs and capture of samp, xyz_of_samp, ux, and u

        Parameters
        ----------
        x : [A_ay A_az A_bx A_by A_bz A_cx A_cy A_cz A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """
        anchors = anchorsvec2matrix(anchvec)
        pos = np.zeros((u, 3))
        if np.size(xyz_of_samp) != 0:
            pos[0:ux] = xyz_of_samp
        pos[ux:] = np.reshape(posvec, (u - ux, 3))
        return _cost(
            anchors, pos, motor_pos_samp[:u], spool_buildup_factor, np.array(spool_r)
        )

    l_long = 4000.0
    l_short = 1700.0
    data_z_min = -50.0
    # Limits of anchor positions:
    #     |ANCHOR_XY|    < 4000
    #      ANCHOR_B_X    > 0
    #      ANCHOR_C_X    < 0
    #     |ANCHOR_ABC_Z| < 1700
    # 0 <  ANCHOR_D_Z    < 4000
    # Limits of data collection volume:
    #         |x| < 1700
    #         |y| < 1700
    # -20.0 <  z  < 3400.0
    # Define bounds
    #lb = (
    #    [
    #        -l_long,  # A_ay > -4000.0
    #        -300.0,  # A_az > -300.0
    #        300.0,  # A_bx > 300
    #        300.0,  # A_by > 300
    #        -300.0,  # A_bz > -300.0
    #        -l_long,  # A_cx > -4000
    #        300.0,  # A_cy > 300
    #        -300.0,  # A_cz > -300.0
    #        1000.0,  # A_dz > 1000
    #    ]
    #    + [-l_short, -l_short, data_z_min] * (u - ux)
    #    + [0.00005, 64.6, 64.6, 64.6, 64.6]
    #)
    #ub = (
    #    [
    #        500.0,  # A_ay < 500
    #        200.0,  # A_az < 200
    #        l_long,  # A_bx < 4000
    #        l_long,  # A_by < 4000
    #        200.0,  # A_bz < 200
    #        -300.0,  # A_cx < -300
    #        l_long,  # A_cy < 4000.0
    #        200.0,  # A_cz < 200
    #        l_long,  # A_dz < 4000.0
    #    ]
    #    + [l_short, l_short, 2 * l_short] * (u - ux)
    #    + [0.01, 67, 67, 67, 67]
    #)
    lb = (
        [
            -2000.0,  # A_ay > -2000.0
            -200.0,  # A_az > -200.0
            1000.0,  # A_bx > 1000
            900.0,  # A_by > 900
            -200.0,  # A_bz > -200.0
            -2000.0,  # A_cx > -2000
            700.0,  # A_cy > 700
            -200.0,  # A_cz > -200.0
            2300.0,  # A_dz > 2300
        ]
        + [-l_short, -l_short, data_z_min] * (u - ux)
        + [0.00005, 64.6, 64.6, 64.6, 64.6]
    )
    ub = (
        [
            -1600.0,  # A_ay < -1600
            0.0,  # A_az < 0
            1600.0,  # A_bx < 1600
            1500.0,  # A_by < 1500
            0.0,  # A_bz < 200
            -1300.0,  # A_cx < -1300
            1000.0,  # A_cy < 1000.0
            0.0,  # A_cz < 0
            2700.0,  # A_dz < 2700.0
        ]
        + [l_short, l_short, 2 * l_short] * (u - ux)
        + [0.01, 67, 67, 67, 67]
    )
    #lb = (
    #    [
    #        -999999.0,  # A_ay > -4000.0
    #        -999999.0,  # A_az > -300.0
    #        -999999.0,  # A_bx > 300
    #        -999999.0,  # A_by > 300
    #        -999999.0,  # A_bz > -300.0
    #        -999999.0,  # A_cx > -4000
    #        -999999.0,  # A_cy > 300
    #        -999999.0,  # A_cz > -300.0
    #        -999999.0,  # A_dz > 1000
    #    ]
    #    + [-999999.0, -999999.0, -999999.0] * (u - ux)
    #    + [0.0001, 60.5, 60.5, 60.5, 60.5]
    #)
    #ub = (
    #    [
    #        999999.9,  # A_ay < 500
    #        999999.9,  # A_az < 200
    #        999999.9,  # A_bx < 4000
    #        999999.9,  # A_by < 4000
    #        999999.9,  # A_bz < 200
    #        999999.9,  # A_cx < -300
    #        999999.9,  # A_cy < 4000.0
    #        999999.9,  # A_cz < 200
    #        999999.9,  # A_dz < 4000.0
    #    ]
    #    + [999999.0, 999999.0, 999999.0] * (u - ux)
    #    + [100, 670, 670, 670, 670]
    #)

    # If the user has input xyz data, then signs should be ok anyways
    if ux > 2:
        lb[A_bx] = -l_long

    # It would work to just swap the signs of bx and cx after the optimization
    # But there are fewer assumptions involved in setting correct bounds from the start instead
    if cx_is_positive:
        tmp = lb[A_bx]
        lb[A_bx] = lb[A_cx]
        lb[A_cx] = tmp
        tmp = ub[A_bx]
        ub[A_bx] = ub[A_cx]
        ub[A_cx] = tmp

    pos_est = np.zeros((u - ux, 3))  # The positions we need to estimate
    anchors_est = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    x_guess = (
        list(anchorsmatrix2vec(anchors_est))
        + list(posmatrix2vec(pos_est))
        + [0.006, 64.5, 64.5, 64.5, 64.5]
    )

    def constraints(x):
        for samp_num in range(u):
            # If A motor wound in line
            # Then the position of measurment has negative y coordinate
            if motor_pos_samp[samp_num][A] < 0.0:
                x[params_anch + samp_num*3 + Y] = -np.abs(x[params_anch + samp_num*3 + Y])
            # If both B and C wound in line
            # Then position of measurement has positive Y coordinate
            elif motor_pos_samp[samp_num][B] < 0.0 and motor_pos_samp[samp_num][C] < 0.0:
                x[params_anch + samp_num*3 + Y] = np.abs(x[params_anch + samp_num*3 + Y])
            # If both A and B wound in line
            # Then pos of meas has positive X coord
            if motor_pos_samp[samp_num][A] < 0.0 and motor_pos_samp[samp_num][B] < 0.0:
                x[params_anch + samp_num*3 + X] = np.abs(x[params_anch + samp_num*3 + X])
            # If both A and C wound in line
            # Then pos of meas has negative X coord
            elif motor_pos_samp[samp_num][A] < 0.0 and motor_pos_samp[samp_num][C] < 0.0:
                x[params_anch + samp_num*3 + X] = -np.abs(x[params_anch + samp_num*3 + X])
        return x


    if method == "combo":  # Method 3
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG

        term = COG(generations=100)
        solver = PowellDirectionalSolver(
            number_of_params_pos + params_anch + params_buildup
        )
        solver.SetEvaluationLimits(evaluations=3200000, generations=10000)
        solver.SetTermination(term)
        solver.SetInitialPoints(x_guess)
        solver.SetStrictRanges(lb, ub)
        solver.Solve(
            lambda x: costx(
                cost_sqsq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            )
        )
        x_guess = solver.bestSolution
        old_lb = np.copy(lb)
        old_ub = np.copy(ub)
        for i in range(1, 10):
            print(".")
            solver = PowellDirectionalSolver(
                number_of_params_pos + params_anch + params_buildup
            )
            solver.SetInitialPoints(x_guess)
            lb[:-params_buildup] = (
                x_guess[:-params_buildup]
                - np.abs(x_guess[:-params_buildup] - lb[:-params_buildup]) / 1.5
            )
            ub[:-params_buildup] = (
                x_guess[:-params_buildup]
                + np.abs(x_guess[:-params_buildup] - ub[:-params_buildup]) / 1.5
            )
            solver.SetStrictRanges(lb, ub)
            solver.Solve(
                lambda x: costx(
                    cost_sqsqsq_for_pos_samp,
                    x[params_anch:-params_buildup],
                    x[0:params_anch],
                    x[-params_buildup],
                    x[-params_buildup + 1 :],
                    u
                )
            )
            x_guess = solver.bestSolution
            x_guess = scipy.optimize.minimize(
                lambda x: costx(
                    cost_for_pos_samp,
                    x[params_anch:-params_buildup],
                    x[0:params_anch],
                    x[-params_buildup],
                    x[-params_buildup + 1 :],
                    u
                ),
                x_guess,
                method="SLSQP",
                bounds=list(zip(old_lb, old_ub)),
                options={"disp": True, "ftol": 1e-10, "maxiter": 15000},
            ).x

        return x_guess

    elif method == "differentialEvolutionSolver":
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, ChangeOverGeneration, And, Or

        stop = Or(VTR(1e-12), ChangeOverGeneration(0.0000001, 5000))
        ndim = number_of_params_pos + params_anch + params_buildup
        npop = 30*ndim
        stepmon = VerboseMonitor(100)
        solver = DifferentialEvolutionSolver2(ndim,npop)
        solver.SetRandomInitialPoints(lb, ub)
        solver.SetStrictRanges(lb, ub)
        solver.SetConstraints(constraints)
        solver.SetGenerationMonitor(stepmon)
        solver.enable_signal_handler() # Handle Ctrl+C gracefully. Be restartable
        solver.Solve(lambda x: costx(
                cost_sq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            ), termination=stop)

        # use monitor to retrieve results information
        iterations = len(stepmon)
        cost = stepmon.y[-1]
        print("Generation %d has best Chi-Squared: %f" % (iterations, cost))
        return solver.Solution()

    elif method == "BuckShot":
        try:
            from pathos.helpers import freeze_support
            freeze_support()
            from pathos.pools import ProcessPool as Pool
        except ImportError:
            from mystic.pools import SerialPool as Pool
        from mystic.termination import VTR, ChangeOverGeneration as COG
        from mystic.termination import Or
        from mystic.termination import NormalizedChangeOverGeneration as NCOG
        from mystic.monitors import LoggingMonitor, VerboseMonitor, Monitor
        from klepto.archives import dir_archive
        stop = NCOG(1e-4)
        archive = False # save an archive
        ndim = number_of_params_pos + params_anch + params_buildup
        from mystic.solvers import BuckshotSolver, LatticeSolver
        # the local solvers
        from mystic.solvers import PowellDirectionalSolver
        sprayer = BuckshotSolver
        term = Or(VTR(1e-9), COG(0.01, 50))
        seeker = PowellDirectionalSolver(ndim)
        seeker.SetConstraints(constraints)
        seeker.SetTermination(term)
        seeker.SetEvaluationLimits(evaluations=320000, generations=10000)
        seeker.SetStrictRanges(lb, ub)
        seeker.enable_signal_handler() # Handle Ctrl+C gracefully. Be restartable
        npts = 3 # number of solvers
        _map = Pool().map
        retry = 1 # max consectutive iteration retries without a cache 'miss'
        tol = 8   # rounding precision
        mem = 1 # cache rounding precision
        from mystic.search import Searcher
        searcher = Searcher(npts, retry, tol, mem, _map, None, sprayer, seeker)
        searcher.Verbose(True)
        searcher.UseTrajectories(False)
        #searcher.Reset(None, inv=False)
        searcher.Search(
            lambda x: costx(
                cost_sq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            ),
            bounds=list(zip(lb, ub)),
            stop=stop,
            monitor=LoggingMonitor(1))#,
            #constraints=constraints)
        searcher._summarize()
        xyz = self.sampler.Samples()

        xyz = np.hstack((xyz, searcher.Samples())) # extract results
        x, z = xyz.T[:,:-1], xyz.T[:,-1]  # get params (x) and cost (z)
        best_cost = 99999999
        best_i = 0
        for i in range(len(z)):
            if z[i] < best_cost:
                best_cost = z[i]
                best_i = i
        return x[best_i]


    elif method == "PowellDirectionalSolver":  # Method 1
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, And, Or


        big_ndim = number_of_params_pos + params_anch + params_buildup

        term = Or(VTR(1e-20), COG(0.01, 50))
        # Solver for both simultaneously
        solver = PowellDirectionalSolver(big_ndim)
        solver.enable_signal_handler() # Handle Ctrl+C gracefully. Be restartable
        solver.SetConstraints(constraints)
        solver.SetGenerationMonitor(VerboseMonitor(5))
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetTermination(term)
        solver.SetInitialPoints(x_guess)
        solver.SetStrictRanges(lb, ub)
        solver.Solve(
            lambda x: costx(
                cost_sqsq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            )
        )
        x_guess = solver.bestSolution

        term = Or(VTR(1e-20), COG(0.01, 50))
        solver = PowellDirectionalSolver(big_ndim)
        solver.enable_signal_handler() # Handle Ctrl+C gracefully. Be restartable
        solver.SetConstraints(constraints)
        solver.SetGenerationMonitor(VerboseMonitor(5))
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetTermination(term)
        solver.SetInitialPoints(x_guess)
        lb[:-params_buildup] = (
            x_guess[:-params_buildup]
            - np.abs(x_guess[:-params_buildup] - lb[:-params_buildup]) / 1.2
        )
        ub[:-params_buildup] = (
            x_guess[:-params_buildup]
            + np.abs(x_guess[:-params_buildup] - ub[:-params_buildup]) / 1.2
        )
        solver.SetStrictRanges(lb, ub)
        solver.Solve(
            lambda x: costx(
                cost_sqsq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            )
        )
        x_guess = solver.bestSolution

        term = Or(VTR(1e-20), COG(0.01, 50))
        solver = PowellDirectionalSolver(big_ndim)
        solver.enable_signal_handler() # Handle Ctrl+C gracefully. Be restartable
        solver.SetConstraints(constraints)
        solver.SetGenerationMonitor(VerboseMonitor(5))
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetTermination(term)
        solver.SetInitialPoints(x_guess)
        lb[:-params_buildup] = (
            x_guess[:-params_buildup]
            - np.abs(x_guess[:-params_buildup] - lb[:-params_buildup]) / 1.2
        )
        ub[:-params_buildup] = (
            x_guess[:-params_buildup]
            + np.abs(x_guess[:-params_buildup] - ub[:-params_buildup]) / 1.2
        )
        solver.SetStrictRanges(lb, ub)
        solver.Solve(
            lambda x: costx(
                cost_sq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            )
        )
        x_guess = solver.bestSolution

        term = Or(VTR(1e-20), COG(0.01, 50))
        solver = PowellDirectionalSolver(big_ndim)
        solver.enable_signal_handler() # Handle Ctrl+C gracefully. Be restartable
        solver.SetConstraints(constraints)
        solver.SetGenerationMonitor(VerboseMonitor(5))
        solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
        solver.SetTermination(term)
        solver.SetInitialPoints(x_guess)
        lb[:-params_buildup] = (
            x_guess[:-params_buildup]
            - np.abs(x_guess[:-params_buildup] - lb[:-params_buildup]) / 1.2
        )
        ub[:-params_buildup] = (
            x_guess[:-params_buildup]
            + np.abs(x_guess[:-params_buildup] - ub[:-params_buildup]) / 1.2
        )
        solver.SetStrictRanges(lb, ub)
        solver.Solve(
            lambda x: costx(
                cost_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            )
        )
        x_guess = solver.bestSolution
        return x_guess

    elif method == "SLSQP":  # Method 2
        x_guess = scipy.optimize.minimize(
            lambda x: costx(
                cost_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            ),
            x_guess,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            #constraints=[{'type': 'eq', 'fun': constraints}], # This doesn't seem to work?
            options={"disp": True, "ftol": 1e-20, "maxiter": 150000},
        ).x
        return x_guess

    elif method == "L-BFGS-B":  # Method 0
        x_guess = scipy.optimize.minimize(
            lambda x: costx(
                cost_sqsqsq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u
            ),
            x_guess,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"disp": False, "ftol": 1e-20, "maxiter": 15000},
        ).x
        return x_guess

    else:
        print("Method %s is not supported!" % method)
        sys.exit(1)


def print_anch(anch):
    print("\n#define ANCHOR_A_Y %5d" % round(anch[A, Y]))
    print("#define ANCHOR_A_Z %5d" % round(anch[A, Z]))
    print("#define ANCHOR_B_X %5d" % round(anch[B, X]))
    print("#define ANCHOR_B_Y %5d" % round(anch[B, Y]))
    print("#define ANCHOR_B_Z %5d" % round(anch[B, Z]))
    print("#define ANCHOR_C_X %5d" % round(anch[C, X]))
    print("#define ANCHOR_C_Y %5d" % round(anch[C, Y]))
    print("#define ANCHOR_C_Z %5d" % round(anch[C, Z]))
    print("#define ANCHOR_D_Z %5d" % round(anch[D, Z]))
    print(
        "\nM665 W%.2f E%.2f R%.2f T%.2f Y%.2f U%.2f I%.2f O%.2f P%.2f"
        % (
            anch[A, Y],
            anch[A, Z],
            anch[B, X],
            anch[B, Y],
            anch[B, Z],
            anch[C, X],
            anch[C, Y],
            anch[C, Z],
            anch[D, Z],
        )
    )


def print_anch_err(sol_anch, anchors):
    print("\nErr_A_Y: %9.3f" % (sol_anch[A, Y] - anchors[A, Y]))
    print("Err_A_Z: %9.3f" % (sol_anch[A, Z] - anchors[A, Z]))
    print("Err_B_X: %9.3f" % (sol_anch[B, X] - anchors[B, X]))
    print("Err_B_Y: %9.3f" % (sol_anch[B, Y] - anchors[B, Y]))
    print("Err_B_Z: %9.3f" % (sol_anch[B, Z] - anchors[B, Z]))
    print("Err_C_X: %9.3f" % (sol_anch[C, X] - anchors[C, X]))
    print("Err_C_Y: %9.3f" % (sol_anch[C, Y] - anchors[C, Y]))
    print("Err_C_Z: %9.3f" % (sol_anch[C, Z] - anchors[C, Z]))
    print("Err_D_Z: %9.3f" % (sol_anch[D, Z] - anchors[D, Z]))


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(Store_as_array, self).__call__(
            parser, namespace, values, option_string
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure out where Hangprinter anchors are by looking at line difference samples."
    )
    parser.add_argument(
        "-d", "--debug", help="Print debug information", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--cx_is_positive",
        help="Use this flag if your C anchor should have a positive X-coordinate",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--method",
        help='Available methods are L-BFGS-B, PowellDirectionalSolver (requires a library called Mystic), SLSQP, a combo of PowellDirectionalSolver and SLSQP, and differentialEvolutionSolver. As a shorthand, you can use 0, 1, 2, or 3 for referring to the four methods respectively. Specify "combo" or "3" if you want the combo. Specify "all" or "4" if you want the script to run all four methods (default). Specify "differentialEvolutionSolver" or "5" if you want the differential evolution solver (slow).',
        default="PowellDirectionalSolver",
    )
    parser.add_argument(
        "-x",
        "--xyz_of_samp",
        help="Specify the XYZ-positions where samples were made as numbers separated by spaces.",
        action=Store_as_array,
        type=float,
        nargs="+",
        default=np.array([]),
    )
    parser.add_argument(
        "-s",
        "--sample_data",
        help="Specify the sample data you have found as numbers separated by spaces.",
        action=Store_as_array,
        type=float,
        nargs="+",
        default=np.array([]),
    )
    args = vars(parser.parse_args())

    if args["method"] == "0":
        args["method"] = "L-BFGS-B"
    if args["method"] == "1":
        args["method"] = "PowellDirectionalSolver"
    if args["method"] == "2":
        args["method"] = "SLSQP"
    if args["method"] == "3":
        args["method"] = "combo"
    if args["method"] == "4":
        args["method"] = "all"
    if args["method"] == "5":
        args["method"] = "differentialEvolutionSolver"
    if args["method"] == "6":
        args["method"] = "BuckShot"
    print(args["method"])

    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    anchors = np.array(
        [
            [0, -1620, -10],
            [1800 * np.cos(np.pi / 5), 1800 * np.sin(np.pi / 5), -10],
            [-1620 * np.cos(np.pi / 6), 1620 * np.sin(np.pi / 6), -10],
            [0, 0, 2350],
        ]
    )

    # a = np.zeros((2,3))
    xyz_of_samp = args["xyz_of_samp"]
    if np.size(xyz_of_samp) != 0:
        if np.size(xyz_of_samp) % 3 != 0:
            print(
                "Error: You specified %d numbers after your -x/--xyz_of_samp option, which is not a multiple of 3 numbers."
            )
            sys.exit(1)
        xyz_of_samp = xyz_of_samp.reshape((int(np.size(xyz_of_samp) / 3), 3))
    else:
        xyz_of_samp = np.array(
            [
                # You might want to manually input positions where you made samples here like
                # [ -13.82573298,  185.92015633, 664.66427937],
                # [-389.81246064, -32.85556323 , 587.55219886],
                # [ 237.76400537, -126.40678778, 239.0320148],
                # [ 143.2309169 ,  -15.59590026,  722.89425101],
                # [-267.39107719, -139.31819633,  934.36563975],
                # [-469.27951032,  102.82165224,  850.67454249],
                # [-469.27950169,  102.82167868,  850.6745381 ],
                # [  59.64224478, -448.29890816,  911.68810588],
                # [ 273.18632979,   -1.66414539,  591.93608109],
                # [ 345.42863651,  365.92077557,  180.51780131],
                # [  -2.49959496, -527.89199888,   53.34811685],
            ]
        )

    motor_pos_samp = args["sample_data"]
    if np.size(motor_pos_samp) != 0:
        if np.size(motor_pos_samp) % 4 != 0:
            print("Please specify motor positions (angles) of sampling points.")
            print(
                "You specified %d numbers after your -s/--sample_data option, which is not a multiple of 4 number of numbers."
            )
            sys.exit(1)
        motor_pos_samp = motor_pos_samp.reshape((int(np.size(motor_pos_samp) / 4), 4))
    else:
        # You might want to manually replace this with your collected data
        # motor_pos_samp = np.array(
        #        [
        #            [  2185.65,-   1565.88,    901.40,- 3497.24, ], # First collected data. B and D had wrong signs
        #            [ 13289.12,-  16251.47,   9180.44,- -35.29, ],
        #            [ 19102.38,-  19737.29,  10106.74,- -34.05, ],
        #            [ 15736.09,-  12568.62,   7477.67,- 6882.47, ],
        #            [ 11833.54,-   3475.36,   7092.05,- 15260.72, ],
        #            [ 10453.83,-  -8991.93,   7252.55,- 24340.97, ],
        #            [ 12342.05,-   -286.95,   3259.66,- 16596.90, ],
        #            [ 12341.57,-   4064.08,   -840.31,- 8899.52, ],
        #            [ 14965.81,-   4064.08,  -6865.62,- 2128.97, ],
        #            [ 16145.44,-  -5181.89, -15089.08,- 2098.95, ],
        #            [ 11140.04,-  -5179.96,  -9597.86,- 7383.00, ],
        #            [  5789.15,- -11239.36,  -4038.70,- 13199.07, ],
        #            [ -7015.04,- -14092.20,   5979.81,- 10498.90, ],
        #            [-10983.31,- -10043.28,   6839.74,- 2878.31, ],
        #            [ -7130.69,- -10049.42,   6989.61,- 9051.27, ],
        #            [ -3162.74,-  -5991.19,   2751.78,- 7802.40, ],
        #            [  2492.55,-    -45.91,    361.06,- 6290.42, ],
        #        ]
        #        )
        motor_pos_samp = np.array(
            [
                #[    -0.0  ,     -0.0  ,     -0.0  ,     -0.0  ],
                [ -1559.522,  35112.937,  11052.954,  -8872.595], # Simulated data w no fuzz
                [ 15947.869,  44648.943,  25081.529, -20058.220],
                [  6380.425,  19670.677, -16066.842,   4584.479],
                [ 12033.385,  23821.841,  -5934.602, -15049.766],
                [ 25842.555,  34790.088,  13138.384, -28972.382],
                [ 26651.024,  14774.897, -26368.609,   8830.624],
                [ 30623.769,  19265.113, -11595.884,  -8872.595],
                [ 41211.318,  30931.858,   9866.130, -20058.220],
                [-22455.217,  16219.845,  15070.811,   4584.479],
                [ -9771.808,  20604.526,  19861.334, -15049.766],
                [ 10871.057,  32057.528,  32119.331, -28972.382],
                [ -9961.996,  31610.402,   5267.875,   8830.624],
                [  6498.393,   5934.960,   6498.393, -22456.334],
                [ 21632.763,  20205.165,  21632.763, -44870.348],
                [ 22498.630,  -7698.495,  -4589.645,   4584.479],
                [ 26731.305,   -604.054,   2676.888, -15049.766],
                [ 37871.408,  15355.987,  18863.009, -28972.382],
                [ -9961.996,   6924.497,  31248.118,   8830.624],
                [ -1559.522,  12086.068,  34967.311,  -8872.595],
                [ 15947.869,  25035.973,  44996.327, -20058.220],
                [  6380.425, -14551.956,  20602.771,   4584.479],
                [ 12033.385,  -6018.578,  24965.465, -15049.766],
                [ 25842.555,  11621.846,  36374.371, -28972.382],
                [ 26651.024, -30082.168,  17780.735,   8830.624],
                [ 30623.769, -15504.967,  22351.992,  -8872.595],
                [ 41211.318,   5888.695,  34181.575, -20058.220]
            ]
        )

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    if ux > u:
        print("Error: You have more xyz positions than samples!")
        print("You have %d xyz positions and %d samples" % (ux, u))
        sys.exit(1)

    st1 = timeit.default_timer()
    if args["method"] == "all":
        solution0 = solve(
            motor_pos_samp, xyz_of_samp, "L-BFGS-B", args["cx_is_positive"]
        )
        solution1 = solve(
            motor_pos_samp,
            xyz_of_samp,
            "PowellDirectionalSolver",
            args["cx_is_positive"],
        )
        solution2 = solve(motor_pos_samp, xyz_of_samp, "SLSQP", args["cx_is_positive"])
        solution3 = solve(motor_pos_samp, xyz_of_samp, "combo", args["cx_is_positive"])
        solution4 = solve(motor_pos_samp, xyz_of_samp, "differentialEvolutionSolver", args["cx_is_positive"])

        sol_anch0 = anchorsvec2matrix(solution0[0:params_anch])
        sol_pos0 = np.zeros((u, 3))
        sol_spool_buildup_factor0 = solution0[-params_buildup]
        sol_spool_r0 = solution0[-params_buildup + 1 :]
        if np.size(xyz_of_samp) != 0:
            sol_pos0 = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(solution0[params_anch:-params_buildup], (u - ux, 3)),
                )
            )
        else:
            sol_pos0 = np.reshape(solution0[params_anch:-params_buildup], (u, 3))
        the_cost0 = cost_for_pos_samp(
            anchorsvec2matrix(solution0[0:params_anch]),
            sol_pos0,
            motor_pos_samp,
            sol_spool_buildup_factor0,
            np.array([sol_spool_r0]),
        )
        sol_anch1 = anchorsvec2matrix(solution1[0:params_anch])
        sol_pos1 = np.zeros((u, 3))
        sol_spool_buildup_factor1 = solution1[-params_buildup]
        sol_spool_r1 = solution1[-params_buildup + 1 :]
        if np.size(xyz_of_samp) != 0:
            sol_pos1 = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(solution1[params_anch:-params_buildup], (u - ux, 3)),
                )
            )
        else:
            sol_pos1 = np.reshape(solution1[params_anch:-params_buildup], (u, 3))
        the_cost1 = cost_for_pos_samp(
            anchorsvec2matrix(solution1[0:params_anch]),
            sol_pos1,
            motor_pos_samp,
            sol_spool_buildup_factor1,
            np.array([sol_spool_r1]),
        )

        sol_anch2 = anchorsvec2matrix(solution2[0:params_anch])
        sol_pos2 = np.zeros((u, 3))
        sol_spool_buildup_factor2 = solution2[-params_buildup]
        sol_spool_r2 = solution2[-params_buildup + 1 :]
        if np.size(xyz_of_samp) != 0:
            sol_pos2 = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(solution2[params_anch:-params_buildup], (u - ux, 3)),
                )
            )
        else:
            sol_pos2 = np.reshape(solution2[params_anch:-params_buildup], (u, 3))
        the_cost2 = cost_for_pos_samp(
            anchorsvec2matrix(solution2[0:params_anch]),
            sol_pos2,
            motor_pos_samp,
            sol_spool_buildup_factor2,
            np.array([sol_spool_r2]),
        )

        sol_anch3 = anchorsvec2matrix(solution3[0:params_anch])
        sol_pos3 = np.zeros((u, 3))
        sol_spool_buildup_factor3 = solution3[-params_buildup]
        sol_spool_r3 = solution3[-params_buildup + 1 :]
        if np.size(xyz_of_samp) != 0:
            sol_pos3 = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(solution3[params_anch:-params_buildup], (u - ux, 3)),
                )
            )
        else:
            sol_pos3 = np.reshape(solution3[params_anch:-params_buildup], (u, 3))
        the_cost3 = cost_for_pos_samp(
            anchorsvec2matrix(solution3[0:params_anch]),
            sol_pos3,
            motor_pos_samp,
            sol_spool_buildup_factor3,
            np.array([sol_spool_r3]),
        )

        sol_anch4 = anchorsvec2matrix(solution3[0:params_anch])
        sol_pos4 = np.zeros((u, 3))
        sol_spool_buildup_factor4 = solution4[-params_buildup]
        sol_spool_r4 = solution4[-params_buildup + 1 :]
        if np.size(xyz_of_samp) != 0:
            sol_pos4 = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(solution4[params_anch:-params_buildup], (u - ux, 3)),
                )
            )
        else:
            sol_pos4 = np.reshape(solution4[params_anch:-params_buildup], (u, 3))
        the_cost4 = cost_for_pos_samp(
            anchorsvec2matrix(solution4[0:params_anch]),
            sol_pos4,
            motor_pos_samp,
            sol_spool_buildup_factor4,
            np.array([sol_spool_r4]),
        )

        if the_cost0 < the_cost1 and the_cost0 < the_cost2 and the_cost0 < the_cost3 and the_cost0 < the_cost4:
            solution = solution0
            print("Winner method was L-BFGS-B")
        if the_cost1 < the_cost0 and the_cost1 < the_cost2 and the_cost1 < the_cost3 and the_cost1 < the_cost4:
            solution = solution1
            print("Winner method was PowellDirectionalSolver")
        if the_cost2 < the_cost0 and the_cost2 < the_cost1 and the_cost2 < the_cost3 and the_cost2 < the_cost4:
            solution = solution2
            print("Winner method was SLSQP")
        if the_cost3 < the_cost0 and the_cost3 < the_cost1 and the_cost3 < the_cost2 and the_cost3 < the_cost4:
            print("Winner method was PowellDirectionalSolver+SLSQP combo")
            solution = solution3
        if the_cost4 < the_cost0 and the_cost4 < the_cost1 and the_cost4 < the_cost2 and the_cost4 < the_cost3:
            print("Winner method was differentialEvolutionSolver")
            solution = solution4
        print("Average solution: ")

        print((solution0 + solution1 + solution2 + solution3 + solution4) / 4)
    else:
        solution = solve(
            motor_pos_samp, xyz_of_samp, args["method"], args["cx_is_positive"]
        )
    st2 = timeit.default_timer()
    sol_anch = anchorsvec2matrix(solution[0:params_anch])
    sol_pos = np.zeros((u, 3))
    sol_spool_buildup_factor = solution[-params_buildup]
    sol_spool_r = solution[-params_buildup + 1 :]

    if np.size(xyz_of_samp) != 0:
        sol_pos = np.vstack(
            (
                xyz_of_samp,
                np.reshape(solution[params_anch:-params_buildup], (u - ux, 3)),
            )
        )
    else:
        sol_pos = np.reshape(solution[params_anch:-params_buildup], (u, 3))

    the_cost = cost_for_pos_samp(
        anchorsvec2matrix(solution[0:params_anch]),
        sol_pos,
        motor_pos_samp,
        sol_spool_buildup_factor,
        np.array([sol_spool_r]),
    )
    print("number of samples: %d" % u)
    print("input xyz coords:  %d" % (3 * ux))
    print("total cost:        %3.9f" % the_cost)
    print("cost per sample:   %3.9f" % (the_cost / u))

    if (u + 3 * ux) < params_anch:
        print("\nError: Lack of data detected.\n       Collect more samples.")
        if not args["debug"]:
            sys.exit(1)
        else:
            print("       Debug flag is set, so printing bogus anchor values anyways.")
    elif (u + 3 * ux) < params_anch + 4:
        print(
            "\nWarning: Data set might be too small.\n         The below values are unreliable unless input data is extremely accurate."
        )

    print_anch(sol_anch)
    print("Spool buildup factor:", sol_spool_buildup_factor)
    print("Spool radii:", sol_spool_r)
    if args["debug"]:
        print_anch_err(sol_anch, anchors)
        print("Method: %s" % args["method"])
        print("RUN TIME : {0}".format(st2 - st1))
        example_data_pos = np.array([[-1000., -1000.,  1000.],
                                     [-1000., -1000.,  2000.],
                                     [-1000.,     0.,     0.],
                                     [-1000.,     0.,  1000.],
                                     [-1000.,     0.,  2000.],
                                     [-1000.,  1000.,     0.],
                                     [-1000.,  1000.,  1000.],
                                     [-1000.,  1000.,  2000.],
                                     [    0., -1000.,     0.],
                                     [    0., -1000.,  1000.],
                                     [    0., -1000.,  2000.],
                                     [-1000., -1000.,     0.],
                                     [    0.,     0.,  1000.],
                                     [    0.,     0.,  2000.],
                                     [    0.,  1000.,     0.],
                                     [    0.,  1000.,  1000.],
                                     [    0.,  1000.,  2000.],
                                     [ 1000., -1000.,     0.],
                                     [ 1000., -1000.,  1000.],
                                     [ 1000., -1000.,  2000.],
                                     [ 1000.,     0.,     0.],
                                     [ 1000.,     0.,  1000.],
                                     [ 1000.,     0.,  2000.],
                                     [ 1000.,  1000.,     0.],
                                     [ 1000.,  1000.,  1000.],
                                     [ 1000.,  1000.,  2000.]])
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)  # No scientific notation
        print("pos err: ")
        print(sol_pos - example_data_pos)
        print(sol_pos)
        print(
            "spool_buildup_compensation err: %1.6f" % (sol_spool_buildup_factor - 0.008)
        )
        print("spool_r err:", sol_spool_r - np.array([65, 65, 65, 65]))
