#!/usr/bin/python3

"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division
import numpy as np
import scipy.optimize
import argparse
import timeit
import sys

import signal
import time

from hangprinter_forward_transform import forward_transform
from flex_distance import *

# Config values should be based on HP4 defaults
## Spool buildup
constant_spool_buildup_factor = 0.006875*2  # Qualified first guess for 1.1 mm line
spool_r_in_origin_first_guess = np.array([75.0, 75.0, 75.0, 75.0])
spool_gear_teeth = 255
motor_gear_teeth = 20
mechanical_advantage = np.array([2.0, 2.0, 2.0, 4.0])
lines_per_spool = np.array([1.0, 1.0, 1.0, 1.0])

## Line flex config
use_flex = True  # Toggle the use of flex compensation in the algorithm
abc_axis_min_force_limit = 10
abc_axis_max_force_limit = 50
springKPerUnitLength = 20000.0
mover_weight = 2.0

## Algorithm help and tuning
use_line_lengths_at_origin_data = False  # Toggle the enforcement of measured distances when at origin
line_lengths_when_at_origin = np.array([1597, 1795, 1582.5, 2355])
# line_lengths_when_at_origin = np.array([865, 870, 925, 1500])

l_long = 14000.0  # The longest distance from the origin that we should consider for anchor positions
l_short = 3000.0  # The longest distance from the origin that we should consider for data point collection
data_z_min = -100.0  # The lowest z-coordinate the algorithm should care about guessing
xyz_offset_max = (
    1.0  # Tell the algorithm to check if all xyz-data may carry an offset error compared to the encoder-data
)

# Force series 2. The enormous data set
xyz_of_samp = np.array([
[-0.527094, -0.634946, 0],#-0.370821],
[-266.144, -284.39,0],# 5.48368],
[240.691, -273.008,0],# 1.84387],
[283.932, 7.41853, 0],#-0.878299],
[304.608, 435.201,0],# 0.00422374],
[-177.608, 438.733,0],# -1.03731],
[-369.145, 45.972,0],# 3.83473],
[-198.326, 25.0843,0],# 1.23042],
[-465.56, -47.6696, 148.958],
[-632.978, 330.731, 123.941],
[-703.697, 410.585, 53.9513],
[-277.863, 522.619, 36.4518],
[443.706, 670.927, 121.135],
[465.545, 131.309, 197.025],
[38.9178, -623.777, 137.685],
[-343.296, -331.299, 175.893],
[-419.43, 25.5785, 243.119],
[-684.896, 395.692, 186.824],
[-287.429, 587.691, 297.107],
[476.717, 650.558, 205.9],
[307.146, 275.748, 231.131],
[43.8489, -415.35, 318.255],
[-28.077, -777.228, 241.797],
[-339.945, -219.036, 269.015],
[-642.961, 364.091, 321.117],
[-340.953, 389.898, 413.864],
[75.86, 545.978, 511.986],
[510.734, 641.667, 514.656],
[238.593, -33.943, 528.039],
[-8.68617, -660.259, 526.475],
[-307.971, -177.118, 588.672],
[-506.395, 298.312, 602.812],
[-59.6972, 35.2041, 611.094],
[28.0547, 457.294, 667.879],
[308.339, 471.581, 815.545],
[189.286, -72.3184, 806.111],
[-23.5655, -517.217, 886.333],
[-275.386, -30.794, 959.031],
[-62.4441, 278.419, 954.677],
[-29.8298, 147.913, 1155.45],
[62.8474, -55.7797, 1388.51],
])

motor_pos_samp = np.array([
[0.00, -0.07, 0.00, 0.00,  ],
[-4834.89, 7431.31, -1079.36, 1309.72,  ],
[-4836.05, 953.66, 6659.45, 970.36,  ],
[564.76, -3822.39, 4905.07, 512.16,  ],
[8813.47, -9838.24, 2856.02, 2158.20,  ],
[8634.09, -2206.69, -6170.42, 1894.75,  ],
[1763.93, 4885.85, -6499.04, 1229.59,  ],
[765.49, 2519.79, -3528.48, 379.65,  ],
[834.06, 7740.15, -6407.29, -3529.51,  ],
[8657.24, 6763.63, -12977.98, -181.22,  ],
[10432.20, 7228.67, -15158.59, 3699.54,  ],
[10533.93, -1127.67, -8259.67, 1458.11,  ],
[13933.80, -14082.43, 5272.85, 548.35,  ],
[4139.84, -7143.87, 7808.98, -5739.96,  ],
[-11416.06, 8940.40, 7919.32, -1647.06,  ],
[-4858.01, 9489.62, -1048.41, -4450.46,  ],
[2283.50, 6524.41, -6137.90, -7380.83,  ],
[10252.54, 7253.15, -13746.58, -1979.99,  ],
[12436.10, -506.54, -7377.42, -7433.21,  ],
[13875.65, -13881.30, 6170.34, -2469.32,  ],
[6343.52, -7037.12, 4239.84, -7534.02,  ],
[-6535.71, 6275.80, 6120.03, -10569.93,  ],
[-13526.80, 12395.02, 9438.00, -3628.34,  ],
[-2405.60, 8279.92, -1951.44, -8821.18,  ],
[9942.96, 7354.62, -11979.24, -6970.15,  ],
[9403.18, 2548.84, -6580.29, -13117.53,  ],
[12284.01, -4685.80, 629.34, -16498.56,  ],
[15060.61, -11561.99, 8359.57, -13338.70,  ],
[2024.87, -146.41, 6661.10, -19852.29,  ],
[-8945.93, 11597.50, 9529.92, -15583.44,  ],
[257.01, 8975.76, 130.90, -20750.08,  ],
[9559.57, 7198.64, -6649.38, -19098.09,  ],
[3646.72, 3292.98, 1921.11, -23278.22,  ],
[11565.72, -1796.95, 1442.24, -23271.50,  ],
[13311.67, -4060.44, 7501.33, -27567.81,  ],
[3651.60, 3241.77, 8378.30, -30394.25,  ],
[-2494.37, 12164.61, 10433.23, -30393.84,  ],
[6196.66, 9796.61, 3111.59, -35307.10,  ],
[10842.77, 3925.26, 3882.29, -35305.57,  ],
[10859.65, 6947.02, 7490.91, -43577.95,  ],
[10714.81, 10904.99, 13048.07, -52669.32,  ],
# hp-mark fails
[-10843.27, 14451.39, 13647.30, -15586.68, ],
[4036.97, 3243.27, 10617.52, -32192.57, ],
[2288.42, 9385.60, -7266.42, -7380.82,  ],
])

force_samp = np.array([
[13.72, 14.54, 14.07, 12.49,  ],
[7.70, 13.37, 12.66, 17.58,  ],
[16.25, 13.43, 12.66, 23.91,  ],
[9.95, 17.60, 12.45, 20.20,  ],
[17.16, 16.44, 12.03, 19.87,  ],
[19.31, 14.89, 13.24, 34.02,  ],
[14.58, 14.19, 9.47, 22.35,  ],
[12.99, 13.57, 13.73, 27.10,  ],
[13.10, 13.69, 13.00, 30.70,  ],
[13.02, 14.28, 10.87, 26.85,  ],
[10.72, 14.44, 17.84, 17.51,  ],
[16.99, 13.59, 14.02, 27.79,  ],
[12.60, 13.30, 14.17, 29.22,  ],
[6.83, 16.71, 17.09, 20.36,  ],
[14.68, 13.58, 12.31, 26.67,  ],
[20.22, 14.01, 12.22, 22.37,  ],
[13.02, 13.49, 14.77, 27.55,  ],
[15.44, 14.05, 10.63, 25.65,  ],
[12.85, 13.57, 15.53, 23.22,  ],
[13.12, 14.16, 12.44, 26.05,  ],
[13.07, 15.49, 8.60, 29.26,  ],
[14.35, 14.06, 12.83, 25.36,  ],
[13.51, 14.11, 11.71, 26.58,  ],
[15.39, 13.54, 16.00, 21.33,  ],
[17.05, 14.38, 18.41, 24.36,  ],
[11.03, 14.97, 16.24, 24.12,  ],
[15.09, 14.37, 19.41, 37.12,  ],
[11.23, 11.10, 16.99, 31.87,  ],
[10.50, 14.75, 13.67, 24.32,  ],
[8.33, 14.85, 14.19, 23.76,  ],
[12.28, 14.03, 15.06, 34.55,  ],
[10.51, 13.30, 15.43, 37.29,  ],
[18.20, 13.48, 15.45, 33.85,  ],
[12.94, 14.28, 16.53, 26.47,  ],
[14.46, 14.91, 14.97, 33.12,  ],
[14.14, 13.54, 12.08, 36.25,  ],
[20.27, 14.32, 9.42, 36.36,  ],
[12.86, 13.85, 14.39, 34.25,  ],
[13.59, 14.30, 13.52, 34.87,  ],
[7.83, 9.46, 10.95, 36.81,  ],
[13.16, 9.34, 15.68, 35.77,  ],
])


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


# Axes indexing
A = 0
B = 1
C = 2
D = 3
X = 0
Y = 1
Z = 2
params_anch = 12
params_buildup = 2
params_perturb = 3
A_bx = 3
A_cx = 6


def symmetric_anchors(l, az=-120.0, bz=-120.0, cz=-120.0):
    anchors = np.array(np.zeros((4, 3)))
    anchors[A, X] = 0
    anchors[A, Y] = -l
    anchors[A, Z] = az
    anchors[B, X] = l * np.cos(np.pi / 6)
    anchors[B, Y] = l * np.sin(np.pi / 6)
    anchors[B, Z] = bz
    anchors[C, X] = -l * np.cos(np.pi / 6)
    anchors[C, Y] = l * np.sin(np.pi / 6)
    anchors[C, Z] = cz
    anchors[D, X] = 0
    anchors[D, Y] = 0
    anchors[D, Z] = l
    return anchors


def positions(n, l, fuzz=0):
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


def distance_samples_relative_to_origin(anchors, pos):
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


def pos_to_motor_pos_samples(
    anchors,
    pos,
    spool_buildup_factor=constant_spool_buildup_factor,
    spool_r_in_origin=spool_r_in_origin_first_guess,
    spool_to_motor_gearing_factor=spool_gear_teeth / motor_gear_teeth,
    mech_adv_=mechanical_advantage,
    lines_per_spool_=lines_per_spool,
):
    """
    What motor positions (in degrees) motors would be at,
    given anchor and data collection positions.
    """

    # Assure np.array type
    spool_r_in_origin = np.array(spool_r_in_origin)
    mech_adv_ = np.array(mech_adv_)
    lines_per_spool_ = np.array(lines_per_spool_)

    spool_r_in_origin_sq = spool_r_in_origin * spool_r_in_origin

    # Buildup per line times lines. Minus sign because more line in air means less line on spool
    k2 = -1.0 * mech_adv_ * lines_per_spool_ * spool_buildup_factor

    # we now want to use degrees instead of steps as unit of rotation
    # so setting 360 where steps per motor rotation is in firmware buildup compensation algorithms
    degrees_per_unit_times_r = (spool_to_motor_gearing_factor * mech_adv_ * 360.0) / (2.0 * np.pi)
    k0 = 2.0 * degrees_per_unit_times_r / k2

    relative_line_lengths = distance_samples_relative_to_origin(anchors, pos)
    motor_positions = k0 * (np.sqrt(spool_r_in_origin_sq + relative_line_lengths * k2) - spool_r_in_origin)

    return motor_positions


def motor_pos_samples_to_distances_relative_to_origin(
    motor_samps,
    spool_buildup_factor=constant_spool_buildup_factor,
    spool_r=spool_r_in_origin_first_guess,
    spool_to_motor_gearing_factor=spool_gear_teeth / motor_gear_teeth,
    mech_adv_=mechanical_advantage,
    lines_per_spool_=lines_per_spool,
):
    # Buildup per line times lines. Minus sign because more line in air means less line on spool
    c1 = -mech_adv_ * lines_per_spool_ * spool_buildup_factor

    # we now want to use degrees instead of steps as unit of rotation
    # so setting 360 where steps per motor rotation is in firmware buildup compensation algorithms
    degrees_per_unit_times_r = (spool_to_motor_gearing_factor * mech_adv_ * 360.0) / (2.0 * np.pi)
    k0 = 2.0 * degrees_per_unit_times_r / c1

    return (((motor_samps / k0) + spool_r) ** 2.0 - spool_r * spool_r) / c1


def cost_from_forces(anchors, pos, force_samps, mover_weight):
    [ABC_forces_pre, D_forces_pre, ABC_forces_grav, D_forces_grav] = forces_gravity_and_pretension(
        abc_axis_max_force, np.max(np.array([abc_axis_max_force - 1, 0])), anch_to_pos, distances, mover_weight
    )

    synthetic_forces_pre = np.c_[ABC_forces_pre, D_forces_pre]
    synthetic_forces_grav = np.c_[ABC_forces_grav, D_forces_grav]

    # Remove gravity related forces from force_samp
    force_samps_pre = force_samps - synthetic_forces_grav

    # Normalize. we don't care about pretension force sizes
    synthetic_forces_pre = synthetic_forces_pre / np.linalg.norm(synthetic_forces_pre, 2, 1)[:, np.newaxis]
    force_samps_pre = force_samps_pre / np.linalg.norm(force_samps_pre, 2, 1)[:, np.newaxis]

    return np.sum(pow(synthetic_force_pre - force_samp_pre, 2))


def cost_sq_for_pos_samp(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, abc_axis_max_force=1
):
    """
    Sum of squares

    Creates samples based on guessed anchor and data collection positions.
    May take line flex into account when generating the samples.

    For all samples sum
    (Sample value if anchor position A and cartesian position x were guessed - actual sample)^2
    """

    err = 0
    if use_flex:
        err = np.sum(
            pow(
                distance_samples_relative_to_origin(anchors, pos)
                - (
                    motor_pos_samples_to_distances_relative_to_origin(motor_pos_samp, spool_buildup_factor, spool_r)
                    - flex_distance(
                        abc_axis_max_force,
                        np.max(np.array([abc_axis_max_force - 1, 0.0001])),
                        anchors,
                        pos,
                        mechanical_advantage,
                        springKPerUnitLength,
                        mover_weight,
                    )
                ),
                2,
            )
        )
    else:
        err = np.sum(
            pow(
                distance_samples_relative_to_origin(anchors, pos)
                - motor_pos_samples_to_distances_relative_to_origin(motor_pos_samp, spool_buildup_factor, spool_r),
                2,
            )
        )
    if use_line_lengths_at_origin_data:
        line_lengths_when_at_origin_err = np.linalg.norm(anchors, 2, 1) - line_lengths_when_at_origin
        err += line_lengths_when_at_origin_err.dot(line_lengths_when_at_origin_err)

    return err


def cost_sq_for_pos_samp_forward_transform(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, abc_axis_max_force=1
):
    line_lengths_when_at_origin_err = np.linalg.norm(anchors, 2, 1) - line_lengths_when_at_origin
    line_length_samp = np.zeros((np.size(motor_pos_samp, 0), 3))
    if use_flex:
        line_length_samp = motor_pos_samples_to_distances_relative_to_origin(
            motor_pos_samp, spool_buildup_factor, spool_r
        ) - flex_distance(
            abc_axis_max_force,
            np.max(np.array([abc_axis_max_force - 1, 0.0001])),
            anchors,
            pos,
            mechanical_advantage,
            springKPerUnitLength,
            mover_weight,
        )
    else:
        line_length_samp = motor_pos_samples_to_distances_relative_to_origin(
            motor_pos_samp, spool_buildup_factor, spool_r
        )

    tot_err = 0
    for i in range(np.size(line_length_samp, 0)):
        diff = pos[i] - forward_transform(anchors, line_length_samp[i])
        tot_err += diff.dot(diff)

    if use_line_lengths_at_origin_data:
        return tot_err + line_lengths_when_at_origin_err.dot(line_lengths_when_at_origin_err)

    return tot_err


def cost_sq_for_pos_samp_combined(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, abc_axis_max_force=1
):
    return 10 * cost_sq_for_pos_samp_forward_transform(
        anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, abc_axis_max_force
    ) + cost_sq_for_pos_samp(
        anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, abc_axis_max_force
    )


def anchorsvec2matrix(anchorsvec):
    """Create a 4x3 anchors matrix from anchors vector."""
    anchors = np.array(
        [
            [anchorsvec[0], anchorsvec[1], anchorsvec[2]],
            [anchorsvec[3], anchorsvec[4], anchorsvec[5]],
            [anchorsvec[6], anchorsvec[7], anchorsvec[8]],
            [anchorsvec[9], anchorsvec[10], anchorsvec[11]],
        ]
    )

    return anchors


def anchorsmatrix2vec(a):
    return [a[A, X], a[A, Y], a[A, Z], a[B, X], a[B, Y], a[B, Z], a[C, X], a[C, Y], a[C, Z], a[D, X], a[D, Y], a[D, Z]]


def posvec2matrix(v, u):
    return np.reshape(v, (u, 3))


def posmatrix2vec(m):
    return np.reshape(m, np.shape(m)[0] * 3)


def pre_list(l, num):
    return np.append(np.append(l[0:params_anch], l[params_anch : params_anch + 3 * num]), l[-params_buildup:])


def solve(motor_pos_samp, xyz_of_samp, line_lengths_when_at_origin, method, debug=False):
    """Find reasonable positions and anchors given a set of samples."""

    print(method)
    if use_flex:
        print("Using flex compensation")
    else:
        print("Assuming zero flex")

    if use_line_lengths_at_origin_data:
        print("Using hand measured line lengths at the origin")
    else:
        print("Not using hand measured line lengths")

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    number_of_params_pos = 3 * (u - ux)

    def costx(
        _cost,
        posvec,
        anchvec,
        spool_buildup_factor,
        spool_r,
        u,
        line_lengths_when_at_origin,
        perturb,
        abc_axis_max_force=1.0,
    ):
        """Identical to cost, except the shape of inputs and capture of samp, xyz_of_samp, ux, and u

        Parameters
        ----------
        x : [A_ay A_az A_bx A_by A_bz A_cx A_cy A_cz A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """

        if len(posvec) > 0:
            posvec = np.array([pos for pos in posvec])
        anchvec = np.array([anch for anch in anchvec])
        spool_r = np.array([r for r in spool_r])
        spool_r = np.r_[spool_r[0], spool_r[0], spool_r]
        perturb = np.array([p for p in perturb])

        anchors = anchorsvec2matrix(anchvec)
        # Adds in known positions back in before calculating the cost
        pos = np.zeros((u, 3))
        if np.size(xyz_of_samp) != 0:
            pos[0:ux] = xyz_of_samp
        if u > ux:
            pos[ux:] = np.reshape(posvec, (u - ux, 3))

        return _cost(
            anchors,
            pos + perturb,
            motor_pos_samp[:u],
            spool_buildup_factor,
            spool_r,
            line_lengths_when_at_origin,
            abc_axis_max_force,
        )

    # Limits of anchor positions:
    lb = np.array(
        [
            -l_long,  # A_ax > x
            -l_long,  # A_ay > x
            -1300.0,  # A_az > x
            0.0,  # A_bx > x
            0.0,  # A_by > x
            -1300.0,  # A_bz > x
            -l_long,  # A_cx > x
            0.0,  # A_cy > x
            -1300.0,  # A_cz > x
            -500.0,  # A_dx > x
            -500.0,  # A_dy > x
            1000.0,  # A_dz > x
        ]
        + [-l_short, -l_short, data_z_min] * (u - ux)
        + [
            spool_r_in_origin_first_guess[0] - 1.0,
            spool_r_in_origin_first_guess[3] - 1.0,
        ]
        + [-xyz_offset_max, -xyz_offset_max, -xyz_offset_max]
    )
    if use_flex:
        lb = np.append(lb, abc_axis_min_force_limit)

    ub = np.array(
        [
            l_long,  # A_ax < x
            0.0,  # A_ay < x
            200.0,  # A_az < x
            l_long,  # A_bx < x
            l_long,  # A_by < x
            200.0,  # A_bz < x
            0.0,  # A_cx < x
            l_long,  # A_cy < x
            200.0,  # A_cz < x
            500.0,  # A_dx < x
            500.0,  # A_dy < x
            l_long,  # A_dz < x
        ]
        + [l_short, l_short, 2.0 * l_short] * (u - ux)
        + [
            spool_r_in_origin_first_guess[0] + 5.0,
            spool_r_in_origin_first_guess[3] + 5.0,
        ]
        + [xyz_offset_max, xyz_offset_max, xyz_offset_max]
    )
    if use_flex:
        ub = np.append(ub, abc_axis_max_force_limit)

    pos_est = np.zeros((u - ux, 3))  # The positions we need to estimate
    anchors_est = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x_guess = (
        list(anchorsmatrix2vec(anchors_est))[0:params_anch]
        + list(posmatrix2vec(pos_est))
        + list([spool_r_in_origin_first_guess[0], spool_r_in_origin_first_guess[3]])
        + [0, 0, 0]
    )
    if use_flex:
        x_guess += [10.0]

    disp = False
    if debug:
        disp = True

    if method == "differentialEvolutionSolver":
        print("Hit Ctrl+C to stop solver. Then type exit to get the current solution.")
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.monitors import VerboseMonitor, Monitor
        from mystic.termination import VTR, ChangeOverGeneration, And, Or
        from mystic.strategy import Best1Exp, Best1Bin

        stop = Or(VTR(1e-12), ChangeOverGeneration(0.0000001, 5000))
        ndim = number_of_params_pos + params_anch + params_buildup + params_perturb + use_flex
        npop = 3
        stepmon = VerboseMonitor(100)
        if not disp:
            stepmon = Monitor()
        solver = DifferentialEvolutionSolver2(ndim, npop)

        solver.SetRandomInitialPoints(lb, ub)
        solver.SetStrictRanges(lb, ub)
        solver.SetGenerationMonitor(stepmon)
        solver.enable_signal_handler()  # Handle Ctrl+C gracefully. Be restartable
        if use_flex:
            solver.Solve(
                lambda x: costx(
                    cost_sq_for_pos_samp,
                    x[params_anch : -(params_buildup + params_perturb + 1)],
                    x[0:params_anch],
                    constant_spool_buildup_factor,
                    x[-(params_buildup + params_perturb + 1) : -(params_perturb + 1)],
                    u,
                    line_lengths_when_at_origin,
                    x[-(params_perturb + 1) : -1],
                    x[-1],
                ),
                termination=stop,
                strategy=Best1Bin,
            )
        else:
            solver.Solve(
                lambda x: costx(
                    cost_sq_for_pos_samp,
                    x[params_anch : -(params_buildup + params_perturb)],
                    x[0:params_anch],
                    constant_spool_buildup_factor,
                    x[-(params_buildup + params_perturb) : -params_perturb],
                    u,
                    line_lengths_when_at_origin,
                    x[-params_perturb:],
                ),
                termination=stop,
                strategy=Best1Bin,
            )

        # use monitor to retrieve results information
        iterations = len(stepmon)
        cost = stepmon.y[-1]
        if disp:
            print("Generation %d has best Chi-Squared: %f" % (iterations, cost))
        best_x = solver.Solution()
        if not type(best_x[0]) == float:
            best_x = np.array([float(pos) for pos in best_x])
        return best_x

    elif method == "PowellDirectionalSolver":
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG
        from mystic.monitors import VerboseMonitor, Monitor
        from mystic.termination import VTR, And, Or

        ndim = number_of_params_pos + params_anch + params_buildup + params_perturb + use_flex
        killer = GracefulKiller()
        best_cost = 9999999999999.9
        i = 0
        print("Hit Ctrl+C and wait a bit to stop solver and get current best solution.")
        while True:
            i = i + 1
            if disp:
                print("Try: %d/5" % i)
            if killer.kill_now or i == 5:
                break
            solver = PowellDirectionalSolver(ndim)
            solver.SetRandomInitialPoints(lb, ub)
            solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
            solver.SetTermination(Or(VTR(1e-25), COG(1e-10, 10)))
            solver.SetStrictRanges(lb, ub)
            solver.SetGenerationMonitor(VerboseMonitor(5))
            if not disp:
                solver.SetGenerationMonitor(Monitor())
            if use_flex:
                solver.Solve(
                    lambda x: costx(
                        cost_sq_for_pos_samp,
                        x[params_anch : -(params_buildup + params_perturb + 1)],
                        x[0:params_anch],
                        constant_spool_buildup_factor,
                        x[-(params_buildup + params_perturb + 1) : -(params_perturb + 1)],
                        u,
                        line_lengths_when_at_origin,
                        x[-(params_perturb + 1) : -1],
                        x[-1],
                    )
                )
            else:
                solver.Solve(
                    lambda x: costx(
                        cost_sq_for_pos_samp,
                        x[params_anch : -(params_buildup + params_perturb)],
                        x[0:params_anch],
                        constant_spool_buildup_factor,
                        x[-(params_buildup + params_perturb) : -(params_perturb)],
                        u,
                        line_lengths_when_at_origin,
                        x[-params_perturb:],
                    )
                )
            if solver.bestEnergy < best_cost:
                if disp:
                    print("New best x: ")
                    print("With cost: ", solver.bestEnergy)
                best_cost = solver.bestEnergy
                best_x = np.array(solver.bestSolution)
            if solver.bestEnergy < 0.0001:
                if disp:
                    print("Found a perfect solution!")
                break

        if not type(best_x[0]) == float:
            best_x = np.array([float(pos) for pos in best_x])
        return best_x

    elif method == "SLSQP":
        # Create a random guess
        best_cost = 999999.9
        best_x = x_guess
        killer = GracefulKiller()
        print("Hit Ctrl+C and wait a bit to stop solver and get current best solution.")
        tries = 8
        for i in range(tries):
            if disp:
                print("Try: %d/%d" % (i + 1, tries))
            if killer.kill_now:
                break
            random_guess = np.array([b[0] + (b[1] - b[0]) * np.random.rand() for b in list(zip(lb, ub))])
            if use_flex:
                sol = scipy.optimize.minimize(
                    lambda x: costx(
                        cost_sq_for_pos_samp,
                        # cost_sq_for_pos_samp_forward_transform,
                        # cost_sq_for_pos_samp_combined,
                        x[params_anch : -(params_buildup + params_perturb + 1)],
                        x[0:params_anch],
                        constant_spool_buildup_factor,
                        x[-(params_buildup + params_perturb + 1) : -(params_perturb + 1)],
                        u,
                        line_lengths_when_at_origin,
                        x[-(params_perturb + 1) : -1],
                        x[-1],
                    ),
                    random_guess,
                    method="SLSQP",
                    bounds=list(zip(lb, ub)),
                    options={"disp": disp, "ftol": 1e-9, "maxiter": 500},
                )
            else:
                sol = scipy.optimize.minimize(
                    lambda x: costx(
                        cost_sq_for_pos_samp,
                        # cost_sq_for_pos_samp_forward_transform,
                        # cost_sq_for_pos_samp_combined,
                        x[params_anch : -(params_buildup + params_perturb)],
                        x[0:params_anch],
                        constant_spool_buildup_factor,
                        x[-(params_buildup + params_perturb) : -(params_perturb)],
                        u,
                        line_lengths_when_at_origin,
                        x[-(params_perturb):],
                    ),
                    random_guess,
                    method="SLSQP",
                    bounds=list(zip(lb, ub)),
                    options={"disp": disp, "ftol": 1e-9, "maxiter": 500},
                )
            if sol.fun < best_cost:
                if disp:
                    print("New best x: ")
                    print("With cost: ", sol.fun)
                best_cost = sol.fun
                best_x = sol.x

        if not type(best_x[0]) == float:
            best_x = np.array([float(pos) for pos in best_x])
        return np.array(best_x)

    elif method == "L-BFGS-B":
        print("You can not interrupt this solver without losing the solution.")
        if use_flex:
            best_x = scipy.optimize.minimize(
                lambda x: costx(
                    cost_sq_for_pos_samp,
                    x[params_anch : -(params_buildup + params_perturb + 1)],
                    x[0:params_anch],
                    constant_spool_buildup_factor,
                    x[-(params_buildup + params_perturb + 1) : -(params_perturb + 1)],
                    u,
                    line_lengths_when_at_origin,
                    x[-(params_perturb + 1) : -1],
                    x[-1],
                ),
                x_guess,
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={"disp": disp, "ftol": 1e-9, "gtol": 1e-12, "maxiter": 50000, "maxfun": 1000000},
            ).x
        else:
            best_x = scipy.optimize.minimize(
                lambda x: costx(
                    cost_sq_for_pos_samp,
                    x[params_anch : -(params_buildup + params_perturb)],
                    x[0:params_anch],
                    constant_spool_buildup_factor,
                    x[-(params_buildup + params_perturb) : -(params_perturb)],
                    u,
                    line_lengths_when_at_origin,
                    x[-(params_perturb):],
                ),
                x_guess,
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={"disp": disp, "ftol": 1e-12, "gtol": 1e-12, "maxiter": 50000, "maxfun": 1000000},
            ).x
        if not type(best_x[0]) == float:
            best_x = np.array([float(pos) for pos in best_x])
        return best_x

    else:
        print("Method %s is not supported!" % method)
        sys.exit(1)


def print_copypasteable(anch, spool_buildup_factor, spool_r):
    print(
            "\nM669 A%.2f:%.2f:%.2f B%.2f:%.2f:%.2f C%.2f:%.2f:%.2f D%.2f:%.2f:%.2f\nM666 R%.3f:%.3f:%.3f:%.3f\n;Here follows constants that are set in the script\nM666 Q%.6f W%.2f S%.2f U%d:%d:%d:%d O%d:%d:%d:%d L%d:%d:%d:%d H%d:%d:%d:%d"
        % (
            anch[A, X],
            anch[A, Y],
            anch[A, Z],
            anch[B, X],
            anch[B, Y],
            anch[B, Z],
            anch[C, X],
            anch[C, Y],
            anch[C, Z],
            anch[D, X],
            anch[D, Y],
            anch[D, Z],
            spool_r[A],
            spool_r[B],
            spool_r[C],
            spool_r[D],
            spool_buildup_factor,
            mover_weight,
            springKPerUnitLength,
            mechanical_advantage[A],
            mechanical_advantage[B],
            mechanical_advantage[C],
            mechanical_advantage[D],
            lines_per_spool[A],
            lines_per_spool[B],
            lines_per_spool[C],
            lines_per_spool[D],
            motor_gear_teeth,
            motor_gear_teeth,
            motor_gear_teeth,
            motor_gear_teeth,
            spool_gear_teeth,
            spool_gear_teeth,
            spool_gear_teeth,
            spool_gear_teeth,
        )
    )


def print_anch_err(sol_anch, anchors):
    print("\nErr_A_X: %9.3f" % (float(sol_anch[A, X]) - (anchors[A, X])))
    print("Err_A_Y: %9.3f" % (float(sol_anch[A, Y]) - (anchors[A, Y])))
    print("Err_A_Z: %9.3f" % (float(sol_anch[A, Z]) - (anchors[A, Z])))
    print("Err_B_X: %9.3f" % (float(sol_anch[B, X]) - (anchors[B, X])))
    print("Err_B_Y: %9.3f" % (float(sol_anch[B, Y]) - (anchors[B, Y])))
    print("Err_B_Z: %9.3f" % (float(sol_anch[B, Z]) - (anchors[B, Z])))
    print("Err_C_X: %9.3f" % (float(sol_anch[C, X]) - (anchors[C, X])))
    print("Err_C_Y: %9.3f" % (float(sol_anch[C, Y]) - (anchors[C, Y])))
    print("Err_C_Z: %9.3f" % (float(sol_anch[C, Z]) - (anchors[C, Z])))
    print("Err_D_X: %9.3f" % (float(sol_anch[D, X]) - (anchors[D, X])))
    print("Err_D_Y: %9.3f" % (float(sol_anch[D, Y]) - (anchors[D, Y])))
    print("Err_D_Z: %9.3f" % (float(sol_anch[D, Z]) - (anchors[D, Z])))


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(Store_as_array, self).__call__(parser, namespace, values, option_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find best Hangprinter config based on true line lengths, line difference samples, and xyz positions if known."
    )
    parser.add_argument("-d", "--debug", help="Print debug information", action="store_true")
    parser.add_argument(
        "-m",
        "--method",
        help="Available methods are SLSQP (0), PowellDirectionalSolver (1), L-BFGS-B (2), differentialEvolutionSolver (3), and all (4). Try 0 first, then 1, and so on.",
        default="SLSQP",
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
    parser.add_argument(
        "-l",
        "--line_lengths",
        help="Specify the four line lenghts (A, B, C, and D) hand measured when nozzle is at the origin.",
        action=Store_as_array,
        type=float,
        nargs="+",
        default=np.array([]),
    )
    args = vars(parser.parse_args())

    if args["method"] == "0":
        args["method"] = "SLSQP"
    if args["method"] == "1":
        args["method"] = "PowellDirectionalSolver"
    if args["method"] == "2":
        args["method"] = "L-BFGS-B"
    if args["method"] == "3":
        args["method"] = "differentialEvolutionSolver"
    if args["method"] == "4":
        args["method"] = "all"
        print(args["method"])

    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    anchors = np.array(
        [
            [0, -1620, -150],
            [1800 * np.cos(np.pi / 4), 1800 * np.sin(np.pi / 4), -150],
            [-1620 * np.cos(np.pi / 6), 1620 * np.sin(np.pi / 6), -150],
            [0, 0, 2350],
        ]
    )

    # Possibly overwrite hard-coded data with command line-provided data
    xyz_of_samp_ = args["xyz_of_samp"]
    if np.size(xyz_of_samp_) != 0:
        if np.size(xyz_of_samp_) % 3 != 0:
            print(
                "Error: You specified %d numbers after your -x/--xyz_of_samp option, which is not a multiple of 3 numbers."
            )
            sys.exit(1)
        xyz_of_samp = xyz_of_samp_.reshape((int(np.size(xyz_of_samp_) / 3), 3))

    motor_pos_samp_ = args["sample_data"]
    if np.size(motor_pos_samp_) != 0:
        if np.size(motor_pos_samp_) % 4 != 0:
            print("Please specify motor positions (angles) of sampling points.")
            print(
                "You specified %d numbers after your -s/--sample_data option, which is not a multiple of 4 number of numbers."
                % (np.size(motor_pos_samp_))
            )
            sys.exit(1)
        motor_pos_samp = motor_pos_samp_.reshape((int(np.size(motor_pos_samp_) / 4), 4))
    line_lengths_when_at_origin_ = args["line_lengths"]
    if np.size(line_lengths_when_at_origin_) != 0:
        if np.size(line_lengths_when_at_origin_) != 4:
            print("Please specify four measured line lengths.")
            print(
                "You specified %d numbers after your -l/--line_lengths option."
                % (np.size(line_lengths_when_at_origin_))
            )
            sys.exit(1)
        line_lengths_when_at_origin = line_lengths_when_at_origin_

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]

    motor_pos_samp = np.array([r for r in motor_pos_samp.reshape(u * 4)]).reshape((u, 4))
    xyz_of_samp = np.array([r for r in xyz_of_samp.reshape(ux * 3)]).reshape((ux, 3))

    if ux > u:
        print("Error: You have more xyz positions than samples!")
        print("You have %d xyz positions and %d samples" % (ux, u))
        sys.exit(1)

    def computeCost(solution):
        anch = np.zeros((4, 3))
        anch = anchorsvec2matrix(solution[0:params_anch])
        spool_buildup_factor = constant_spool_buildup_factor
        spool_r = np.array(
            [x for x in solution[-(params_buildup + params_perturb + use_flex) : -(params_perturb + use_flex)]]
        )
        spool_r = np.r_[spool_r[0], spool_r[0], spool_r]

        line_max_force = solution[-use_flex]  # This actually works
        pos = np.zeros((u, 3))
        if np.size(xyz_of_samp) != 0:
            pos = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(solution[params_anch : -(params_buildup + params_perturb + use_flex)], (u - ux, 3)),
                )
            )
        else:
            pos = np.reshape([x for x in solution[params_anch : -(params_buildup + params_perturb + use_flex)]], (u, 3))
        if use_flex:
            # return cost_sq_for_pos_samp_forward_transform(
            return cost_sq_for_pos_samp(
                anch,
                pos + solution[-(params_perturb + 1) : -1],
                motor_pos_samp,
                constant_spool_buildup_factor,
                spool_r,
                line_lengths_when_at_origin,
                line_max_force,
            )
        else:
            # return cost_sq_for_pos_samp_forward_transform(
            return cost_sq_for_pos_samp(
                anch,
                pos + solution[-params_perturb:],
                motor_pos_samp,
                constant_spool_buildup_factor,
                spool_r,
                line_lengths_when_at_origin,
                line_max_force,
            )

    ndim = 3 * (u - ux) + params_anch + params_buildup + params_perturb + use_flex

    class candidate:
        name = "no_name"
        solution = np.zeros(ndim)
        anch = np.zeros((4, 3))
        spool_buildup_factor = constant_spool_buildup_factor
        cost = 9999.9
        pos = np.zeros(3 * (u - ux))
        xyz_offset = np.zeros(3)
        line_max_force = 0.0

        def __init__(self, name, solution):
            self.name = name
            self.solution = solution
            if np.array(solution).any():
                self.cost = computeCost(solution)
                np.set_printoptions(suppress=False)
                if args["debug"]:
                    print("%s has cost %e" % (self.name, self.cost))
                np.set_printoptions(suppress=True)
                self.anch = anchorsvec2matrix(self.solution[0:params_anch])
                self.spool_buildup_factor = constant_spool_buildup_factor  # self.solution[-params_buildup]
                self.spool_r = self.solution[
                    -(params_buildup + params_perturb + use_flex) : -(params_perturb + use_flex)
                ]
                self.spool_r = np.r_[self.spool_r[0], self.spool_r[0], self.spool_r]
                if np.size(xyz_of_samp) != 0:
                    self.pos = np.vstack(
                        (
                            xyz_of_samp,
                            np.reshape(
                                solution[params_anch : -(params_buildup + params_perturb + use_flex)], (u - ux, 3)
                            ),
                        )
                    )
                else:
                    self.pos = np.reshape(solution[params_anch : -(params_buildup + params_perturb + use_flex)], (u, 3))
                if use_flex:
                    self.xyz_offset = solution[-(params_perturb + 1) : -1]
                    self.line_max_force = solution[-1]
                else:
                    self.xyz_offset = solution[-params_perturb:]

    the_cand = candidate("no_name", np.zeros(ndim))
    st1 = timeit.default_timer()
    if args["method"] == "all":
        cands = [
            candidate(
                cand_name, solve(motor_pos_samp, xyz_of_samp, line_lengths_when_at_origin, cand_name, args["debug"])
            )
            for cand_name in ["SLSQP", "PowellDirectionalSolver", "L-BFGS-B", "differentialEvolutionSolver"]
        ]
        cands[:] = sorted(cands, key=lambda cand: cand.cost)
        print("Winner method: is %s" % cands[0].name)
        the_cand = cands[0]
    else:
        the_cand = candidate(
            args["method"],
            solve(motor_pos_samp, xyz_of_samp, line_lengths_when_at_origin, args["method"], args["debug"]),
        )

    st2 = timeit.default_timer()

    print("number of samples: %d" % u)
    print("input xyz coords:  %d" % (3 * ux))
    np.set_printoptions(suppress=False)
    print("total cost:        %e" % the_cand.cost)
    print("cost per sample:   %e" % (the_cand.cost / u))
    np.set_printoptions(suppress=True)

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

    print_copypasteable(the_cand.anch, the_cand.spool_buildup_factor, the_cand.spool_r)

    if args["debug"]:
        print("Anchors:")
        print(
            "A=[%.2f, %.2f, %.2f]\nB=[%.2f, %.2f, %.2f]\nC=[%.2f, %.2f, %.2f]\nD=[%.2f, %.2f, %.2f]"
            % (
                the_cand.anch[A, X],
                the_cand.anch[A, Y],
                the_cand.anch[A, Z],
                the_cand.anch[B, X],
                the_cand.anch[B, Y],
                the_cand.anch[B, Z],
                the_cand.anch[C, X],
                the_cand.anch[C, Y],
                the_cand.anch[C, Z],
                the_cand.anch[D, X],
                the_cand.anch[D, Y],
                the_cand.anch[D, Z],
            )
        )
        print("Spool buildup factor:", the_cand.spool_buildup_factor)  # err
        print("Spool radii:", the_cand.spool_r)
        print("XYZ offset: ", the_cand.xyz_offset)
        if use_flex:
            print("line_max_force: %s" % the_cand.line_max_force)
        # print_anch_err(the_cand.anch, anchors)
        # print("Method: %s" % args["method"])
        print("RUN TIME : {0}".format(st2 - st1))
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)  # No scientific notation
        print("Data collected at positions: ")
        print(the_cand.pos)
        L_errs = np.linalg.norm(the_cand.anch, 2, 1) - line_lengths_when_at_origin
        print("Line length errors:")
        print("line_length_error_a=%.2f" % (L_errs[0]))
        print("line_length_error_b=%.2f" % (L_errs[1]))
        print("line_length_error_c=%.2f" % (L_errs[2]))
        print("line_length_error_d=%.2f" % (L_errs[3]))
        print("Tot line length err=%.2f" % (np.linalg.norm(L_errs)))
        # example_data_pos = np.array(
        #    [
        #        [-1000.0, -1000.0, 1000.0],
        #        [-1000.0, -1000.0, 2000.0],
        #        [-1000.0, 0.0, 0.0],
        #        [-1000.0, 0.0, 1000.0],
        #        [-1000.0, 0.0, 2000.0],
        #        [-1000.0, 1000.0, 0.0],
        #        [-1000.0, 1000.0, 1000.0],
        #        [-1000.0, 1000.0, 2000.0],
        #        [0.0, -1000.0, 0.0],
        #        [0.0, -1000.0, 1000.0],
        #        [0.0, -1000.0, 2000.0],
        #        [-1000.0, -1000.0, 0.0],
        #        [0.0, 0.0, 1000.0],
        #        [0.0, 0.0, 2000.0],
        #        [0.0, 1000.0, 0.0],
        #        [0.0, 1000.0, 1000.0],
        #        [0.0, 1000.0, 2000.0],
        #        [1000.0, -1000.0, 0.0],
        #        [1000.0, -1000.0, 1000.0],
        #        [1000.0, -1000.0, 2000.0],
        #        [1000.0, 0.0, 0.0],
        #        [1000.0, 0.0, 1000.0],
        #        [1000.0, 0.0, 2000.0],
        #        [1000.0, 1000.0, 0.0],
        #        [1000.0, 1000.0, 1000.0],
        #        [1000.0, 1000.0, 2000.0],
        #    ]
        # )
        # print("pos err: ")
        # print(the_cand.pos - example_data_pos)
        # print(
        #    "spool_buildup_compensation err: %1.6f"
        #    % (the_cand.spool_buildup_factor - 0.008)
        # )
        # print("spool_r err:", the_cand.spool_r - np.array([65, 65, 65, 65]))
