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
constant_spool_buildup_factor = 0.08  # Qualified first guess for 1.1 mm line
spool_r_in_origin_first_guess = np.array([75.0, 75.0, 75.0, 75.0])
spool_gear_teeth = 255
motor_gear_teeth = 20
mechanical_advantage = np.array([2.0, 2.0, 2.0, 4.0])
lines_per_spool = np.array([1.0, 1.0, 1.0, 1.0])

## Line flex config
use_flex = True  # Toggle the use of flex compensation in the algorithm
abc_axis_min_force_limit = 8
abc_axis_max_force_limit = 100
springKPerUnitLength = 20000.0
mover_weight = 1.0

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

xyz_of_samp = np.array(
    [
        ##[-0.147263, -0.892487, 2.56335],
        ##[0.00, 0.00, 0.00],
        # [223.856, 125.465, 20.5782],
        # [128.653, -133.586, 9.02497],
        # [-2.84022, -283.596, 23.9668],
        # 		#[-181.868, -162.114, 4.75218], #2.4
        # [-271.716, 68.3684, 24.108],
        # 		#[-42.3365, 204.751, 15.2165], #1.6 / 1.2
        # [55.7334, 72.3623, 3.08083], # 0.4
        # [134.097, 59.1235, 108.452],
        # [6.28163, -53.5486, 127.514],
        # [-101.178, -198.513, 98.2964],
        # [-102.854, 59.0736, 156.536],
        # [129.568, 163.237, 179.094],
        # [145.472, 23.3536, 33.189],
        # [59.7281, -86.3042, 115.33],
        # [15.2818, 12.9319, 130.221],
        [-204.986, 481.404, 61.3799],
        [286.227, 478.716, 63.8561],
        [120.141, -71.862, 587.729],
        [-217.577, 293.977, 626.494],
        [-479.619, 188.839, 756.471],
        [233.685, 351.62, 469.438],
        [67.3321, 325.857, 239.275],
        [-198.748, 63.0038, 1.22],
        ## Using bed probe for Z-measurements
        # [198.64, -198.88, 0],
        # [-199.06, -199.93, -0.579], #0],
        # [-199.85, 200.87, -0.521], #0],
        # [200.67, 200.46, -0.294], #0],
    ]
)

motor_pos_samp = np.array(
    [
        ##[0.00, 0.00, 0.00, 0.00],
        # [4645.53, -6722.25, 4217.10, -26.06, ],
        # [-3023.68, -397.72, 5039.10, 114.94, ],
        # [-7526.98, 5155.73, 4937.55, 217.24, ],
        ##[-4191.99, 6644.66, -1028.40, 617.14, ],
        # [2704.71, 5926.61, -7001.70, 125.10, ],
        ##[5777.71, -1370.89, -3431.55, 9.34, ], # 1.6
        # [2184.96, -2339.02, 386.40, 21.04, ],
        # [2876.09, -2879.89, 3148.95, -2853.26, ],
        # [-419.65, 1735.16, 1787.55, -3487.35, ],
        # [-4554.13, 6103.16, 1817.66, -2121.15, ],
        # [2846.12, 2957.51, -1992.19, -4203.86, ],
        # [6230.65, -3222.48, 2597.55, -4653.45, ],
        # [1423.71, -3350.62, 3370.35, -732.41, ],
        # [-1251.09, 1017.12, 3377.25, -3084.71, ],
        # [1393.11, 567.08, 1066.05, -3625.16, ],
        [9568.20, -1954.78, -6728.08, 1.80],
        [9679.44, -9815.10, 2424.67, 1.39],
        [1610.73, 2192.66, 5428.78, -22385.79],
        [8571.15, 2945.31, -2174.37, -22384.61],
        [8610.92, 8887.24, -3690.93, -25555.04],
        [8612.37, -5273.09, 3779.56, -16405.40],
        [6864.52, -4144.30, -358.82, -8330.11],
        [1465.66, 2095.35, -3899.67, 385.29],
        # [-3602.33, 298.79, 5238.63, 576.91,  ],
        # [-3524.99, 5434.89, -1170.04, 734.34,  ],
        # [4091.48, 554.49, -5127.85, 731.61,  ],
        # [4030.31, -5413.11, 2056.51, 573.32,  ],
        # [2169.85, 2176.20, 2286.17, -19236.88,  ],
        # [2678.39, 6333.78, -2229.26, -18149.32,  ],
        # [8050.31, 3287.79, -4644.66, -17241.76,  ],
        # [7807.11, -4014.60, 3629.79, -18006.33,  ],
        # [8050.47, -5316.00, 5367.43, -17543.66,  ],
        # [-1123.25, 2413.00, 7189.07, -18498.38,  ],
        # [-1377.10, 4754.60, 4130.07, -18811.40,  ],
        # [6793.76, 6516.43, 6978.43, -38306.95,  ],
        # [6993.55, 8981.18, 4328.00, -37597.85,  ],
        # [8431.25, 6675.74, 4911.03, -37958.17,  ],
        # [10050.68, 5693.20, 4288.40, -37539.39,  ],
        # [10050.68, 3215.88, 7138.63, -37678.89,  ],
        # [10188.45, 2064.54, 8630.30, -37326.44,  ],
        # [6843.29, 7683.97, 5624.65, -38093.35,  ],
        # [10050.67, 5693.11, 4287.45, -37539.31,  ],
        # [6993.59, 4277.63, 9730.37, -37878.09,  ],
        # [10036.10, -10489.40, 2532.32, 2636.98,  ],
        # [-5074.01, 689.36, 7851.53, 1293.21,  ],
        # [2397.58, -335.64, 5499.65, -18922.93,  ],
        # [3723.43, 8847.10, 8508.66, -37724.91,  ],
        # [3778.36, 9895.85, 7309.68, -37514.10,  ],
    ]
)


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
params_buildup = 4  # four spool radii, one spool buildup factor
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
                    + flex_distance(
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
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin
):
    line_lengths_when_at_origin_err = np.linalg.norm(anchors, 2, 1) - line_lengths_when_at_origin
    line_length_samp = np.zeros((np.size(motor_pos_samp, 0), 3))
    if use_flex:
        line_length_samp = motor_pos_samples_to_distances_relative_to_origin(
            motor_pos_samp, spool_buildup_factor, spool_r
        ) + flex_distance(
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
        anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin
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
            spool_r_in_origin_first_guess[1] - 1.0,
            spool_r_in_origin_first_guess[2] - 1.0,
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
            spool_r_in_origin_first_guess[1] + 5.0,
            spool_r_in_origin_first_guess[2] + 5.0,
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
        + list(spool_r_in_origin_first_guess)
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
        for i in range(8):
            if disp:
                print("Try: %d/8" % i)
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
        "\nM669 A%.2f:%.2f:%.2f B%.2f:%.2f:%.2f C%.2f:%.2f:%.2f D%.2f:%.2f:%.2f\nM666 Q%.6f R%.3f:%.3f:%.3f:%.3f\n"
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
            spool_buildup_factor,
            spool_r[A],
            spool_r[B],
            spool_r[C],
            spool_r[D],
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
            return cost_sq_for_pos_samp(
                # return cost_sq_for_pos_samp_forward_transform(
                anch,
                pos + solution[-(params_perturb + 1) : -1],
                motor_pos_samp,
                constant_spool_buildup_factor,
                spool_r,
                line_lengths_when_at_origin,
                line_max_force,
            )
        else:
            return cost_sq_for_pos_samp(
                # return cost_sq_for_pos_samp_forward_transform(
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
