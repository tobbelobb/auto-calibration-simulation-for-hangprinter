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

# Config values should be based on your physical machine
constant_spool_buildup_factor = 0.006875 * 10  # Qualified first guess for 1.1 mm line
spool_r_in_origin_first_guess = np.array([75.0, 75.0, 75.0, 75.0, 75.0])
spool_gear_teeth = 255
motor_gear_teeth = 20
mechanical_advantage = np.array([2.0, 2.0, 2.0, 2.0, 4.0])
lines_per_spool = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
springKPerUnitLength = 20000.0
mover_weight = 2.0

## Set use_advanced = False if you have a few xyz_of_samp and an abundance of motor_pos samp.
## Set use_advanced = True  if you have only a few motor_pos_samp
use_advanced = False


line_lengths_when_at_origin = np.array([2003.59676582, 2003.59676582, 2003.59676582, 2003.59676582, 2000.])

xyz_of_samp = np.array(
    [
       [-1.58825333e+01, -2.59800500e+01,  3.14318400e+01],
       [-1.88119329e+02, -1.87956224e+02,  2.25112072e+02],
       [-2.05721889e+02, -1.88517340e+02,  4.01833700e+02],
    ]
)

# Generated with no flex compensation
# motor_pos_samples = pos_to_motor_pos_samples(anchors, pos)
motor_pos_samp = np.array(
    [
       [  -461.88115732,    353.26918953,    547.43686961,
          -263.73154369,  -1214.981907  ],
       [ -2908.01556397,   4282.33644063,   4279.47017583,
         -2911.43673094,  -7978.79821393],
       [ -2088.07550613,   5284.85923333,   4988.20221246,
         -2442.75400058, -14646.65856109],
       [   667.75651095,   2941.80994764,   -458.89136295,
         -2940.97818842,    462.44463301],
       [  -110.45155032,   3409.88139212,   1130.57465467,
         -2603.47140643,  -7078.96142876],
       [  2043.44252387,   5047.23999715,    747.37788663,
         -2599.20920467, -14812.41959874],
       [  4598.15530873,   4697.90989172,  -4026.28664401,
         -4150.29922059,    321.09343903],
       [  5308.91715873,   3878.53829854,  -4252.88836512,
         -2511.31296268,  -6202.15897315],
       [  5082.40744018,   5169.38956566,  -2107.9573385 ,
         -2211.63614742, -15084.37008429],
       [ -3958.53387372,    551.24894324,   4025.18406782,
           -90.22985729,   -381.91258828],
       [ -3346.37590635,   1264.11255495,   4014.01080461,
          -276.31555963,  -5940.27538673],
       [ -2884.73110765,    589.87248208,   5373.84861705,
          2293.10243262, -14825.41406898],
       [ -4330.37063959,   4000.19990097,   4654.20384265,
         -3526.72378124,   2076.09125022],
       [   557.52673466,   1364.98724304,    572.33251852,
          -251.20482177,  -9306.66241   ],
       [   555.41987022,    968.44335815,   2212.52515483,
          1811.75760526, -16664.78611131],
       [  3674.48017258,   -660.2868102 ,  -3593.33797371,
          1052.20965006,   -437.05389452],
       [  5058.95097975,    644.33193566,  -4249.51584536,
           698.8856142 ,  -6711.13744807],
       [  4309.8394761 ,   1994.2643967 ,  -1624.24977873,
           896.13914236, -15985.05296962],
       [ -4226.98098217,  -4396.76323456,   4758.50365237,
          4893.68334632,   1992.83897614],
       [ -2279.24832759,  -2820.68069681,   3615.88536212,
          4078.41769984,  -8136.72019128],
       [ -1976.10488366,  -2756.20960714,   4480.58843843,
          5136.04895473, -13086.62527733],
       [   322.57556755,  -4479.88335141,    197.91554529,
          4498.4885926 ,    453.02787327],
       [   632.74533886,  -2814.91999945,    782.88111803,
          3949.56394814,  -9012.41411774],
       [  1277.19679188,  -2172.58473421,   1269.31318565,
          4453.38232427, -14363.81796633],
       [  3320.70669414,  -4292.26500022,  -2811.5220689 ,
          4549.18748964,    826.6771929 ],
       [  3940.13090639,  -3407.54767911,  -2533.50839887,
          4670.79425717,  -7668.32297016],
       [  4279.06016137,  -3014.2559298 ,  -1423.09632764,
          5617.11007048, -14170.30135705],
    ]
)

force_samp = np.array(
    [
    ]
)

#################################################################################
## Warning! User interface ends here. Edit below this line at your own risk. ####
#################################################################################
## Algorithm help and tuning
low_axis_min_force_limit = 8
low_axis_max_force_limit = 140

l_long = 14000.0  # The longest distance from the origin that we should consider for anchor positions
l_short = 3000.0  # The longest distance from the origin that we should consider for data point collection
data_z_min = -100.0  # The lowest z-coordinate the algorithm should care about guessing
xyz_offset_max = (
    1.0  # Tell the algorithm to check if all xyz-data may carry an offset error compared to the encoder-data
)

# Rotational errors are just harder to use, but sometimes faster, if you have huge data sets
# Combine with use_flex_errors = True can improve convergence (make it faster)
# Use flex error is you have approximately as many xyz_of_samp as you have motor_pos_samp
use_flex_errors = use_advanced
use_rotational_errors = use_advanced
use_flex_in_rotational_errors = use_advanced
use_line_lengths_at_origin_data = False
use_forces = False
use_flex = (use_flex_errors or use_flex_in_rotational_errors)

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
I = 4
X = 0
Y = 1
Z = 2
params_anch = 15
params_buildup = 2
params_perturb = 3


def symmetric_anchors(l, az=-120.0, bz=-120.0, cz=-120.0, dz=-120.0):
    anchors = np.array(np.zeros((5, 3)))
    anchors[A, X] = 0
    anchors[A, Y] = -l
    anchors[A, Z] = az
    anchors[B, X] = l
    anchors[B, Y] = 0
    anchors[B, Z] = bz
    anchors[C, X] = 0
    anchors[C, Y] = l
    anchors[C, Z] = cz
    anchors[D, X] = -l
    anchors[D, Y] = 0
    anchors[D, Z] = dz
    anchors[I, X] = 0
    anchors[I, Y] = 0
    anchors[I, Z] = l
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
    index_closest_to_origin = int(np.shape(pos)[0] / 2) - int(n / 2)
    # Make pos[0] a point fairly close to origin
    tmp = pos[0].copy()
    pos[0] = pos[index_closest_to_origin]
    pos[index_closest_to_origin] = tmp
    return pos


def distance_samples_relative_to_origin(anchors, pos):
    """Possible relative line length measurments according to anchors and position.

    Parameters
    ----------
    anchors : 5x3 matrix of anhcor positions in mm
    pos : ux3 matrix of positions
    fuzz: Maximum measurment error per motor in mm
    """
    # pos[:,np.newaxis,:]: ux1x3
    # Broadcasting happens u times and we get ux5x3 output before norm operation
    line_lengths = np.linalg.norm(anchors - pos[:, np.newaxis, :], 2, 2)
    return line_lengths - np.linalg.norm(anchors, 2, 1)


def pos_to_motor_pos_samples(
    anchors,
    pos,
    low_axis_max_force,
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
    if use_flex_in_rotational_errors:
        relative_line_lengths += flex_distance(
            low_axis_max_force,
            np.max(np.array([low_axis_max_force - 1, 0.0001])),
            anchors,
            pos,
            mechanical_advantage,
            springKPerUnitLength,
            mover_weight,
        )
    motor_positions = k0 * (np.sqrt(abs(spool_r_in_origin_sq + relative_line_lengths * k2)) - spool_r_in_origin)

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


def cost_from_forces(anchors, pos, force_samps, mover_weight, low_axis_max_force):

    pos_w_origin = np.r_[[[0.0, 0.0, 0.0]], pos]
    anch_to_pos = anchors - pos_w_origin[:, np.newaxis, :]
    distances = np.linalg.norm(anch_to_pos, 2, 2)
    [low_forces_pre, top_forces_pre, low_forces_grav, top_forces_grav] = forces_gravity_and_pretension(
        low_axis_max_force, np.max(np.array([low_axis_max_force - 1, 0])), anch_to_pos, distances, mover_weight
    )

    synthetic_forces_pre = np.c_[low_forces_pre, top_forces_pre][1:]
    synthetic_forces_grav = np.c_[low_forces_grav, top_forces_grav][1:]

    # Remove gravity related forces from force_samp
    force_samps_pre = force_samps - synthetic_forces_grav

    # Normalize. we don't care about pretension force sizes
    synthetic_forces_pre = synthetic_forces_pre / np.linalg.norm(synthetic_forces_pre, 2, 1)[:, np.newaxis]
    force_samps_pre = force_samps_pre / np.linalg.norm(force_samps_pre, 2, 1)[:, np.newaxis]

    return np.sum(pow(synthetic_forces_pre - force_samps_pre, 2)) * 1000.0



def cost_sq_for_pos_samp(
    anchors,
    pos,
    motor_pos_samp,
    spool_buildup_factor,
    spool_r,
    line_lengths_when_at_origin,
    low_axis_max_force=1,
    printit=False,
):
    """
    Sum of squares

    Creates samples based on guessed anchor and data collection positions.
    May take line flex into account when generating the samples.

    For all samples sum
    (Sample value if anchor position A and cartesian position x were guessed - actual sample)^2

    Then, some variations of that
    """

    err = 0


    if not (use_rotational_errors) and not (use_flex_errors):
        err += np.sum(pow(
            distance_samples_relative_to_origin(anchors, pos)
            - motor_pos_samples_to_distances_relative_to_origin(motor_pos_samp, spool_buildup_factor, spool_r),
            2,
        ))

    if use_rotational_errors:
        synthetic_motor_samp = pos_to_motor_pos_samples(anchors, pos, low_axis_max_force, spool_r_in_origin=spool_r)
        err += np.sum(np.sqrt(np.sum(pow((synthetic_motor_samp - motor_pos_samp) / mechanical_advantage, 2))))

    if use_flex_errors:
        err += np.sum(pow(
            distance_samples_relative_to_origin(anchors, pos)
            - (
                motor_pos_samples_to_distances_relative_to_origin(motor_pos_samp, spool_buildup_factor, spool_r)
                - flex_distance(
                    low_axis_max_force,
                    np.max(np.array([low_axis_max_force - 1, 0.0001])),
                    anchors,
                    pos,
                    mechanical_advantage,
                    springKPerUnitLength,
                    mover_weight,
                )
            ),
            2,
        ))

    if use_line_lengths_at_origin_data:
        line_lengths_when_at_origin_err = np.linalg.norm(anchors, 2, 1) - line_lengths_when_at_origin
        err += np.sum(abs(line_lengths_when_at_origin_err.dot(line_lengths_when_at_origin_err)))

    if use_forces:
        err += cost_from_forces(anchors, pos, force_samp, mover_weight, low_axis_max_force)

    if printit:
        synthetic_motor_samp = pos_to_motor_pos_samples(anchors, pos, low_axis_max_force, spool_r_in_origin=spool_r)
        print("Rotational errors:")
        print((synthetic_motor_samp - motor_pos_samp) / mechanical_advantage)


    return err


def cost_sq_for_pos_samp_forward_transform(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, low_axis_max_force=1
):
    line_lengths_when_at_origin_err = np.linalg.norm(anchors, 2, 1) - line_lengths_when_at_origin
    line_length_samp = np.zeros((np.size(motor_pos_samp, 0), 3))
    if use_flex_errors:
        line_length_samp = motor_pos_samples_to_distances_relative_to_origin(
            motor_pos_samp, spool_buildup_factor, spool_r
        ) - flex_distance(
            low_axis_max_force,
            np.max(np.array([low_axis_max_force - 1, 0.0001])),
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
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, low_axis_max_force=1
):
    return 10 * cost_sq_for_pos_samp_forward_transform(
        anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, low_axis_max_force
    ) + cost_sq_for_pos_samp(
        anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r, line_lengths_when_at_origin, low_axis_max_force
    )


def anchorsvec2matrix(anchorsvec):
    """Create a 5x3 anchors matrix from anchors vector."""
    anchors = np.array(
        [
            [anchorsvec[0], anchorsvec[1], anchorsvec[2]],
            [anchorsvec[3], anchorsvec[4], anchorsvec[5]],
            [anchorsvec[6], anchorsvec[7], anchorsvec[8]],
            [anchorsvec[9], anchorsvec[10], anchorsvec[11]],
            [anchorsvec[12], anchorsvec[13], anchorsvec[14]],
        ]
    )

    return anchors


def anchorsmatrix2vec(a):
    return [a[A, X], a[A, Y], a[A, Z], a[B, X], a[B, Y], a[B, Z], a[C, X], a[C, Y], a[C, Z], a[D, X], a[D, Y], a[D, Z], a[I, X], a[I, Y], a[I, Z]]


def posvec2matrix(v, u):
    return np.reshape(v, (u, 3))


def posmatrix2vec(m):
    return np.reshape(m, np.shape(m)[0] * 3)


def pre_list(l, num):
    return np.append(np.append(l[0:params_anch], l[params_anch : params_anch + 3 * num]), l[-params_buildup:])


def solve(motor_pos_samp, xyz_of_samp, line_lengths_when_at_origin, method, debug=False):
    """Find reasonable positions and anchors given a set of samples."""

    print(method)
    if use_flex_errors:
        print("Using flex compensation")
    else:
        print("Assuming zero flex")

    if use_rotational_errors:
        print("Using rotational errors")
        if use_flex_in_rotational_errors:
            print("... using flex compensation in rotational error")
        else:
            print("... not using flex compensation in rotational error")
    else:
        print("Not using rotational error")

    if use_line_lengths_at_origin_data:
        print("Using hand measured line lengths at the origin")
    else:
        print("Not forcing hand measured line lengths")

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
        low_axis_max_force=1.0,
    ):
        """Identical to cost, except the shape of inputs and capture of samp, xyz_of_samp, ux, and u

        Parameters
        ----------
        x : [A_ay A_az A_bx A_by A_bz A_cx A_cy A_cz A_dz A_ix A_iy A_iz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """

        if len(posvec) > 0:
            posvec = np.array([pos for pos in posvec])
        anchvec = np.array([anch for anch in anchvec])
        spool_r = np.array([r for r in spool_r])
        spool_r = np.r_[spool_r[0], spool_r[0], spool_r[0], spool_r]
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
            low_axis_max_force,
        )

    # Limits of anchor positions:
    lb = np.array(
        [
            -l_long,  # A_ax > x
            -l_long,  # A_ay > x
            -1300.0,  # A_az > x
                0.0,  # A_bx > x
            -l_long,  # A_by > x
            -1300.0,  # A_bz > x
            -l_long,  # A_cx > x
                0.0,  # A_cy > x
            -1300.0,  # A_cz > x
            -l_long,  # A_dx > x
            -l_long,  # A_dy > x
            -1300.0,  # A_dz > x
             -500.0,  # A_ix > x
             -500.0,  # A_iy > x
                0.0,  # A_iz > x
        ]
        + [-l_short, -l_short, data_z_min] * (u - ux)
        + [spool_r_in_origin_first_guess[0] - 0.50, spool_r_in_origin_first_guess[3] - 0.50]
        + [-xyz_offset_max, -xyz_offset_max, -xyz_offset_max]
    )
    if use_flex:
        lb = np.append(lb, low_axis_min_force_limit)

    ub = np.array(
        [
            l_long,  # A_ax < x
               0.0,  # A_ay < x
             200.0,  # A_az < x
            l_long,  # A_bx < x
            l_long,  # A_by < x
             200.0,  # A_bz < x
            l_long,  # A_cx < x
            l_long,  # A_cy < x
             200.0,  # A_cz < x
               0.0,  # A_dx < x
            l_long,  # A_dy < x
             200.0,  # A_dz < x
             500.0,  # A_ix < x
             500.0,  # A_iy < x
            l_long,  # A_iz < x
        ]
        + [l_short, l_short, 2.0 * l_short] * (u - ux)
        + [spool_r_in_origin_first_guess[0] + 1.5, spool_r_in_origin_first_guess[3] + 1.5]
        + [xyz_offset_max, xyz_offset_max, xyz_offset_max]
    )
    if use_flex:
        ub = np.append(ub, low_axis_max_force_limit)

    pos_est = np.zeros((u - ux, 3))  # The positions we need to estimate
    #pos_est = 500.0*np.random.random((u - ux, 3))  # The positions we need to estimate
    anchors_est = symmetric_anchors(
        1500
    )  # np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x_guess = (
        list(anchorsmatrix2vec(anchors_est))[0:params_anch]
        + list(posmatrix2vec(pos_est))
        + list([spool_r_in_origin_first_guess[0], spool_r_in_origin_first_guess[4]])
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

        stop = Or(VTR(1e-12), ChangeOverGeneration(0.001, 500))
        ndim = number_of_params_pos + params_anch + params_buildup + params_perturb + use_flex
        npop = 3
        stepmon = VerboseMonitor(100)
        if not disp:
            stepmon = Monitor()
        solver = DifferentialEvolutionSolver2(ndim, npop)

        solver.SetRandomInitialPoints(lb, ub)
        solver.SetStrictRanges(lb, ub)
        solver.SetGenerationMonitor(stepmon)
        solver.SetEvaluationLimits(evaluations=3200000, generations=10000)
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
                    options={"disp": disp, "ftol": 1e-9, "maxiter": 1500},
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
        "\nM669 A%.2f:%.2f:%.2f B%.2f:%.2f:%.2f C%.2f:%.2f:%.2f D%.2f:%.2f:%.2f I%.2f:%.2f:%.2f\nM666 R%.3f:%.3f:%.3f:%.3f:%.3f\n;Here follows constants that are set in the script\nM666 Q%.6f W%.2f S%.2f U%d:%d:%d:%d:%d O%d:%d:%d:%d:%d L%d:%d:%d:%d:%d H%d:%d:%d:%d:%d"
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
            anch[I, X],
            anch[I, Y],
            anch[I, Z],
            spool_r[A],
            spool_r[B],
            spool_r[C],
            spool_r[D],
            spool_r[I],
            spool_buildup_factor,
            mover_weight,
            springKPerUnitLength,
            mechanical_advantage[A],
            mechanical_advantage[B],
            mechanical_advantage[C],
            mechanical_advantage[D],
            mechanical_advantage[I],
            lines_per_spool[A],
            lines_per_spool[B],
            lines_per_spool[C],
            lines_per_spool[D],
            lines_per_spool[I],
            motor_gear_teeth, # A
            motor_gear_teeth, # B
            motor_gear_teeth, # C
            motor_gear_teeth, # D
            motor_gear_teeth, # I
            spool_gear_teeth, # A
            spool_gear_teeth, # B
            spool_gear_teeth, # C
            spool_gear_teeth, # D
            spool_gear_teeth, # I
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
    print("Err_I_X: %9.3f" % (float(sol_anch[I, X]) - (anchors[I, X])))
    print("Err_I_Y: %9.3f" % (float(sol_anch[I, Y]) - (anchors[I, Y])))
    print("Err_I_Z: %9.3f" % (float(sol_anch[I, Z]) - (anchors[I, Z])))


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(Store_as_array, self).__call__(parser, namespace, values, option_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find best Hangprinter config based on true line lengths, line difference samples, and xyz positions if known."
    )
    parser.add_argument("-a", "--advanced", help="Use the advanced cost function", action="store_true")
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
        help="Specify the five line lenghts (A, B, C, D, and I) hand measured when nozzle is at the origin.",
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

    use_advanced = args["advanced"]
    use_flex_errors = use_advanced
    use_rotational_errors = use_advanced
    use_flex_in_rotational_errors = use_advanced
    use_flex = (use_flex_errors or use_flex_in_rotational_errors)

    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    anchors = np.array(
        [
            [0, -1620, -150],
            [1800, 0, -150],
            [0, 1620, -150],
            [-1800, 0, -150],
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
        if np.size(motor_pos_samp_) % 5 != 0:
            print("Please specify motor positions (angles) of sampling points.")
            print(
                "You specified %d numbers after your -s/--sample_data option, which is not a multiple of 5 number of numbers."
                % (np.size(motor_pos_samp_))
            )
            sys.exit(1)
        motor_pos_samp = motor_pos_samp_.reshape((int(np.size(motor_pos_samp_) / 5), 5))
    line_lengths_when_at_origin_ = args["line_lengths"]
    if np.size(line_lengths_when_at_origin_) != 0:
        if np.size(line_lengths_when_at_origin_) != 5:
            print("Please specify five measured line lengths.")
            print(
                "You specified %d numbers after your -l/--line_lengths option."
                % (np.size(line_lengths_when_at_origin_))
            )
            sys.exit(1)
        line_lengths_when_at_origin = line_lengths_when_at_origin_

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]

    motor_pos_samp = np.array([r for r in motor_pos_samp.reshape(u * 5)]).reshape((u, 5))
    xyz_of_samp = np.array([r for r in xyz_of_samp.reshape(ux * 3)]).reshape((ux, 3))

    if ux > u:
        print("Error: You have more xyz positions than samples!")
        print("You have %d xyz positions and %d samples" % (ux, u))
        sys.exit(1)

    def computeCost(solution):
        anch = np.zeros((5, 3))
        anch = anchorsvec2matrix(solution[0:params_anch])
        spool_buildup_factor = constant_spool_buildup_factor
        spool_r = np.array(
            [x for x in solution[-(params_buildup + params_perturb + use_flex) : -(params_perturb + use_flex)]]
        )
        spool_r = np.r_[spool_r[0], spool_r[0], spool_r[0], spool_r]

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
        anch = np.zeros((5, 3))
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
                self.spool_r = np.r_[self.spool_r[0], self.spool_r[0], self.spool_r[0], self.spool_r]
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

    samples_limit = 40
    xyz_coords_limit = 18
    enough_samples = u > samples_limit
    enough_xyz_coords = (3 * ux) > xyz_coords_limit
    cost_per_sample = the_cand.cost / u
    cost_per_sample_upper_limit = 10.0
    cost_per_sample_low_enough = cost_per_sample < cost_per_sample_upper_limit
    L_errs = np.linalg.norm(the_cand.anch, 2, 1) - line_lengths_when_at_origin
    line_length_error = np.linalg.norm(L_errs)
    line_length_error_upper_limit = 50.0
    line_length_error_low_enough = line_length_error < line_length_error_upper_limit

    print("Number of samples: %d (is above %s? %s)" % (u ,samples_limit, enough_samples))
    print("Input xyz coords:  %d (is above %s? %s)" % ((3 * ux), xyz_coords_limit, enough_xyz_coords))
    np.set_printoptions(suppress=False)
    if args["debug"]:
        print("Total cost:        %e" % the_cand.cost)
    print("Cost per sample:   %e (is below %s? %s)" % (cost_per_sample, cost_per_sample_upper_limit, cost_per_sample_low_enough))
    print("Line length error: %e (is below %s? %s)" % (line_length_error, line_length_error_upper_limit, line_length_error_low_enough))
    print("All quality conditions met? (%s)" % (enough_samples and enough_xyz_coords and cost_per_sample_low_enough and line_length_error_low_enough))
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
            "A=[%.2f, %.2f, %.2f]\nB=[%.2f, %.2f, %.2f]\nC=[%.2f, %.2f, %.2f]\nD=[%.2f, %.2f, %.2f]\nI=[%.2f, %.2f, %.2f]"
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
                the_cand.anch[I, X],
                the_cand.anch[I, Y],
                the_cand.anch[I, Z],
            )
        )

        if args["debug"]:
            cost_sq_for_pos_samp(
                the_cand.anch,
                the_cand.pos + the_cand.xyz_offset,
                motor_pos_samp,
                constant_spool_buildup_factor,
                the_cand.spool_r,
                line_lengths_when_at_origin,
                the_cand.line_max_force,
                printit=True,
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
        print("Length errors along each line:")
        print("line_length_error_a=%.2f" % (L_errs[0]))
        print("line_length_error_b=%.2f" % (L_errs[1]))
        print("line_length_error_c=%.2f" % (L_errs[2]))
        print("line_length_error_d=%.2f" % (L_errs[3]))
        print("line_length_error_i=%.2f" % (L_errs[4]))
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
