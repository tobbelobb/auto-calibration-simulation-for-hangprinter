#!/usr/bin/python3

"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division
import numpy as np
import scipy.optimize
import argparse
import timeit
import sys
import warnings

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
       [ 1.79255679e+01, -1.76352632e+01, -2.74971510e+01],
       [-5.23885148e+02, -5.01061599e+02,  5.07741532e+02],
       [-5.38352510e+02, -5.30255594e+02,  9.53040004e+02],
       [-5.45189816e+02, -5.58037791e-01,  2.64689901e+01],
       [-5.20999273e+02, -3.27341154e+00,  5.37019049e+02],
       [-4.70799477e+02,  1.65498517e+01,  1.04717018e+03],
       [-5.05362595e+02,  4.67529134e+02, -4.25453256e+01],
       [-5.13234032e+02,  4.64122399e+02,  5.33480869e+02],
       [-5.38293120e+02,  5.26060302e+02,  1.03039159e+03],
       [ 8.98405788e+00, -5.20642383e+02, -2.66148823e+01],
       [-5.52678502e+00, -5.14840431e+02,  5.17628213e+02],
       [-3.86000122e+01, -4.92617793e+02,  9.91936222e+02],
       [-5.07629503e+02, -4.65976040e+02,  3.17537695e+01],
       [-3.05601452e+01,  3.59665477e+01,  4.58719776e+02],
       [ 4.32898928e+01,  2.88400594e+01,  1.04672968e+03],
       [ 1.05874156e+01,  5.04407891e+02, -4.42588518e+01],
       [-1.99804722e+01,  5.20417440e+02,  5.17004705e+02],
       [-3.79958866e+01,  4.54266612e+02,  9.75222737e+02],
       [ 4.55930483e+02, -4.87688259e+02,  2.60515823e+01],
       [ 4.82712486e+02, -5.20125685e+02,  5.35481389e+02],
       [ 4.95517745e+02, -5.39356090e+02,  9.65228753e+02],
       [ 5.00096373e+02, -2.24374570e+01, -4.20139801e+01],
       [ 4.90657689e+02, -3.04215669e+01,  5.27319817e+02],
       [ 5.06742907e+02, -3.19689121e+01,  9.98360926e+02],
       [ 4.90590662e+02,  5.23118645e+02,  4.14092948e+01],
       [ 5.17848381e+02,  5.01639033e+02,  4.82878157e+02],
       [ 5.25399857e+02,  4.59863258e+02,  9.81665044e+02]
    ]
)

# Generated with no flex compensation
# motor_pos_samp = pos_to_motor_pos_samples(anchors, pos, 0, False)
#motor_pos_samp = np.array(
#    [
#       [  -369.97133086,   -375.66959393,    316.34560633,
#           321.94609589,   1077.75975713],
#       [ -5759.01057516,  12615.08424755,  12275.59850843,
#         -6281.97726753, -13229.98337407],
#       [ -2060.97219543,  15716.33224264,  15602.69034285,
#         -2227.35643322, -27381.62276038],
#       [  1442.4251883 ,  10668.27508761,   1463.3685043 ,
#        -10512.72933158,   1849.83037262],
#       [  3159.98245383,  11763.18458403,   3277.83725587,
#         -7486.81252308, -17322.05908512],
#       [  7291.94468887,  14266.33816963,   6743.79551879,
#         -1553.68867173, -36100.64439146],
#       [ 10090.57203732,  10675.889441  ,  -7542.39424108,
#         -8463.82172607,   6068.07267818],
#       [ 11669.77976994,  12405.8842912 ,  -5005.52677327,
#         -6115.64295821, -14676.39535396],
#       [ 16129.59499481,  16299.63450814,  -1127.9130486 ,
#         -1373.47262378, -29820.36571367],
#       [-10122.43960565,   1100.72500198,  10138.58559936,
#          1439.38883774,   3605.756548  ],
#       [ -7527.74534992,   3176.81909782,  11551.45807825,
#          2977.18307984, -16695.41664699],
#       [ -2531.87450143,   7228.3147109 ,  14207.41418159,
#          5938.05560439, -33823.45808762],
#       [ -7398.16588201,  10779.40735575,  10135.92434965,
#         -8409.43737118,   3330.02537493],
#       [  2207.56498303,   2107.80296987,    860.35188277,
#           963.14232699, -17745.30959353],
#       [  6582.41640758,   5361.04628015,   5608.31555101,
#          6823.30727184, -40219.08056667],
#       [  9809.00939173,    977.4905076 ,  -9827.85176463,
#          1377.48954933,   4119.45515998],
#       [ 11655.94023884,   3460.42828979,  -7629.63832939,
#          2739.08083977, -16597.3181957 ],
#       [ 13385.27733777,   6915.40099512,  -2118.11173865,
#          5637.00997696, -33863.99419124],
#       [ -8108.90080311,  -7342.20860536,  10353.26565632,
#          9859.27771014,   3267.94097714],
#       [ -6116.93384181,  -5267.96158535,  12609.82866003,
#         12052.1408033 , -14452.27937288],
#       [ -2290.52818859,  -1396.80907196,  15701.50047006,
#         15083.33018867, -28278.36387567],
#       [   734.77431785,  -9739.7352902 ,   1582.85973463,
#          9727.42550404,   3997.73874831],
#       [  2473.96096152,  -7017.4007458 ,   3574.18124601,
#         11142.58224923, -17338.278132  ],
#       [  6168.36875155,  -2680.22457923,   7234.30238782,
#         14510.73989763, -33810.94743147],
#       [ 11178.91365375,  -7734.00545613,  -8529.47143446,
#         10680.97899928,   3344.9573659 ],
#       [ 12148.6236308 ,  -6352.25541301,  -5979.22868207,
#         12390.52086778, -12429.90917227],
#       [ 14529.08862584,  -2070.20634574,   -749.1973495 ,
#         15459.74674515, -29542.51506619]
#    ]
#)

# Generated with flex compensation
# motor_pos_samples = pos_to_motor_pos_samples(anchors, pos, 20, True)
motor_pos_samp = np.array(
    [
       [  -369.58609387,   -375.29396855,    317.91184282,
           323.52214156,   1076.80959977],
       [ -5752.0896211 ,  12684.81746173,  12344.08999181,
         -6275.70760911, -13233.40119754],
       [ -2048.83305984,  15817.17840356,  15702.84011251,
         -2215.62052102, -27377.19026704],
       [  1465.31821991,  10710.4339081 ,   1486.30672549,
        -10502.25101052,   1843.9394221 ],
       [  3192.47800976,  11825.29426292,   3310.74909443,
         -7479.34319472, -17312.3368333 ],
       [  7338.19413077,  14353.88288287,   6786.70764259,
         -1552.13573883, -36078.36378187],
       [ 10132.61635504,  10719.09762197,  -7533.76001979,
         -8455.38023592,   6051.956053  ],
       [ 11735.89557575,  12474.71717231,  -4997.83528125,
         -6109.53899159, -14677.40138713],
       [ 16233.96918729,  16405.12479906,  -1110.42815967,
         -1356.59254137, -29813.50295791],
       [-10112.34907465,   1121.68560353,  10177.34821782,
          1461.03157744,   3598.89997665],
       [ -7520.23527999,   3209.13006916,  11611.99173652,
          3008.80369674, -16686.02305477],
       [ -2529.34463269,   7275.98142332,  14293.8205203 ,
          5978.38068699, -33803.02264388],
       [ -7389.4678355 ,  10825.3409007 ,  10180.48128046,
         -8401.04998007,   3316.07110403],
       [  2211.40671242,   2111.29269087,    859.49129879,
           962.6350027 , -17732.81159413],
       [  6588.07191703,   5355.67619644,   5605.19961097,
          6831.09313097, -40190.41314644],
       [  9846.17173913,    997.64776716,  -9818.05399677,
          1398.42927125,   4112.4891462 ],
       [ 11716.99576615,   3493.87937612,  -7622.02684706,
          2770.03585674, -16588.12507688],
       [ 13465.52800333,   6960.31851935,  -2115.99503104,
          5674.69661079, -33842.3921186 ],
       [ -8100.81240478,  -7333.85350166,  10397.20035247,
          9902.17005513,   3254.46395956],
       [ -6110.82858923,  -5260.68626986,  12680.08218414,
         12120.30954553, -14454.20707365],
       [ -2281.85299886,  -1385.74175751,  15802.3865989 ,
         15180.39274419, -28273.03733673],
       [   754.41149621,  -9730.0251034 ,   1604.15387749,
          9764.39044336,   3990.93856291],
       [  2503.16665884,  -7010.39870999,   3607.2108333 ,
         11201.26908614, -17327.98898451],
       [  6210.27110993,  -2677.54660214,   7282.28650837,
         14599.417566  , -33790.95427313],
       [ 11227.05547733,  -7725.29526968,  -8520.96464144,
         10728.03213827,   3330.1585467 ],
       [ 12215.56515786,  -6345.91575306,  -5972.44605709,
         12458.32108416, -12433.76787772],
       [ 14622.6285915 ,  -2065.60330095,   -740.5190987 ,
         15559.1232983 , -29535.92901142]
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
low_axis_min_force_limit = 0
low_axis_max_force_limit = 120

l_long = 14000.0  # The longest distance from the origin that we should consider for anchor positions
l_short = 3000.0  # The longest distance from the origin that we should consider for data point collection
data_z_min = -100.0  # The lowest z-coordinate the algorithm should care about guessing
xyz_offset_max = (
    1.0  # Tell the algorithm to check if all xyz-data may carry an offset error compared to the encoder-data
)

use_forces = False

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
    use_flex,
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
    if use_flex:
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
    use_flex,
    use_line_lengths,
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


    if not use_flex:
        # These are rotational errors
        #synthetic_motor_samp = pos_to_motor_pos_samples(anchors, pos, low_axis_max_force, use_flex, spool_r_in_origin=spool_r)
        #err += np.sum(np.sqrt(np.sum(pow((synthetic_motor_samp - motor_pos_samp) / mechanical_advantage, 2))))
        # These are not rotational errors
        err += np.sum(pow(
            distance_samples_relative_to_origin(anchors, pos)
            - motor_pos_samples_to_distances_relative_to_origin(motor_pos_samp, spool_buildup_factor, spool_r),
            2,
        ))

    if use_flex:
        # Implies use_rotational_errors
        synthetic_motor_samp = pos_to_motor_pos_samples(anchors, pos, low_axis_max_force, use_flex, spool_r_in_origin=spool_r)
        err += np.sum(np.sqrt(np.sum(pow((synthetic_motor_samp - motor_pos_samp) / mechanical_advantage, 2))))
        # Add error due to flex
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

    if use_line_lengths:
        line_lengths_when_at_origin_err = np.linalg.norm(anchors, 2, 1) - line_lengths_when_at_origin
        err += np.sum(abs(line_lengths_when_at_origin_err.dot(line_lengths_when_at_origin_err)))

    if use_forces:
        err += cost_from_forces(anchors, pos, force_samp, mover_weight, low_axis_max_force)

    if printit:
        synthetic_motor_samp = pos_to_motor_pos_samples(anchors, pos, low_axis_max_force, use_flex, spool_r_in_origin=spool_r)
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

    if use_line_lengths:
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


def solve(motor_pos_samp, xyz_of_samp, line_lengths_when_at_origin, use_flex, use_line_lengths, debug=False):
    """Find reasonable positions and anchors given a set of samples."""

    if use_flex:
        print("Using flex compensation")
    else:
        print("Assuming zero flex")

    if use_line_lengths:
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
        use_flex,
        use_line_lengths,
        low_axis_max_force=120.0,
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
            use_flex,
            use_line_lengths,
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
        + [spool_r_in_origin_first_guess[0] - 0.50, spool_r_in_origin_first_guess[4] - 0.50]
        + [-xyz_offset_max, -xyz_offset_max, -xyz_offset_max]
    )
    if use_flex:
        lb = np.append(lb, low_axis_min_force_limit)

    ub = np.array(
        [
            l_long,  # A_ax < x
               0.0,  # A_ay < x
               0.0,  # A_az < x
            l_long,  # A_bx < x
            l_long,  # A_by < x
               0.0,  # A_bz < x
            l_long,  # A_cx < x
            l_long,  # A_cy < x
               0.0,  # A_cz < x
               0.0,  # A_dx < x
            l_long,  # A_dy < x
               0.0,  # A_dz < x
             500.0,  # A_ix < x
             500.0,  # A_iy < x
            l_long,  # A_iz < x
        ]
        + [l_short, l_short, 2.0 * l_short] * (u - ux)
        + [spool_r_in_origin_first_guess[0] + 1.5, spool_r_in_origin_first_guess[4] + 1.5]
        + [xyz_offset_max, xyz_offset_max, xyz_offset_max]
    )

    if use_flex:
        ub = np.append(ub, low_axis_max_force_limit)

    #pos_est = 500.0*np.random.random((u - ux, 3)) - 250.0  # The positions we need to estimate
    #anchors_est = symmetric_anchors(
    #    1500
    #)  # np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # Start at zeros
    pos_est = np.zeros((u - ux, 3))  # The positions we need to estimate
    anchors_est = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x_guess = (
        list(anchorsmatrix2vec(anchors_est))[0:params_anch]
        + list(posmatrix2vec(pos_est))
        + list([spool_r_in_origin_first_guess[0], spool_r_in_origin_first_guess[4]])
        + [0, 0, 0]
    )
    maxiter = 1500
    if use_flex:
        x_guess += [0.0]
        maxiter = 500

    disp = False
    if debug:
        disp = True

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

        with warnings.catch_warnings():
          warnings.filterwarnings('ignore', message='Values in x were outside bounds during a minimize step, clipping to bounds')
          sol = scipy.optimize.minimize(
              lambda x: costx(
                  cost_sq_for_pos_samp,
                  # cost_sq_for_pos_samp_forward_transform,
                  # cost_sq_for_pos_samp_combined,
                  x[params_anch : -(params_buildup + params_perturb + use_flex)],
                  x[0:params_anch],
                  constant_spool_buildup_factor,
                  x[-(params_buildup + params_perturb + use_flex) : -(params_perturb + use_flex)],
                  u,
                  line_lengths_when_at_origin,
                  x[-(params_perturb + use_flex):(x.size - use_flex)],
                  use_flex,
                  use_line_lengths,
                  x[-1],
              ),
              random_guess,
              method="SLSQP",
              bounds=list(zip(lb, ub)),
              options={"disp": disp, "ftol": 1e-9, "maxiter": maxiter},
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

    use_flex = args["advanced"]
    use_line_lengths = args["advanced"]
    #use_line_lengths = False

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
        # return cost_sq_for_pos_samp_forward_transform(
        return cost_sq_for_pos_samp(
            anch,
            pos + solution[-(params_perturb + use_flex):(solution.size - use_flex)],
            motor_pos_samp,
            constant_spool_buildup_factor,
            spool_r,
            line_lengths_when_at_origin,
            use_flex,
            use_line_lengths,
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
    the_cand = candidate(
        "no_name",
        solve(motor_pos_samp, xyz_of_samp, line_lengths_when_at_origin, use_flex, use_line_lengths, args["debug"]),
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
                use_flex,
                use_line_lengths,
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
