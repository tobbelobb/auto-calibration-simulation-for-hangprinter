"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division  # Always want 3/2 = 1.5
import numpy as np
import scipy.optimize
import argparse
import timeit
import sys

import signal
import time

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

import decimal
decimal.getcontext().prec = 25
# If you fear that truncation errors dominate the cost
# function computation, uncomment Dec = decimal.Decimal
# and comment out Dec = float
#Dec = decimal.Decimal
Dec = float


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
    fuzz[D, Z] = (
        l * fuzz_percentage * np.random.rand()
    )  # usually higher than A is long
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
    spool_buildup_factor=Dec(0.008),  # Qualified first guess for 0.5 mm line
    spool_r_in_origin=np.array([Dec(65.0), Dec(65.0), Dec(65.0), Dec(65.0)]),
    spool_to_motor_gearing_factor=Dec(12.75),
    mech_adv=np.array([Dec(2.0), Dec(2.0), Dec(2.0), Dec(2.0)]),
    number_of_lines_per_spool=np.array([Dec(1.0), Dec(1.0), Dec(1.0), Dec(1.0)]),
):
    """What motor positions (in degrees) motors would be at
    """

    # Assure np.array type
    spool_r_in_origin = np.array(spool_r_in_origin)
    mech_adv = np.array(mech_adv)
    number_of_lines_per_spool = np.array(number_of_lines_per_spool)

    spool_r_in_origin_sq = spool_r_in_origin * spool_r_in_origin

    # Buildup per line times lines. Minus sign because more line in air means less line on spool
    k2 = -Dec(1.0) * mech_adv * number_of_lines_per_spool * spool_buildup_factor

    # we now want to use degrees instead of steps as unit of rotation
    # so setting 360 where steps per motor rotation is in firmware buildup compensation algorithms
    degrees_per_unit_times_r = (
        spool_to_motor_gearing_factor * mech_adv * Dec(360.0)
    ) / (Dec(2.0) * Dec(np.pi))
    k0 = Dec(2.0) * degrees_per_unit_times_r / k2

    line_lengths_origin = np.linalg.norm(anchors, 2, 1)

    relative_line_lengths = samples_relative_to_origin_no_fuzz(anchors, pos)
    motor_positions = k0 * (
        np.sqrt(spool_r_in_origin_sq + relative_line_lengths * k2)
        - spool_r_in_origin
    )

    return motor_positions


def motor_pos_samples_to_line_length_with_buildup_compensation(
    motor_samps,
    spool_buildup_factor=Dec(0.008),  # Qualified first guess for 0.5 mm line
    spool_r=np.array([Dec(65.0), Dec(65.0), Dec(65.0), Dec(65.0)]),
    spool_to_motor_gearing_factor=Dec(12.75),  # HP4 default (255/20)
    mech_adv=np.array([Dec(2.0), Dec(2.0), Dec(2.0), Dec(2.0)]),  # HP4 default
    number_of_lines_per_spool=np.array([Dec(1.0), Dec(1.0), Dec(1.0), Dec(1.0)]),  # HP4 default
):
    # Buildup per line times lines. Minus sign because more line in air means less line on spool
    c1 = -mech_adv * number_of_lines_per_spool * spool_buildup_factor

    # we now want to use degrees instead of steps as unit of rotation
    # so setting 360 where steps per motor rotation is in firmware buildup compensation algorithms
    degrees_per_unit_times_r = (
        spool_to_motor_gearing_factor * mech_adv * Dec(360.0)
    ) / (Dec(2.0) * Dec(np.pi))
    k0 = Dec(2.0) * degrees_per_unit_times_r / c1

    return (((motor_samps / k0) + spool_r) ** Dec(2.0) - spool_r * spool_r) / c1


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
    return np.sum(
        pow((samples_relative_to_origin_no_fuzz(anchors, pos) - samp), 2)
    )


def cost_for_pos_samp(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r
):
    """
    For all samples sum
    |Sample value if anchor position A and cartesian position x were guessed   - actual sample|

    |sqrt((A_ax-x_i)^2 + (A_ay-y_i)^2 + (A_az-z_i)^2) - sqrt(A_ax^2 + A_ay^2 + A_az^2) - motor_pos_to_samp(t_ia)| +
    |sqrt((A_bx-x_i)^2 + (A_by-y_i)^2 + (A_bz-z_i)^2) - sqrt(A_bx^2 + A_by^2 + A_bz^2) - motor_pos_to_samp(t_ib)| +
    |sqrt((A_cx-x_i)^2 + (A_cy-y_i)^2 + (A_cz-z_i)^2) - sqrt(A_cx^2 + A_cy^2 + A_cz^2) - motor_pos_to_samp(t_ic)| +
    |sqrt((A_dx-x_i)^2 + (A_dy-y_i)^2 + (A_dz-z_i)^2) - sqrt(A_dx^2 + A_dy^2 + A_dz^2) - motor_pos_to_samp(t_id)|
    """

    return np.sum(
        np.abs(
            samples_relative_to_origin_no_fuzz(anchors, pos)
            - motor_pos_samples_to_line_length_with_buildup_compensation(
                motor_pos_samp, spool_buildup_factor, spool_r
            )
        )
    )


def cost_sq_for_pos_samp(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r
):
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


def cost_sqsq_for_pos_samp(
    anchors, pos, motor_pos_samp, spool_buildup_factor, spool_r
):
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
    #anchors = np.array(np.zeros((4, 3)))
    anchors = np.array([[Dec(0.0),Dec(anchorsvec[0]), Dec(anchorsvec[1])],
                        [Dec(anchorsvec[2]), Dec(anchorsvec[3]), Dec(anchorsvec[4])],
                        [Dec(anchorsvec[5]), Dec(anchorsvec[6]), Dec(anchorsvec[7])],
                        [Dec(0.0), Dec(0.0), Dec(anchorsvec[8])],
                        ])
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
    return np.append(
        np.append(l[0:params_anch], l[params_anch : params_anch + 3 * num]),
        l[-params_buildup:],
    )


def solve(motor_pos_samp, xyz_of_samp, method, cx_is_positive=False):
    """Find reasonable positions and anchors given a set of samples.
    """

    print(method)

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    number_of_params_pos = 3 * (u - ux)

    # Changing the shape of the cost function
    # to fit better with our algorithms

    # Make-all-answers ~1 scaling
    # L-BFGS favors this
    #overall_scaler = 1.0
    #anch_scale = 1500.0
    #pos_scale = 500.0
    #sbf_scale = 0.008
    #sr_scale = 65.0


    # SLSQP needs this kind of scaling
    # DifferentialEvolutionSolver also likes it

    # These scaling values work great if there are no xyz_of_samp
    #overall_scaler = 20.0
    #anch_scale = Dec((1.0/overall_scaler)*14.0)
    #pos_scale  = Dec((1.0/overall_scaler)*4.0)
    #sbf_scale  = Dec((1.0/overall_scaler)*0.010)
    #sr_scale   = Dec((1.0/overall_scaler)*0.005)

    # These scaling values work well if there are
    # some known xyz_of_samp
    overall_scaler = 2.0
    anch_scale = 0.2
    pos_scale  = 1.0
    sbf_scale  = 0.0010
    sr_scale   = 0.20

    def scale_back_solution(sol):
        sol[0:params_anch] *= float(anch_scale)
        sol[params_anch:-params_buildup] *= float(pos_scale)
        sol[-params_buildup] *= float(sbf_scale)
        sol[-params_buildup + 1 :] *= float(sr_scale)
        return sol

    def costx(_cost, posvec, anchvec, spool_buildup_factor, spool_r, u):
        """Identical to cost, except the shape of inputs and capture of samp, xyz_of_samp, ux, and u

        Parameters
        ----------
        x : [A_ay A_az A_bx A_by A_bz A_cx A_cy A_cz A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """
        #spool_r = np.array(spool_r)

        if(not type(posvec[0]) == decimal.Decimal):
            posvec = np.array([Dec(pos) for pos in posvec])
        if(not type(anchvec[0]) == decimal.Decimal):
            anchvec = np.array([Dec(anch) for anch in anchvec])
        if(not type(spool_buildup_factor) == decimal.Decimal):
            spool_buildup_factor = Dec(spool_buildup_factor)
        if(not type(spool_r[0]) == decimal.Decimal):
            spool_r = np.array([Dec(r) for r in spool_r])

        anchors = anchorsvec2matrix(anchvec)
        # Adds in known positions back in before calculating the cost
        pos = np.zeros((u, 3))
        if np.size(xyz_of_samp) != 0:
            pos[0:ux] = xyz_of_samp * pos_scale
        pos[ux:] = np.reshape(posvec, (u - ux, 3))
        return _cost(
            anchors * anch_scale,
            pos * pos_scale,
            motor_pos_samp[:u],
            spool_buildup_factor * sbf_scale,
            spool_r * sr_scale,
        )

    l_long = 4000.0
    l_short = 3000.0
    data_z_min = -100.0
    # Limits of anchor positions:
    #     |ANCHOR_XY|    < 4000
    #      ANCHOR_B_X    > 0
    #      ANCHOR_C_X    < 0
    #     |ANCHOR_ABC_Z| < 3000
    # 0 <  ANCHOR_D_Z    < 4000
    # Limits of data collection volume:
    #         |x| < 3000
    #         |y| < 3000
    # -20.0 <  z  < 3400.0
    # Define bounds
    lb = np.array(
      [
          -l_long / float(anch_scale),  # A_ay > -4000.0
           -300.0 / float(anch_scale),  # A_az > -300.0
            300.0 / float(anch_scale),  # A_bx > 300
            300.0 / float(anch_scale),  # A_by > 300
           -300.0 / float(anch_scale),  # A_bz > -300.0
          -l_long / float(anch_scale),  # A_cx > -4000
            300.0 / float(anch_scale),  # A_cy > 300
           -300.0 / float(anch_scale),  # A_cz > -300.0
           1000.0 / float(anch_scale),  # A_dz > 1000
      ]
      + [-l_short / float(pos_scale), -l_short / float(pos_scale), data_z_min / float(pos_scale)] * (u - ux)
      + [0.00005 / float(sbf_scale), 64.0 / float(sr_scale), 64.0 / float(sr_scale), 64.0 / float(sr_scale), 64.0 / float(sr_scale)]
    )
    ub = np.array(
      [
           500.0 / float(anch_scale),  # A_ay < 500
           200.0 / float(anch_scale),  # A_az < 200
          l_long / float(anch_scale),  # A_bx < 4000
          l_long / float(anch_scale),  # A_by < 4000
           200.0 / float(anch_scale),  # A_bz < 200
          -300.0 / float(anch_scale),  # A_cx < -300
          l_long / float(anch_scale),  # A_cy < 4000.0
           200.0 / float(anch_scale),  # A_cz < 200
          l_long / float(anch_scale),  # A_dz < 4000.0
      ]
      + [l_short / float(pos_scale), l_short / float(pos_scale), 2.0 * l_short / float(pos_scale)] * (u - ux)
      + [0.1 / float(sbf_scale), 67.0 / float(sr_scale), 67.0 / float(sr_scale), 67.0 / float(sr_scale), 67.0 / float(sr_scale)]
    )
    #lb = (
    #    [
    #        -2000.0 / anch_scale,  # A_ay > -2000.0
    #         -200.0 / anch_scale,  # A_az > -200.0
    #         1000.0 / anch_scale,  # A_bx > 1000
    #          900.0 / anch_scale,  # A_by > 900
    #         -200.0 / anch_scale,  # A_bz > -200.0
    #        -2000.0 / anch_scale,  # A_cx > -2000
    #          700.0 / anch_scale,  # A_cy > 700
    #         -200.0 / anch_scale,  # A_cz > -200.0
    #         2300.0 / anch_scale,  # A_dz > 2300
    #    ]
    #    + [ -1000.0 / pos_scale,
    #        -1000.0 / pos_scale,
    #          -30.0 / pos_scale ] * (u - ux)
    #    + [0.005 / sbf_scale, 64.6 / sr_scale, 64.6 / sr_scale, 64.6 / sr_scale, 64.6 / sr_scale]
    #)
    #ub = (
    #    [
    #        -1600.0 / anch_scale,  # A_ay < -1600
    #            0.0 / anch_scale,  # A_az < 0
    #         1600.0 / anch_scale,  # A_bx < 1600
    #         1500.0 / anch_scale,  # A_by < 1500
    #            0.0 / anch_scale,  # A_bz < 200
    #        -1300.0 / anch_scale,  # A_cx < -1300
    #         1000.0 / anch_scale,  # A_cy < 1000.0
    #            0.0 / anch_scale,  # A_cz < 0
    #         2700.0 / anch_scale,  # A_dz < 2700.0
    #    ]
    #    + [1000.0 / pos_scale,
    #       1000.0 / pos_scale,
    #       2000.0 / pos_scale] * (u - ux)
    #    + [0.01 / sbf_scale, 66.0 / sr_scale, 66.0 / sr_scale, 66.0 / sr_scale, 66.0 / sr_scale]
    #)
    # lb = (
    #    [
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #        -999999.0,
    #    ]
    #    + [-999999.0, -999999.0, -999999.0] * (u - ux)
    #    + [0.0001 / sbf_scale, 60.5 / sr_scale, 60.5 / sr_scale, 60.5 / sr_scale, 60.5 / sr_scale]
    # )
    # ub = (
    #    [
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #        999999.9,
    #    ]
    #    + [999999.0, 999999.0, 999999.0] * (u - ux)
    #    + [100 / sbf_scale, 670 / sr_scale, 670 / sr_scale, 670 / sr_scale, 670 / sr_scale]
    # )

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
        + [0.006 / float(sbf_scale), 64.5 / float(sr_scale), 64.5 / float(sr_scale), 64.5 / float(sr_scale), 64.5 / float(sr_scale)]
    )

    def constraints(x):
        for samp_num in range(u - ux):
            # If A motor wound in line
            # Then the position of measurment has negative y coordinate
            if motor_pos_samp[samp_num][A] < 0.0:
                x[params_anch + samp_num * 3 + Y] = -np.abs(
                    x[params_anch + samp_num * 3 + Y]
                )
            # If both B and C wound in line
            # Then position of measurement has positive Y coordinate
            elif (
                motor_pos_samp[samp_num][B] < 0.0
                and motor_pos_samp[samp_num][C] < 0.0
            ):
                x[params_anch + samp_num * 3 + Y] = np.abs(
                    x[params_anch + samp_num * 3 + Y]
                )
            # If both A and B wound in line
            # Then pos of meas has positive X coord
            if (
                motor_pos_samp[samp_num][A] < 0.0
                and motor_pos_samp[samp_num][B] < 0.0
            ):
                x[params_anch + samp_num * 3 + X] = np.abs(
                    x[params_anch + samp_num * 3 + X]
                )
            # If both A and C wound in line
            # Then pos of meas has negative X coord
            elif (
                motor_pos_samp[samp_num][A] < 0.0
                and motor_pos_samp[samp_num][C] < 0.0
            ):
                x[params_anch + samp_num * 3 + X] = -np.abs(
                    x[params_anch + samp_num * 3 + X]
                )
            # No sample was made with more positive x coordinate than B anchor
            if (x[params_anch + samp_num * 3 + X] > x[2]):
                x[params_anch + samp_num * 3 + X] = x[2]
            # No sample was made with more negative x coordinate than C anchor
            elif (x[params_anch + samp_num * 3 + X] < x[5]):
                x[params_anch + samp_num * 3 + X] = x[5]
            # No sample was made with more negative y coordinate than A anchor
            if (x[params_anch + samp_num * 3 + Y] < x[0]):
                x[params_anch + samp_num * 3 + Y] = x[0]
            # No sample was made with more positive y coordinate than B and C anchor
            elif (x[params_anch + samp_num * 3 + Y] > x[3] and x[params_anch + samp_num * 3 + Y] > x[6]):
                x[params_anch + samp_num * 3 + Y] = max(x[3], x[6])
            # No sample was made with more positive z coordinate than D anchor
            if (x[params_anch + samp_num * 3 + Z] > x[8]):
                x[params_anch + samp_num * 3 + Z] = x[8]

        #x[9] = -1000.0/pos_scale
        #x[10] = -1000.0/pos_scale
        #x[11] = 1000.0/pos_scale

        return x

    if method == "differentialEvolutionSolver":
        print("Hit Ctrl+C to stop solver. Then type exit to get the current solution.")
        from mystic.solvers import DifferentialEvolutionSolver2
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, ChangeOverGeneration, And, Or
        from mystic.strategy import Best1Exp, Best1Bin

        stop = Or(VTR(1e-12), ChangeOverGeneration(0.0000001, 5000))
        ndim = number_of_params_pos + params_anch + params_buildup
        npop = 3
        stepmon = VerboseMonitor(100)
        solver = DifferentialEvolutionSolver2(ndim, npop)

        solver.SetRandomInitialPoints(lb, ub)
        solver.SetStrictRanges(lb, ub)
        solver.SetConstraints(constraints)
        solver.SetGenerationMonitor(stepmon)
        solver.enable_signal_handler()  # Handle Ctrl+C gracefully. Be restartable
        solver.Solve(
            lambda x: overall_scaler*float(costx(
                cost_sq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u,
            )),
            termination=stop,
            strategy=Best1Bin,
        )

        # use monitor to retrieve results information
        iterations = len(stepmon)
        cost = stepmon.y[-1]
        print("Generation %d has best Chi-Squared: %f" % (iterations, cost))
        best_x = solver.Solution()
        if(not type(best_x[0]) == float):
            best_x = np.array([float(pos) for pos in best_x])
        return scale_back_solution(best_x)

    elif method == "BuckShot":
        print("You can not interrupt this solver without losing the solution.")
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
        archive = False  # save an archive
        ndim = number_of_params_pos + params_anch + params_buildup
        from mystic.solvers import BuckshotSolver, LatticeSolver

        # the local solvers
        from mystic.solvers import PowellDirectionalSolver

        sprayer = BuckshotSolver
        seeker = PowellDirectionalSolver(ndim)
        seeker.SetGenerationMonitor(VerboseMonitor(5))
        #seeker.SetConstraints(constraints)
        seeker.SetTermination(Or(VTR(1e-4), COG(0.01, 20)))
        #seeker.SetEvaluationLimits(evaluations=3200000, generations=100000)
        seeker.SetStrictRanges(lb, ub)
        #seeker.enable_signal_handler()  # Handle Ctrl+C. Be restartable
        npts = 10  # number of solvers
        _map = Pool().map
        retry = 1  # max consectutive iteration retries without a cache 'miss'
        tol = 8  # rounding precision
        mem = 1  # cache rounding precision
        from mystic.search import Searcher

        searcher = Searcher(npts, retry, tol, mem, _map, None, sprayer, seeker)
        searcher.Verbose(True)
        searcher.UseTrajectories(True)
        # searcher.Reset(None, inv=False)
        searcher.Search(
            lambda x: float(costx(
                cost_sq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u,
            )),
            bounds=list(zip(lb, ub)),
            stop=stop,
            monitor=VerboseMonitor(1),
        )  # ,
        # constraints=constraints)
        searcher._summarize()
        print(searcher.Minima())
        best_x = np.array(min(searcher.Minima(), key=searcher.Minima().get))
        if(not type(best_x[0]) == float):
            best_x = np.array([float(pos) for pos in best_x])
        return scale_back_solution(best_x)

    elif method == "PowellDirectionalSolver":
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG
        from mystic.monitors import VerboseMonitor
        from mystic.termination import VTR, And, Or

        ndim = number_of_params_pos + params_anch + params_buildup
        killer = GracefulKiller()
        best_cost = 9999999999999.9
        i = 0
        while(True):
            i = i + 1
            print("Try: %d/30. Hit Ctrl+C and wait a bit to stop solver." % i)
            if killer.kill_now or i == 30:
                break
            solver = PowellDirectionalSolver(ndim)
            solver.SetRandomInitialPoints(lb, ub)
            solver.SetEvaluationLimits(evaluations=3200000, generations=100000)
            solver.SetTermination(Or(VTR(1e-25), COG(1e-10, 10)))
            solver.SetStrictRanges(lb, ub)
            solver.SetConstraints(constraints)
            solver.SetGenerationMonitor(VerboseMonitor(5))
            solver.Solve(
                lambda x: overall_scaler*float(costx(
                    cost_sq_for_pos_samp,
                    x[params_anch:-params_buildup],
                    x[0:params_anch],
                    x[-params_buildup],
                    x[-params_buildup + 1 :],
                    u,
                ))
            )
            if solver.bestEnergy < best_cost:
                print("New best x: ")
                print("With cost: ", solver.bestEnergy)
                best_cost = solver.bestEnergy
                best_x = np.array(solver.bestSolution)
            if solver.bestEnergy < 0.0001:
                print("Found a perfect solution!")
                break

        if(not type(best_x[0]) == float):
            best_x = np.array([float(pos) for pos in best_x])
        return scale_back_solution(best_x)

    elif method == "SLSQP":
        # Create a random guess
        best_cost = 999999.9
        best_x = x_guess
        killer = GracefulKiller()

        for i in range(30):
            print("Try: %d/30. Hit Ctrl+C and wait a bit to stop solver." % i)
            if killer.kill_now:
                break
            random_guess = np.array([ b[0] + (b[1] - b[0])*np.random.rand() for b in list(zip(lb, ub)) ])
            sol = scipy.optimize.minimize(
                lambda x: overall_scaler*float(costx(
                    cost_sq_for_pos_samp,
                    x[params_anch:-params_buildup],
                    x[0:params_anch],
                    x[-params_buildup],
                    x[-params_buildup + 1 :],
                    u,
                )),
                random_guess,
                method="SLSQP",
                bounds=list(zip(lb, ub)),
                #constraints=[{'type': 'eq', 'fun': constraints}], # This doesn't seem to work?
                options={"disp": True, "ftol": 1e-20, "maxiter": 500},
            )
            if sol.fun < best_cost:
                print("New best x: ")
                print("With cost: ", sol.fun)
                best_cost = sol.fun
                best_x = sol.x

        #lb[0:-params_buildup] = best_x[0:-params_buildup] - 23.0
        #ub[0:-params_buildup] = best_x[0:-params_buildup] + 23.0
        #lb[-params_buildup] = best_x[-params_buildup] - 0.001
        #ub[-params_buildup] = best_x[-params_buildup] + 0.001
        #lb[-params_buildup + 1 :] = best_x[-params_buildup + 1 :] - 0.1
        #ub[-params_buildup + 1 :] = best_x[-params_buildup + 1 :] + 0.1
        #from mystic.solvers import DifferentialEvolutionSolver2
        #from mystic.monitors import VerboseMonitor
        #from mystic.termination import VTR, ChangeOverGeneration, And, Or
        #from mystic.strategy import Best1Exp, Best1Bin
        #stop = Or(VTR(1e-12), ChangeOverGeneration(1e-10, 500))
        #ndim = number_of_params_pos + params_anch + params_buildup
        #npop = 2
        #stepmon = VerboseMonitor(100)
        #solver = DifferentialEvolutionSolver2(ndim, npop)
        #solver.SetRandomInitialPoints(lb, ub)
        #solver.SetStrictRanges(lb, ub)
        #solver.SetConstraints(constraints)
        #solver.SetGenerationMonitor(stepmon)
        #solver.enable_signal_handler()  # Handle Ctrl+C gracefully. Be restartable
        #solver.Solve(
        #    lambda x: costx(
        #        cost_sq_for_pos_samp,
        #        x[params_anch:-params_buildup],
        #        x[0:params_anch],
        #        x[-params_buildup],
        #        x[-params_buildup + 1 :],
        #        u,
        #    ),
        #    termination=stop,
        #    strategy=Best1Bin,
        #)
        #iterations = len(stepmon)
        #cost = stepmon.y[-1]
        #print("Generation %d has best Chi-Squared: %f" % (iterations, cost))
        #best_x = solver.Solution()

        if(not type(best_x[0]) == float):
            best_x = np.array([float(pos) for pos in best_x])
        return scale_back_solution(np.array(best_x))

    elif method == "L-BFGS-B":
        print("You can not interrupt this solver without losing the solution.")
        best_x = scipy.optimize.minimize(
            lambda x: overall_scaler*float(costx(
                cost_sq_for_pos_samp,
                x[params_anch:-params_buildup],
                x[0:params_anch],
                x[-params_buildup],
                x[-params_buildup + 1 :],
                u,
            )),
            x_guess,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"disp": True, "ftol": 1e-12, "gtol": 1e-12, "maxiter": 50000, "maxfun": 1000000},
        ).x
        if(not type(best_x[0]) == float):
            best_x = np.array([float(pos) for pos in best_x])
        return scale_back_solution(best_x)

    else:
        print("Method %s is not supported!" % method)
        sys.exit(1)


def print_copypasteable(anch, spool_buildup_factor, spool_r):
    print(
        "\nM669 A0.0:%.2f:%.2f B%.2f:%.2f:%.2f C%.2f:%.2f:%.2f D%.2f\nM666 Q%.6f R%.3f:%.3f:%.3f:%.3f"
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
            spool_buildup_factor,
            spool_r[A],
            spool_r[B],
            spool_r[C],
            spool_r[D],
        )
    )


def print_anch_err(sol_anch, anchors):
    print("\nErr_A_Y: %9.3f" % (float(sol_anch[A, Y]) - (anchors[A, Y])))
    print("Err_A_Z: %9.3f" %   (float(sol_anch[A, Z]) - (anchors[A, Z])))
    print("Err_B_X: %9.3f" %   (float(sol_anch[B, X]) - (anchors[B, X])))
    print("Err_B_Y: %9.3f" %   (float(sol_anch[B, Y]) - (anchors[B, Y])))
    print("Err_B_Z: %9.3f" %   (float(sol_anch[B, Z]) - (anchors[B, Z])))
    print("Err_C_X: %9.3f" %   (float(sol_anch[C, X]) - (anchors[C, X])))
    print("Err_C_Y: %9.3f" %   (float(sol_anch[C, Y]) - (anchors[C, Y])))
    print("Err_C_Z: %9.3f" %   (float(sol_anch[C, Z]) - (anchors[C, Z])))
    print("Err_D_Z: %9.3f" %   (float(sol_anch[D, Z]) - (anchors[D, Z])))


class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super(Store_as_array, self).__call__(
            parser, namespace, values, option_string
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Figure out where Hangprinter anchors are by looking at line difference samples. Practical tips: Your best bet is to collect many samples (> 20), know the xyz of at least a couple of them, and use PowellDirectionalSolver (method 1) to find your parameters."
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
        help="Available methods are L-BFGS-B (0), PowellDirectionalSolver (1), SLSQP (2), differentialEvolutionSolver (3), BuckShot (4), and all (5). Try 0 first, then 1, and so on.",
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
    #parser.add_argument(
    #    "-p",
    #    "--precision",
    #    help="Specify that you want better than float64 precision. This makes computations slow and shouldn't be necessary.",
    #    action="store_true",
    #)
    args = vars(parser.parse_args())

    if args["method"] == "0":
        args["method"] = "L-BFGS-B"
    if args["method"] == "1":
        args["method"] = "PowellDirectionalSolver"
    if args["method"] == "2":
        args["method"] = "SLSQP"
    if args["method"] == "3":
        args["method"] = "differentialEvolutionSolver"
    if args["method"] == "4":
        args["method"] = "BuckShot"
    if args["method"] == "5":
        args["method"] = "all"
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
                [0.0, 0.0, 2170.0],

                [-210.5, -297.0, 0.0],
                [   0.0, -297.0, 0.0],
                [ 210.5, -284.0, 0.0],
                [ 210.5,    0.0, 0.0],
                [ 210.5,  297.0, 0.0],
                [   0.0,  297.0, 0.0],
                [-210.5,  287.0, 0.0],
                [-210.5,    0.0, 0.0],
              ]
            )

                #[-1000.0, -1000.0, 1000.0],
                #[-1000.0, -1000.0, 2000.0],
                #[-1000.0, 0.0, 0.0],
                #[-1000.0, 0.0, 1000.0],
                #[-1000.0, 0.0, 2000.0],
                #[-1000.0, 1000.0, 0.0],
                #[-1000.0, 1000.0, 1000.0],
                #[-1000.0, 1000.0, 2000.0],
                #[0.0, -1000.0, 0.0],
                #[0.0, -1000.0, 1000.0],
                #[0.0, -1000.0, 2000.0],
                #[-1000.0, -1000.0, 0.0],
                #[0.0, 0.0, 1000.0],
                #[0.0, 0.0, 2000.0],
                #[0.0, 1000.0, 0.0],
                #[0.0, 1000.0, 1000.0],
                #[0.0, 1000.0, 2000.0],
                #[1000.0, -1000.0, 0.0],
                #[1000.0, -1000.0, 1000.0],
                #[1000.0, -1000.0, 2000.0],
                #[1000.0, 0.0, 0.0],
                #[1000.0, 0.0, 1000.0],
                #[1000.0, 0.0, 2000.0],
                #[1000.0, 1000.0, 0.0],
                #[1000.0, 1000.0, 1000.0],
                #[1000.0, 1000.0, 2000.0],

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
        #    ]
        #)

    motor_pos_samp = args["sample_data"]
    if np.size(motor_pos_samp) != 0:
        if np.size(motor_pos_samp) % 4 != 0:
            print("Please specify motor positions (angles) of sampling points.")
            print(
                "You specified %d numbers after your -s/--sample_data option, which is not a multiple of 4 number of numbers."
            )
            sys.exit(1)
        motor_pos_samp = motor_pos_samp.reshape(
            (int(np.size(motor_pos_samp) / 4), 4)
        )
    else:
        # You might want to manually replace this with your collected data
        motor_pos_samp = np.array(
            [
                [26538.55, 25683.74, 27303.89, -48799.39, ],

                [-6250.32, --7945.95,  -279.15, --618.21, ],
                [-6558.04, --4773.87,  3396.71, --435.61, ],
                [-5914.38, --1702.78,  7179.45, --620.31, ],
                [283.49,   - 3286.65,  4227.62, --207.36, ],
                [6885.27,  - 7966.05,  1954.32, --591.93, ],
                [6597.56,  - 4327.81, -2498.74, --360.66, ],
                [6655.95,  -  319.66, -6891.31, --539.56, ],
                [274.94,   --3473.22, -4111.23, --181.35, ],

                [2185.65, -1565.88, 901.40, -3497.24],
                [13289.12, -16251.47, 9180.44, 35.29],
                [19102.38, -19737.29, 10106.74, 34.05],
                [15736.09, -12568.62, 7477.67, -6882.47],
                [11833.54, -3475.36, 7092.05, -15260.72],
                [10453.83, 8991.93, 7252.55, -24340.97],
                [12342.05, 286.95, 3259.66, -16596.90],
                [12341.57, -4064.08, -840.31, -8899.52],
                [14965.81, -4064.08, -6865.62, -2128.97],
                [16145.44, 5181.89, -15089.08, -2098.95],
                [11140.04, 5179.96, -9597.86, -7383.00],
                [5789.15, 11239.36, -4038.70, -13199.07],
                [-7015.04, 14092.20, 5979.81, -10498.90],
                [-10983.31, 10043.28, 6839.74, -2878.31],
                [-7130.69, 10049.42, 6989.61, -9051.27],
                [-3162.74, 5991.19, 2751.78, -7802.40],
                [2492.55, 45.91, 361.06, -6290.42],
                [19829.39, -5890.33, 7533.87, -15096.64],
                [14754.43, -4818.78, -3363.34, -6367.55],
                [14912.98, -4818.78, -3387.19, -6370.97],
                [10131.60, -8274.28, 315.28, -2108.26],
                [7768.62, -2569.03, -4573.51, -2099.56],
                [3141.41, 4041.86, -6660.20, -2098.82],
                [-2441.61, 10314.15, -4563.80, -2843.30],
                [8277.75, 8616.45, -8343.78, -9353.54],
                [16481.21, 3228.76, -2106.80, -15410.39],
                [12644.83, -1824.00, 4526.38, -15778.54],
                [8674.19, 1957.49, 2777.83, -15906.66],
                [6238.78, 5579.11, 25.78, -14755.41],
                [-5335.10, 8043.92, 4345.76, -8144.45],
                [-3890.16, 3188.10, 1555.36, -149.26],
                [-1394.91, 3188.10, 896.54, -5248.52],
                [2026.63, 2691.43, -3303.89, -4314.29],
                [3962.08, 504.26, -2633.29, -5188.25],
                [5130.26, -1638.34, -3161.16, -2275.37],
                [3713.22, -1638.34, -361.39, -4574.90],
                [2859.11, -2159.69, 1389.60, -4575.80],
                [1492.21, -664.47, -102.30, -2519.54],
            ]
        )
        ### Simulated data w no fuzz ###
        #motor_pos_samp = np.array(
        #  [
              # [    -0.0  ,     -0.0  ,     -0.0  ,     -0.0  ],
              # [ -1559.52291448,  35112.9378728 ,  11052.95417008, -8872.59579759],
              # [ 15947.8692276 ,  44648.94325866,  25081.52956038, -20058.22062916],
              # [  6380.4256863 ,  19670.67758147, -16066.84268992, 4584.4799315 ],
              # [ 12033.38507565,  23821.84115755,  -5934.60224362, -15049.76641769],
              # [ 25842.55572463,  34790.08867383,  13138.38431711, -28972.38243599],
              # [ 26651.02404208,  14774.89738023, -26368.60964516, 8830.62448657],
              # [ 30623.76965144,  19265.11350679, -11595.88498992, -8872.59579759],
              # [ 41211.31825491,  30931.85872451,   9866.13078227, -20058.22062916],
              # [-22455.21796312,  16219.84593104,  15070.81118868, 4584.4799315 ],
              # [ -9771.80893521,  20604.52665964,  19861.33450594, -15049.76641769],
              # [ 10871.05742706,  32057.52863075,  32119.33188027, -28972.38243599],
               #[ -9961.99651941,  31610.40213247,   5267.87521722, 8830.62448657],
               #[  6498.39340409,   5934.96082808,   6498.39340409, -22456.33469473],
               #[ 21632.76309175,  20205.16553181,  21632.76309175, -44870.3487401 ],
               #[ 22498.6306856 ,  -7698.49546393,  -4589.64508816, 4584.4799315 ],
               #[ 26731.30557098,   -604.05476093,   2676.88882131, -15049.76641769],
               #[ 37871.408152  ,  15355.98761036,  18863.00930889, -28972.38243599],
               #[ -9961.99651941,   6924.49753218,  31248.11861017, 8830.62448657],
               #[ -1559.52291448,  12086.06829491,  34967.31118595, -8872.59579759],
               #[ 15947.8692276 ,  25035.97366952,  44996.32727706, -20058.22062916],
               #[  6380.4256863 , -14551.95666055,  20602.77158704, 4584.4799315 ],
               #[ 12033.38507565,  -6018.57873363,  24965.46565502, -15049.76641769],
               #[ 25842.55572463,  11621.84687368,  36374.37196585, -28972.38243599],
               #[ 26651.02404208, -30082.16835223,  17780.73559022, 8830.62448657],
               #[ 30623.76965144, -15504.96783497,  22351.99273756, -8872.59579759],
               #[ 41211.31825491,   5888.69586035,  34181.57523639, -20058.22062916],

              # Fuzzy ones (100)
               #[ 27765.090805702617  ,   9542.153896217365  , -121.27495398726754, -19224.630476231792  ],
               #[ -7849.704802767238  ,  32240.435045207076  , 9260.209694379788  ,  -3861.7826603984527 ],
               #[  3059.971445530039  ,  39456.73152058913   , 15951.644149703327  , -11189.1259472278    ],
               #[ 16634.859388065364  ,  45949.39891587497   , 24625.37189445659   , -18428.56064192675   ],
               #[  1927.1095774495243 ,  24309.131416949003  , -10313.727832261467  ,   4381.924525060732  ],
               #[  3885.34781686248   ,  23535.505846773343  , -6044.364387930406  ,  -8852.419589568106  ],
               #[ 10521.922875251319  ,  28556.690312204057  , 5298.544008343454  , -21729.20211362577   ],
               #[ 21347.656324639916  ,  38379.03049118139   , 16022.330346267825  , -26210.899017511387  ],
               #[ 13395.896337404982  ,  16366.230960938496  , -22425.914098829457  ,   3438.1023111175496 ],
               #[ 16062.251961564463  ,  19255.104310353898  , -14218.332551172925  ,  -9995.168304740819  ],
               #[ 20312.172602620965  ,  23978.58014395706   , -4075.172490465647  , -19181.885861862815  ],
               #[ 29347.683663947537  ,  32940.723710201135  , 11700.718132275426  , -28888.770561439353  ],
               #[ 25927.408406153012  ,  13879.914079178425  , -25425.63431815051   ,   6526.159319879889  ],
               #[ 29993.845445456813  ,  16212.07037499737   , -18650.325373231168  ,  -1366.5651518894972 ],
               #[ 35587.88319033556   ,  23959.39790919933   , -3592.5452406445766 , -12822.834994475     ],
               #[ 42465.46143870988   ,  31540.27109457229   , 8532.381240941779  , -17246.418818771137  ],
               #[-21737.834618918238  ,  22679.454309210923  , 12135.133486943303  ,   4841.770335439292  ],
               #[-12668.065466676766  ,  22423.183763105517  , 11172.808227726164  ,  -8947.473873888095  ],
               #[ -3086.364858722371  ,  25827.662473938817  , 17956.9288895432    , -19957.42569048575   ],
               #[ 12979.967916004483  ,  36564.1294255325    , 30409.074343378477  , -28380.71987362472   ],
               #[ -6749.7304349296965 ,   9071.185683251666  , -60.57972611673642,   2678.3217493843695 ],
               #[ -4552.209587590612  ,  14532.219334960811  , 2540.1141968853804 , -12489.547233588028  ],
               #[  8436.277633838692  ,  19442.181674280248  , 9235.557068796263  , -28833.473783308233  ],
               #[ 17996.096083494325  ,  25982.987180022796  , 20794.065108186944  , -40891.18873136719   ],
               #[  8662.950703583967  ,   1507.627312847311  , -9171.95184262119   ,   2372.090240845315  ],
               #[  9831.55704404989   ,   8459.225016544275  , -5781.679101870769  , -15114.92110946377   ],
               #[ 18066.206258903367  ,  13080.726938284568  , 4646.420088910607  , -28755.27371136958   ],
               #[ 25084.157314557153  ,  20889.535638861318  , 15002.423310376409  , -39377.458006073604  ],
               #[ 21898.01919495756   ,   1594.990139963376  , -13917.176803168708  ,   4851.903903503193  ],
               #[ 23654.393508705696  ,   2055.772495128701  , -7320.380896868105  ,  -8835.996138735596  ],
               #[ -8244.977976085165  ,  31339.820712938974  , 3495.1802200650827 ,   7702.863129947008  ],
               #[ 37455.60445187385   ,  21449.84086019472   , 14690.997304552258  , -28974.014246832594  ],
               #[-18490.97476292576   ,  10177.370851454192  , 18825.364855657284  ,   4174.818039348462  ],
               #[-14488.500659597257  ,  12335.880073917098  , 21500.452234443765  ,  -6893.290698344427  ],
               #[ -1514.7357188430442 ,  21933.466765324378  , 28153.204265273962  , -20560.305563263515  ],
               #[ 12811.88412200408   ,  27133.535551087076  , 36059.93804680912   , -29110.21796887811   ],
               #[ -6268.094547620793  ,    212.1210295292896 , 8754.441792859758  ,    442.35700197011687],
               #[ -2371.2646054673    ,   1356.2709623024425 , 9918.076597842572  , -12052.39693517536   ],
               #[  6728.979265543948  ,   7156.084935686261  , 18227.15393351809   , -26513.57484132787   ],
               #[ 17155.61872184854   ,  21776.817854268415  , 28435.222182094796  , -39802.62192450442   ],
               #[  7358.520704524515  ,  -8660.746179409562  , 2578.799568550206  ,   -207.13664579754274],
               #[  8259.983321978394  ,  -5443.970552922985  , 5286.420514748826  , -12402.663132939108  ],
               #[ 18012.970599915396  ,   1986.683313294882  , 16274.102777999085  , -28042.81186149403   ],
               #[ 26406.415645849473  ,  12853.442664551341  , 23910.834385450227  , -38786.72950855659   ],
               #[ 23552.295508386935  , -17282.999298394672  , 4973.058037349916  ,   6186.922407945081  ],
               #[ 22569.396662060215  , -12719.265268874213  , 5728.314412080501  ,  -7084.363683388169  ],
               #[ 29357.566068204018  ,   -344.8853973398591 , 12861.561372421176  , -21081.267792388287  ],
               #[ 35700.10179233264   ,   9354.109817193366  , 22119.029731216495  , -28415.252152458863  ],
               #[-12143.246181972478  ,   9304.325391729337  , 31395.384058641182  ,   7237.147491898275  ],
               #[ -4252.550777411426  ,   7519.658007052243  , 32119.072173514964  ,  -4846.3100920239485 ],
               #[  3635.1379578203882 ,  14995.735536343693  , 35520.640399769225  , -16014.462747490608  ],
               #[ 16666.33261515447   ,  25463.251035728365  , 45741.95771702346   , -19758.46184204609   ],
               #[  1574.5488261582616 ,  -9195.300418814248  , 22533.366477056483  ,   4304.760507189294  ],
               #[  2550.849088781054  ,  -3908.160660561578  , 26450.020619498886  ,  -6460.6950465755135 ],
               #[ 12186.068833934241  ,   5472.8392849257625 , 31860.530178867026  , -20670.114822509688  ],
               #[ 22514.791740481334  ,  16536.089732434488  , 39066.09640356857   , -28532.093602592075  ],
               #[ 12928.473026323036  , -21086.967346594512  , 18614.547588293004  ,   6849.166159532339  ],
               #[ 14704.701763914212  , -15759.808856791195  , 20232.646027242234  ,  -8169.299502481842  ],
               #[ 19849.16127304192   ,  -2227.0814527066664 , 25527.207772188514  , -22471.01943596388   ],
               #[ 31996.863270436897  ,   9617.286587025734  , 33674.85826326058   , -29412.75972535348   ],
               #[ 27571.184845574342  , -30154.92833132108   , 18052.342571839046  ,  10994.484422065303  ],
               #[ 28739.967883616857  , -21708.041199466363  , 17956.427584180565  ,  -2942.9148440614954 ],
               #[ 32641.135622960424  ,  -6470.498554625317  , 24584.093550676018  , -16242.532681627823  ],
               #[ 42161.65891472082   ,   5797.052892924028  , 35563.21295339845   , -18727.363312963513  ],


               #[ 30227.779042908023  ,   9857.710175581427  , 2419.0919326443245 , -19887.52566781699   ],
               #[ -5903.964593440247  ,  33202.77481393229   , 7957.209762033468  ,  -3403.7969365721247 ],
               #[  3725.809446728156  ,  37700.79548882806   , 15063.548724431288  , -13664.649144477333  ],
               #[ 15947.869227600464  ,  44648.943258660605  , 25081.529560377585  , -20058.220629161304  ],
               #[   215.19603769850224,  23014.993488327826  , -9161.103815995526  ,   5071.569380244025  ],
               #[  3247.3054686833893 ,  24815.691618990004  , -5206.879884421554  ,  -8176.043923204597  ],
               #[ 11060.826883155185  ,  29847.855418648338  , 4258.595435264461  , -19887.52566781699   ],
               #[ 21768.294475061582  ,  37492.793506439346  , 16357.901777084895  , -27824.289099848742  ],
               #[ 12918.58013762351   ,  17115.432697396976  , -22361.66412851339   ,   5071.569380244025  ],
               #[ 15212.571847603098  ,  19093.55370936996   , -15711.57358585601   ,  -8176.043923204597  ],
               #[ 21438.599578036166  ,  24566.850898536428  , -3118.807183979872  , -19887.52566781699   ],
               #[ 30514.472945250072  ,  32760.863642183478  , 10898.798653724454  , -27824.289099848742  ],
               #[ 26651.024042081077  ,  14774.897380234697  , -26368.609645164128  ,   8830.624486571112  ],
               #[ 28463.873516037293  ,  16833.393361246715  , -18190.048791130077  ,  -3403.7969365721247 ],
               #[ 33526.83314853147   ,  22502.206175302246  , -4601.374137811253  , -13664.649144477333  ],
               #[ 41211.318254910875  ,  30931.858724513793  , 9866.130782274127  , -20058.220629161304  ],
               #[-20572.404528182687  ,  20860.7196881017    , 10848.610496132776  ,   5071.569380244025  ],
               #[-14458.089010508977  ,  22722.41796176582   , 13237.84027636323   ,  -8176.043923204597  ],
               #[ -2324.7956058676386 ,  27907.710842064876  , 19682.253928553346  , -19887.52566781699   ],
               #[ 11461.601489872537  ,  35745.97327067807   , 29000.68495320255   , -27824.289099848742  ],
               #[ -6535.799660216543  ,  10497.057811477613  , -1221.1754893015348 ,   1052.3346847687444 ],
               #[ -2889.254957297928  ,  12720.454456439455  , 1924.341296317239  , -13521.681811552347  ],
               #[  6064.250086156992  ,  18781.74766682315   , 9960.452700006292  , -27600.951225698656  ],
               #[ 17763.45544999196   ,  27670.082531109707  , 20872.924194352636  , -39559.11760543963   ],
               #[  8129.88862372445   ,   2938.709960143384  , -10087.171353494752  ,   1052.3346847687444 ],
               #[ 10656.6567097932    ,   5527.190972711832  , -6012.451677690179  , -13521.681811552347  ],
               #[ 17409.389728946684  ,  12415.117401520445  , 3643.346694295878  , -27600.951225698656  ],
               #[ 27060.29264904244   ,  22207.253239195466  , 15884.603974216403  , -39559.11760543963   ],
               #[ 22974.24866148881   ,   -213.44168372549908, -11988.273915172887  ,   5071.569380244025  ],
               #[ 24895.143476497047  ,   2564.3648422348965 , -7643.405641863044  ,  -8176.043923204597  ],
               #[ -9961.996519405897  ,  31610.40213246657   , 5267.875217223638  ,   8830.624486571112  ],
               #[ 38249.821665391624  ,  20062.738235228288  , 14951.777979000393  , -27824.289099848742  ],
               #[-20572.404528182687  ,  12243.470116270528  , 19980.036341503604  ,   5071.569380244025  ],
               #[-14458.089010508977  ,  14396.495712479493  , 21998.831666752998  ,  -8176.043923204597  ],
               #[ -2324.7956058676386 ,  20291.73276848505   , 27571.64010737195   , -19887.52566781699   ],
               #[ 11461.601489872537  ,  28988.337966178     , 35886.73009573617   , -27824.289099848742  ],
               #[ -6535.799660216543  ,   -270.97992436726537, 10319.856408871154  ,   1052.3346847687444 ],
               #[ -2889.254957297928  ,   2510.533318014749  , 12734.672244257818  , -13521.681811552347  ],
               #[  6064.250086156992  ,   9811.657251373803  , 19237.07546884779   , -27600.951225698656  ],
               #[ 17763.45544999196   ,  20024.415289947305  , 28618.927805690077  , -39559.11760543963   ],
               #[  8129.88862372445   , -10415.02312968063   , 4058.6343137488393 ,   1052.3346847687444 ],
               #[ 10656.6567097932    ,  -6787.41202223908   , 6822.949212091105  , -13521.681811552347  ],
               #[ 17409.389728946684  ,   2132.4396902077647 , 14088.425500850133  , -27600.951225698656  ],
               #[ 27060.29264904244   ,  13802.87437353505   , 24266.70463548644   , -39559.11760543963   ],
               #[ 22974.24866148881   , -15175.889542857863  , 2847.2840290781314 ,   5071.569380244025  ],
               #[ 24895.143476497047  , -10957.792099873388  , 5690.900889047167  ,  -8176.043923204597  ],
               #[ 30227.779042908023  ,  -1083.961831855856  , 13121.972456049007  , -19887.52566781699   ],
               #[ 38249.821665391624  ,  11313.69459873673   , 23463.860860499182  , -27824.289099848742  ],
               #[ -9961.996519405897  ,   6924.497532175368  , 31248.118610173056  ,   8830.624486571112  ],
               #[ -5903.964593440247  ,   9306.950688632409  , 32941.774222152664  ,  -3403.7969365721247 ],
               #[  3725.809446728156  ,  15735.610968462599  , 37701.75227204455   , -13664.649144477333  ],
               #[ 15947.869227600464  ,  25035.97366951648   , 44996.327277064534  , -20058.220629161304  ],
               #[   215.19603769850224,  -7544.58017471757   , 23424.159013185614  ,   5071.569380244025  ],
               #[  3247.3054686833893 ,  -4202.4832301639835 , 25331.151294055915  ,  -8176.043923204597  ],
               #[ 11060.826883155185  ,   4205.506698655637  , 30629.35616979539   , -19887.52566781699   ],
               #[ 21768.294475061582  ,  15446.65323711384   , 38608.80190105827   , -27824.289099848742  ],
               #[ 12918.58013762351   , -21191.72369492827   , 18665.934577308853  ,   5071.569380244025  ],
               #[ 15212.571847603098  , -15917.238877446995  , 20730.886694925626  ,  -8176.043923204597  ],
               #[ 21438.599578036166  ,  -4651.138555544674  , 26415.526672222244  , -19887.52566781699   ],
               #[ 30514.472945250072  ,   8652.190685780086  , 34864.623699097574  , -27824.289099848742  ],
               #[ 26651.024042081077  , -30082.168352230332  , 17780.735590224125  ,   8830.624486571112  ],
               #[ 28463.873516037293  , -22049.475358343363  , 19877.97443593676   ,  -3403.7969365721247 ],
               #[ 33526.83314853147   ,  -8541.749712252547  , 25640.3014050656    , -13664.649144477333  ],
               #[ 41211.318254910875  ,   5888.695860348433  , 34181.57523638666   , -20058.220629161304  ],
       #   ]
       # )

    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]

    motor_pos_samp = np.array([Dec(r) for r in motor_pos_samp.reshape(u*4)]).reshape((u, 4))
    xyz_of_samp = np.array([Dec(r) for r in xyz_of_samp.reshape(ux*3)]).reshape((ux, 3))


    if ux > u:
        print("Error: You have more xyz positions than samples!")
        print("You have %d xyz positions and %d samples" % (ux, u))
        sys.exit(1)

    def computeCost(solution):
        anch = anchorsvec2matrix(solution[0:params_anch])
        spool_buildup_factor = Dec(solution[-params_buildup])
        spool_r = np.array([Dec(x) for x in solution[-params_buildup + 1 :]])
        pos = np.zeros((u, 3))
        if np.size(xyz_of_samp) != 0:
            pos = np.vstack(
                (
                    xyz_of_samp,
                    np.reshape(
                        solution[params_anch:-params_buildup], (u - ux, 3)
                    ),
                )
            )
        else:
            pos = np.reshape([Dec(x) for x in solution[params_anch:-params_buildup]], (u, 3))
        return float(cost_sq_for_pos_samp(
            anchorsvec2matrix(solution[0:params_anch]),
            pos,
            motor_pos_samp,
            spool_buildup_factor,
            spool_r,
        ))

    ndim = 3 * (u - ux) + params_anch + params_buildup

    class candidate:
        name = "no_name"
        solution = np.zeros(ndim)
        anch = np.zeros((4, 3))
        spool_buildup_factor = 0.0
        cost = 9999.9
        pos = np.zeros(3 * (u - ux))

        def __init__(self, name, solution):
            self.name = name
            self.solution = solution
            if np.array(solution).any():
                self.cost = computeCost(solution)
                np.set_printoptions(suppress=False)
                print("%s has cost %e" % (self.name, self.cost))
                np.set_printoptions(suppress=True)
                self.anch = anchorsvec2matrix(self.solution[0:params_anch])
                self.spool_buildup_factor = self.solution[-params_buildup]
                self.spool_r = self.solution[-params_buildup + 1 :]
                if np.size(xyz_of_samp) != 0:
                    self.pos = np.vstack(
                        (
                            xyz_of_samp,
                            np.reshape(
                                solution[params_anch:-params_buildup], (u - ux, 3)
                            ),
                        )
                    )
                else:
                    self.pos = np.reshape(
                        solution[params_anch:-params_buildup], (u, 3)
                    )

    the_cand = candidate("no_name", np.zeros(ndim))
    st1 = timeit.default_timer()
    if args["method"] == "all":
        cands = [
                candidate(
                    cand_name,
                    solve(
                        motor_pos_samp, xyz_of_samp, cand_name, args["cx_is_positive"]
                        ),
                    )
                for cand_name in [
                    "L-BFGS-B",
                    "PowellDirectionalSolver",
                    "SLSQP",
                    "differentialEvolutionSolver",
                    "BuckShot",
                    ]
                ]
        cands[:] = sorted(cands, key=lambda cand: cand.cost)
        print("Winner method: is %s" % cands[0].name)
        the_cand = cands[0]
    else:
        the_cand = candidate(
            args["method"],
            solve(
                motor_pos_samp,
                xyz_of_samp,
                args["method"],
                args["cx_is_positive"],
            ),
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
            print(
                "       Debug flag is set, so printing bogus anchor values anyways."
            )
    elif (u + 3 * ux) < params_anch + 4:
        print(
            "\nWarning: Data set might be too small.\n         The below values are unreliable unless input data is extremely accurate."
        )

    print_copypasteable(
        the_cand.anch, the_cand.spool_buildup_factor, the_cand.spool_r
    )
    print("Spool buildup factor:", the_cand.spool_buildup_factor) # err
    print("Spool radii:", the_cand.spool_r)
    if args["debug"]:
        print_anch_err(the_cand.anch, anchors)
        print("Method: %s" % args["method"])
        print("RUN TIME : {0}".format(st2 - st1))
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)  # No scientific notation
        print("Data collected at positions: ")
        print(the_cand.pos)
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
