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
    fuzz=0,
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
    degrees_per_unit_times_r_tmp = (
        spool_to_motor_gearing_factor * mech_adv * 360.0
    ) / (2.0 * np.pi)
    k0 = 2.0 * degrees_per_unit_times_r_tmp / k2

    line_lengths_origin = np.linalg.norm(anchors - np.array([[[0, 0, 0]]]), 2, 2)

    relative_line_lengths = samples_relative_to_origin_no_fuzz(anchors, pos)
    motor_positions = k0 * (
        np.sqrt(spool_r_in_origin_sq + relative_line_lengths * k2) - spool_r_in_origin
    )

    return motor_positions


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
    )  # Sum of squares


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


def solve(samp, xyz_of_samp, _cost, method, cx_is_positive=False):
    """Find reasonable positions and anchors given a set of samples.
    """

    u = np.shape(samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    number_of_params_pos = 3 * (u - ux)

    def costx(posvec, anchvec):
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
        return _cost(anchors, pos, samp)

    l_long = 5000.0
    l_short = 1700.0
    data_z_min = -20.0
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
    lb = [
        -l_long,  # A_ay > -4000.0
        -l_short,  # A_az > -1700.0
        0.0,  # A_bx > 0
        0.0,  # A_by > 0
        -l_short,  # A_bz > -1700.0
        -l_long,  # A_cx > -4000
        0.0,  # A_cy > 0
        -l_short,  # A_cz > -1700.0
        0.0,  # A_dz > 0
    ] + [-l_short, -l_short, data_z_min] * (u - ux)
    ub = [
        0.0,  # A_ay < 0
        l_short,  # A_az < 1700
        l_long,  # A_bx < 4000
        l_long,  # A_by < 4000
        l_short,  # A_bz < 1700
        0.0,  # A_cx < 0
        l_long,  # A_cy < 4000.0
        l_short,  # A_cz < 1700
        l_long,  # A_dz < 4000.0
    ] + [l_short, l_short, 2 * l_short] * (u - ux)

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

    pos_est0 = np.zeros((u - ux, 3))  # The positions we need to estimate
    anchors_est = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    x_guess0 = list(anchorsmatrix2vec(anchors_est)) + list(posmatrix2vec(pos_est0))

    if method == "PowellDirectionalSolver":
        from mystic.termination import (
            ChangeOverGeneration,
            NormalizedChangeOverGeneration,
            VTR,
        )
        from mystic.solvers import PowellDirectionalSolver
        from mystic.termination import Or, CollapseAt, CollapseAs
        from mystic.termination import ChangeOverGeneration as COG

        target = 1.0
        term = Or((COG(generations=100), CollapseAt(target, generations=100)))
        # Solver 0
        solver0 = PowellDirectionalSolver(number_of_params_pos + params_anch)
        solver0.SetEvaluationLimits(evaluations=3200000, generations=10000)
        solver0.SetTermination(term)
        solver0.SetInitialPoints(x_guess0)
        solver0.SetStrictRanges(lb, ub)
        solver0.Solve(lambda x: costx(x[params_anch:], x[0:params_anch]))
        x_guess0 = solver0.bestSolution
        # PowellDirectional sometimes finds new ways if kickstarted anew
        for i in range(1, 20):
            solver0 = PowellDirectionalSolver(number_of_params_pos + params_anch)
            solver0.SetInitialPoints(x_guess0)
            solver0.SetStrictRanges(lb, ub)
            solver0.Solve(lambda x: costx(x[params_anch:], x[0:params_anch]))
            x_guess0 = solver0.bestSolution
        return x_guess0
    elif method == "SLSQP":
        # 'SLSQP' is crazy fast and lands on 0.0000 error
        x_guess0 = scipy.optimize.minimize(
            lambda x: costx(x[params_anch:], x[0:params_anch]),
            x_guess0,
            method=method,
            bounds=list(zip(lb, ub)),
            options={"disp": True, "ftol": 1e-20, "maxiter": 150000},
        )
        return x_guess0.x
    elif method == "L-BFGS-B":
        ## 'L-BFGS-B' Is crazy fast but doesn't quite land at 0.0000 error
        x_guess0 = scipy.optimize.minimize(
            lambda x: costx(x[params_anch:], x[0:params_anch]),
            x_guess0,
            method=method,
            bounds=list(zip(lb, ub)),
            options={"ftol": 1e-12, "maxiter": 150000},
        )
        return x_guess0.x
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
        help="Available methods are L-BFGS-B (default), PowellDirectionalSolver (requires a library called Mystic), and SLSQP. As a shorthand, you can use 0, 1, or 2, for referring to the three methods respectively.",
        default="L-BFGS-B",
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

    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    anchors = np.array(
        [
            [0.0, -1112.0, -115.0],
            [970.0, 550.0, -115.0],
            [-970.0, 550.0, -115.0],
            [0.0, 0.0, 2865.0],
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

    samp = args["sample_data"]
    if np.size(samp) != 0:
        if np.size(samp) % 4 != 0:
            print("Please specify A, B, C, and D diffs of sampling points.")
            print(
                "You specified %d numbers after your -s/--sample_data option, which is not a multiple of 4 number of numbers."
            )
            sys.exit(1)
        samp = samp.reshape((int(np.size(samp) / 4), 4))
    else:
        # You might want to manually replace this with your collected data
        samp = np.array(
            [
                [400.53, 175.53, 166.10, -656.90],
                [229.27, 511.14, -48.41, -554.31],
                [-41.69, -62.87, 306.76, -225.31],
                [272.97, 176.65, 381.13, -717.81],
                [338.07, 633.70, 309.27, -911.22],
                [504.47, 658.88, 48.60, -794.42],
                [504.47, 658.88, 48.60, -794.42],
                [103.50, 569.98, 633.68, -860.25],
                [229.37, 7.32, 411.98, -575.81],
                [428.73, -413.46, 250.38, -133.93],
                [-506.97, 343.33, 327.68, -4.40],
            ]
        )

    u = np.shape(samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    if ux > u:
        print("Error: You have more xyz positions than samples!")
        print("You have %d xyz positions and %d samples" % (ux, u))
        sys.exit(1)

    st1 = timeit.default_timer()
    solution = solve(samp, xyz_of_samp, cost_sq, args["method"], args["cx_is_positive"])
    st2 = timeit.default_timer()
    sol_anch = anchorsvec2matrix(solution[0:params_anch])
    sol_pos = np.zeros((u, 3))

    if np.size(xyz_of_samp) != 0:
        sol_pos = np.vstack(
            (xyz_of_samp, np.reshape(solution[params_anch:], (u - ux, 3)))
        )
        the_cost = cost_sq(anchorsvec2matrix(solution[0:params_anch]), sol_pos, samp)
    else:
        sol_pos = np.reshape(solution[params_anch:], (u, 3))
        the_cost = cost_sq(anchorsvec2matrix(solution[0:params_anch]), sol_pos, samp)
    print("samples:          %d" % u)
    print("input xyz coords: %d" % (3 * ux))
    print("total cost:       %f" % the_cost)
    print("cost per sample:  %f" % (the_cost / u))

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
    if args["debug"]:
        print_anch_err(sol_anch, anchors)
        print("Method: %s" % args["method"])
        print("RUN TIME : {0}".format(st2 - st1))
        print("positions: ")
        print(sol_pos)
