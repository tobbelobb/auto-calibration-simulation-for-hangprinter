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
import concurrent.futures

import signal
import time

from hangprinter_forward_transform import forward_transform
from flex_distance import *
from util import *
from data import *


## Algorithm help and tuning
low_axis_min_force_limit = 0
low_axis_max_force_limit = 120

l_long = 14000.0  # The longest distance from the origin that we should consider for anchor positions
l_short = 3000.0  # The longest distance from the origin that we should consider for data point collection
data_z_min = -100.0  # The lowest z-coordinate the algorithm should care about guessing
xyz_offset_max = (
    1.0  # Tell the algorithm to check if all xyz-data may carry an offset error compared to the encoder-data
)

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


    # use_forces = False
    # if use_forces:
    #     err += cost_from_forces(anchors, pos, force_samp, mover_weight, low_axis_max_force)

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


def parallel_optimize(random_guess, lb, ub, costx, params_anch, params_buildup, params_perturb, use_flex, use_line_lengths, line_lengths_when_at_origin, constant_spool_buildup_factor, disp, maxiter, motor_pos_samp, xyz_of_samp):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Values in x were outside bounds during a minimize step, clipping to bounds')
        sol = scipy.optimize.minimize(
            lambda x: costx(
                x[params_anch : -(params_buildup + params_perturb + use_flex)],
                x[0:params_anch],
                constant_spool_buildup_factor,
                x[-(params_buildup + params_perturb + use_flex) : -(params_perturb + use_flex)],
                line_lengths_when_at_origin,
                x[-(params_perturb + use_flex):(x.size - use_flex)],
                use_flex,
                use_line_lengths,
                x[-1],
                motor_pos_samp,
                xyz_of_samp,
            ),
            random_guess,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            options={"disp": disp, "ftol": 1e-9, "maxiter": maxiter},
        )
    return sol

def costx(
    posvec,
    anchvec,
    spool_buildup_factor,
    spool_r,
    line_lengths_when_at_origin,
    perturb,
    use_flex,
    use_line_lengths,
    low_axis_max_force,
    motor_pos_samp,
    xyz_of_samp,
):
    """Identical to cost, except the shape of inputs and capture of samp, xyz_of_samp, ux, and u"""


    u = np.shape(motor_pos_samp)[0]
    ux = np.shape(xyz_of_samp)[0]
    if len(posvec) > 0:
        posvec = np.array([pos for pos in posvec])
    anchvec = np.array([anch for anch in anchvec])
    spool_r = np.array([r for r in spool_r])
    spool_r = np.r_[spool_r[0], spool_r[0], spool_r[0], spool_r]
    perturb = np.array([p for p in perturb])

    anchors = anchorsvec2matrix(anchvec)
    pos = np.zeros((u, 3))
    if np.size(xyz_of_samp) != 0:
        pos[0:ux] = xyz_of_samp
    if u > ux:
        pos[ux:] = np.reshape(posvec, (u - ux, 3))

    return cost_sq_for_pos_samp(
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

    tries = 8
    random_guesses = [np.array([b[0] + (b[1] - b[0]) * np.random.rand() for b in list(zip(lb, ub))]) for _ in range(tries)]


    with concurrent.futures.ProcessPoolExecutor() as executor:
        solutions = list(executor.map(parallel_optimize, random_guesses, [lb] * tries, [ub] * tries, [costx] * tries, [params_anch] * tries, [params_buildup] * tries, [params_perturb] * tries, [use_flex] * tries, [use_line_lengths] * tries, [line_lengths_when_at_origin] * tries, [constant_spool_buildup_factor] * tries, [disp] * tries, [maxiter] * tries, [motor_pos_samp] * tries, [xyz_of_samp] * tries ))

    for sol in solutions:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find best Hangprinter config based on true line lengths, line difference samples, and xyz positions if known."
    )
    parser.add_argument("-a", "--advanced", help="Use the advanced cost function", action="store_true")
    parser.add_argument("-d", "--debug", help="Print debug information", action="store_true")
    args = vars(parser.parse_args())

    use_flex = args["advanced"]
    use_line_lengths = True

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

    samples_limit = 26
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
        print(the_cand.anch)

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
