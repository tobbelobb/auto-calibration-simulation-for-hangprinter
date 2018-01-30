"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division # Always want 3/2 = 1.5
import numpy as np

# Tips on how to use differential solver:
# build/lib.linux-x86_64-2.7/mystic/differential_evolution.py
# http://www.icsi.berkeley.edu/~storn/code.html

# Axes indexing
A = 0
B = 1
C = 2
D = 3
X = 0
Y = 1
Z = 2

def symmetric_anchors(l, az=-120., bz=-120., cz=-120.):
    anchors = np.array(np.zeros((4, 3)))
    anchors[A, Y] = -l
    anchors[A, Z] = az
    anchors[B, X] = l*np.cos(np.pi/6)
    anchors[B, Y] = l*np.sin(np.pi/6)
    anchors[B, Z] = bz
    anchors[C, X] = -l*np.cos(np.pi/6)
    anchors[C, Y] = l*np.sin(np.pi/6)
    anchors[C, Z] = cz
    anchors[D, Z] = l
    return anchors

def centered_rand(l):
    """Sample from U(-l, l)"""
    return l*(2.*np.random.rand()-1.)

def irregular_anchors(l, fuzz_percentage = .2, az=-120., bz=-120.,cz=-120.):
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
    fuzz[A, Y] = centered_rand(l*fuzz_percentage)
    #fuzz[A, Z] = 0 # Fixated
    fuzz[B, X] = centered_rand(l*fuzz_percentage*np.cos(np.pi/6))
    fuzz[B, Y] = centered_rand(l*fuzz_percentage*np.sin(np.pi/6))
    #fuzz[B, Z] = 0 # Fixated
    fuzz[C, X] = centered_rand(l*fuzz_percentage*np.cos(np.pi/6))
    fuzz[C, Y] = centered_rand(l*fuzz_percentage*np.sin(np.pi/6))
    #fuzz[C, Z] = 0 # Fixated
    #fuzz[D, X] = 0 # Fixated
    #fuzz[D, Y] = 0 # Fixated
    fuzz[D, Z] = l*fuzz_percentage*np.random.rand() # usually higher than A is long
    return symmetric_anchors(l, az, bz, cz)+fuzz

def positions(n, l, fuzz=10):
    """Return (n^3)x3 matrix of positions in fuzzed grid of side length 2*l

    Move to u=n^3 positions in an fuzzed grid of side length 2*l
    centered around (0, 0, l).

    Parameters
    ----------
    n : Number of positions of which to sample along each axis
    l : Max length from origo along each axis to sample
    fuzz: How much each measurement point can differ from the regular grid
    """
    from itertools import product
    pos = np.array(list(product(np.linspace(-l, l, n), repeat = 3))) \
            + 2.*fuzz*(np.random.rand(n**3, 3) - 0.5) \
            + [0, 0, 1*l]
    index_closest_to_origo = np.int(np.shape(pos)[0]/2)-int(n/2)
    # Make pos[0] a point fairly close to origo
    tmp = pos[0].copy()
    pos[0] = pos[index_closest_to_origo]
    pos[index_closest_to_origo] = tmp
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
    line_lengths = np.linalg.norm(anchors - pos[:,np.newaxis,:], 2, 2)
    return line_lengths - line_lengths[0] + 2.*fuzz*(np.random.rand(np.shape(pos)[0], 1) - 0.5)

def samples_relative_to_origo(anchors, pos, fuzz=1):
    """Possible relative line length measurments according to anchors and position.

    Parameters
    ----------
    anchors : 4x3 matrix of anhcor positions in mm
    pos : ux3 matrix of positions
    fuzz: Maximum measurment error per motor in mm
    """
    # pos[:,np.newaxis,:]: ux1x3
    # Broadcasting happens u times and we get ux4x3 output before norm operation
    line_lengths = np.linalg.norm(anchors - pos[:,np.newaxis,:], 2, 2)
    return line_lengths - np.linalg.norm(anchors,2,1) + 2.*fuzz*(np.random.rand(np.shape(pos)[0], 1) - 0.5)

def samples_relative_to_origo_no_fuzz(anchors, pos):
    """Possible relative line length measurments according to anchors and position.

    Parameters
    ----------
    anchors : 4x3 matrix of anhcor positions in mm
    pos : ux3 matrix of positions
    fuzz: Maximum measurment error per motor in mm
    """
    # pos[:,np.newaxis,:]: ux1x3
    # Broadcasting happens u times and we get ux4x3 output before norm operation
    line_lengths = np.linalg.norm(anchors - pos[:,np.newaxis,:], 2, 2)
    return line_lengths - np.linalg.norm(anchors,2,1)

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
    return np.sum(np.abs(samples(anchors, pos, fuzz = 0) - samp))

def cost_sq(anchors, pos, samp):
    #pos[0] = [0.0, 0.0, 0.0]
    return np.sum(pow((samples_relative_to_origo_no_fuzz(anchors, pos, fuzz = 0) - samp), 2)) # Sum of squares

def anchorsvec2matrix(anchorsvec, az = 0., bz = 0., cz = 0.):
    """ Create a 4x3 anchors matrix from 6 element anchors vector.
    """
    anchors = np.array(np.zeros((4, 3)))
    anchors[A,Y] = anchorsvec[0];
    anchors[A,Z] = az;
    anchors[B,X] = anchorsvec[1];
    anchors[B,Y] = anchorsvec[2];
    anchors[B,Z] = bz;
    anchors[C,X] = anchorsvec[3];
    anchors[C,Y] = anchorsvec[4];
    anchors[C,Z] = cz;
    anchors[D,Z] = anchorsvec[5];
    return anchors

def anchorsmatrix2vec(a):
    return [a[A,Y], a[B, X], a[B,Y], a[C, X], a[C, Y], a[D, Z]]

def posvec2matrix(v, u):
    return np.reshape(v, (u,3))

def posmatrix2vec(m):
    return np.reshape(m, np.shape(m)[0]*3)

def solve(samp, _cost = cost_sq, az = 0., bz = 0., cz = 0.):
    """Find reasonable positions and anchors given a set of samples.
    """
    def costx(posvec, anchvec):
        """Identical to cost, except the shape of inputs and capture of samp and u

        Parameters
        ----------
        x : [A_ay A_bx A_by A_cx A_cy A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """
        anchors = anchorsvec2matrix(anchvec, az, bz, cz)
        pos = np.reshape(posvec, (u,3))
        return _cost(anchors, pos, samp)

    l_anch = 1500.0
    l_pos = 450
    l_long = 4000.0
    l_short = 1700.0
    u = np.shape(samp)[0]
    cos30 = np.cos(30*np.pi/180)
    sin30 = np.sin(30*np.pi/180)
    number_of_params_pos = 3*u
    number_of_params_anch = 6

    # Define bounds
    lb = [      -l_long, # A_ay > -4000.0
                      0, # A_bx > 0
                      0, # A_by > 0
          -l_long*cos30, # A_cx > -4000*cos(30)
                      0, # A_cy > 0
                      0, # A_dz > 0
          ] + [-l_short, -l_short, -10.]*u
    ub = [            0, # A_ay < 0
           l_long*cos30, # A_bx < 4000.0*cos(30)
           l_long*sin30, # A_by < 4000.0*sin(30)
                      0, # A_cx < 0
           l_long*sin30, # A_cy < 4000.0*sin(30)
                 l_long, # A_dz < 4000.0
          ] + [l_short, l_short, 2*l_short]*u

    from mystic.termination import ChangeOverGeneration, NormalizedChangeOverGeneration, VTR
    from mystic.solvers import DifferentialEvolutionSolver2, PowellDirectionalSolver

    pos_est0 = np.zeros((u,3))
    anchors_est = np.array([[0.0, 0.0, az],
                            [0.0, 0.0, bz],
                            [0.0, 0.0, cz],
                            [0.0, 0.0, 0.0]])
    x_guess0 = list(anchorsmatrix2vec(anchors_est)) + list(posmatrix2vec(pos_est0))

    from mystic.termination import Or, CollapseAt, CollapseAs
    from mystic.termination import ChangeOverGeneration as COG

    target = 1.0
    term = Or((COG(generations=100), CollapseAt(target, generations=100)))

    #print("Solver 0")
    solver0 = PowellDirectionalSolver(number_of_params_pos+number_of_params_anch)
    solver0.SetEvaluationLimits(evaluations=3200000, generations=10000)
    solver0.SetTermination(term)

    solver0.SetInitialPoints(x_guess0)
    solver0.SetStrictRanges(lb, ub)
    solver0.Solve(lambda x: costx(x[6:], x[0:6]))
    x_guess0 = solver0.bestSolution

    for i in range(1,20):
        #print(", %d" % i)
        solver0 = PowellDirectionalSolver(number_of_params_pos+number_of_params_anch)
        solver0.SetInitialPoints(x_guess0)
        solver0.SetStrictRanges(lb, ub)
        solver0.Solve(lambda x: costx(x[6:], x[0:6]))
        x_guess0 = solver0.bestSolution

    return solver0.bestSolution

if __name__ == "__main__":
    # Gotten from manual measuring
    az = -115.
    bz = -115.
    cz = -115.
    anchors = np.array([[0.0, -1112.0, az],
                        [970.0, 550.0, bz],
                        [-970.0, 550.0, cz],
                        [0.0, 0.0, 2865.0]])

# data 1
#    samp = np.array([
#[126.31  , 5.02    , -0.21   , -213.52],
#[295.03  , -257.68 , 218.73  , -244.16],
#[511.65  , 94.13   , 116.17  , -585.52],
#[373.57  , 615.00  , -132.03 , -570.93],
#[285.95  , 468.10  , -475.99 , -112.57],
#[411.75  , -471.95 , 279.45  , -61.84],
#[646.11  , 257.49  , 289.34  , -845.42],
#[43.83   , 384.27  , 262.25  , -618.82],
#[-416.94 , 392.71  , 305.03  , -178.76],
#[-355.53 , 308.31  , 408.93  , -267.15],
#[191.34  , 555.78  , 209.78  , -741.28],
#[537.90  , 574.98  , 470.11  , -1102.07],
#[636.51  , 380.17  , 709.07  , -1118.74],
#[897.10  , 913.95  , 702.54  , -1473.05]
#])

# data 2
#    samp = np.array([
#[400.53 , 175.53 , 166.10 , -656.90],
#[229.27 , 511.14 , -48.41 , -554.31],
#[-41.69 , -62.87 , 306.76 , -225.31],
#[272.97 , 176.65 , 381.13 , -717.81],
#[338.07 , 633.70 , 309.27 , -911.22],
#[504.47 , 658.88 , 48.60 , -794.42],
#[504.47 , 658.88 , 48.60 , -794.42],
#[103.50 , 569.98 , 633.68 , -860.25],
#[229.37 , 7.32 , 411.98 , -575.81],
#[428.73 , -413.46 , 250.38 , -133.93],
#[-506.97 , 343.33 , 327.68 , -4.40]
#        ])

# data 3
    samp = np.array([
[571.85 , -773.44 , 578.54 , 59.12],
[647.64 , 654.08 , -858.11 , 90.96],
[-860.99 , 675.57 , 630.82 , 94.65],
[318.70 , 741.68 , 209.66 , -824.57],
[754.92 , 466.22 , 418.71 , -1066.04],
[450.51 , 403.77 , 699.62 , -1046.80],
[1234.76 , 1174.05 , 1183.26 , -1913.14],
[283.26 , 162.84 , 182.21 , -603.33],
[663.00 , -139.32 , 603.13 , -648.02],
[661.00 , 625.20 , -73.58 , -713.35],
[-33.50 , 643.97 , 658.06 , -754.40]
        ])


    u = np.shape(samp)[0]
    pos = np.zeros((u, 3))

    the_cost = 100000.0
    best_cost = 100000.0
    best_az = 1000.0
    best_bz = 1000.0
    best_cz = 1000.0
    for azi in np.arange(az+10.,az-25.1,-5.):
        for bzi in np.arange(bz+10.,bz-25.1,-5.):
            for czi in np.arange(cz+10.,cz-25.1,-5.):
                solution = solve(samp, cost_sq, az, bz, cz)
                sol_anch = anchorsvec2matrix(solution[0:6], az, bz, cz)
                the_cost = cost_sq(anchorsvec2matrix(solution[0:6], az, bz, cz), np.reshape(solution[6:], (u,3)), samp)
                if(the_cost < best_cost):
                    best_cost = the_cost
                    best_az = azi
                    best_bz = bzi
                    best_cz = czi
                    print("New record cost: %f" % the_cost)
                    print("Output Anchors: ")
                    print(sol_anch)
                    print("Errors: ")
                    print(sol_anch - anchors)
