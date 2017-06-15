"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division # Always want 3/2 = 1.5
from itertools import product
import numpy as np
#import mystic
#import scipy

# Axes indexing
A = 0
B = 1
C = 2
D = 3
X = 0
Y = 1
Z = 2

def symmetric_anchors(l):
    anchors = np.array(np.zeros((4, 3)))
    #anchors[A, X] = 0 # Fixated
    anchors[A, Y] = -l
    #anchors[A, Z] = 0 # Fixated
    anchors[B, X] = l*np.cos(np.pi/6)
    anchors[B, Y] = l*np.sin(np.pi/6)
    #anchors[B, Z] = 0 # Fixated
    anchors[C, X] = -l*np.cos(np.pi/6)
    anchors[C, Y] = l*np.sin(np.pi/6)
    #anchors[C, Z] = 0 # Fixated
    #anchors[D, X] = 0 # Fixated
    #anchors[D, Y] = 0 # Fixated
    anchors[D, Z] = l
    return anchors

def centered_rand(l):
    """Sample from U(-l, l)"""
    return l*(2.*np.random.rand()-1.)

def irregular_anchors(l, fuzz_percentage = .2):
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
    #fuzz[A, X] = 0 # Fixated
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
    return symmetric_anchors(l)+fuzz

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
    pos = np.array(list(product(np.linspace(-l, l, n), repeat = 3))) \
            + 2.*fuzz*(np.random.rand(n**3, 3) - 0.5) \
            + [0, 0, l]
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

def cost(anchors, pos, samp):
    """If all positions and samples correspond perfectly, this returns 0.

    This is the systems of equations:
    sum for i from 1 to u
      sum for k from a to d
    |sqrt(sum for s from x to z (A_ks-s_i)^2) - sqrt(sum for s from x to z (A_ks-s_0)^2) - t_ik|

    or...
    sum for i from 1 to u
    |sqrt((A_ax-x_i)^2 + (A_ay-y_i)^2 + (A_az-z_i)^2) - sqrt((A_ax-x_0)^2 + (A_ay-y_0)^2 + (A_az-z_0)^2) - t_ib| +
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
    #return np.sum((samples(anchors, pos, fuzz = 0) - samp) * (samples(anchors, pos, fuzz = 0) - samp)) # Sum of squares

def anchorsvec2matrix(anchorsvec):
    """ Create a 4x3 anchors matrix from 6 element anchors vector.
    """
    anchors = np.array(np.zeros((4, 3)))
    anchors[A,Y] = anchorsvec[0];
    anchors[B,X] = anchorsvec[1];
    anchors[B,Y] = anchorsvec[2];
    anchors[C,X] = anchorsvec[3];
    anchors[C,Y] = anchorsvec[4];
    anchors[D,Z] = anchorsvec[5];
    return anchors

def anchorsmatrix2vec(a):
    return [a[A,Y], a[B, X], a[B,Y], a[C, X], a[C, Y], a[D, Z]]

def posvec2matrix(v, u):
    return np.reshape(v, (u,3))

def posmatrix2vec(m):
    return np.reshape(m, np.shape(m)[0]*3)

def solve(samp, cb):
    """Find reasonable positions and anchors given a set of samples.
    """
    u = np.shape(samp)[0]
    cos30 = np.cos(30*np.pi/180)
    sin30 = np.sin(30*np.pi/180)
    number_of_params_pos = 3*u
    number_of_params_anch = 6
    anchors_est = symmetric_anchors(1500.0)
    pos_est = [[0, 0, 500]]*u
    x_guess = list(anchorsmatrix2vec(anchors_est)) + list(posmatrix2vec(pos_est))


    def costx(posvec, anchvec):
        """Identical to cost, except the shape of inputs and capture of samp and u

        Parameters
        ----------
        x : [A_ay A_bx A_by A_cx A_cy A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """
        anchors = anchorsvec2matrix(anchvec)
        pos = np.reshape(posvec, (u,3))
        return cost(anchors, pos, samp)

    from mystic.termination import ChangeOverGeneration
    from mystic.solvers import DifferentialEvolutionSolver2

    # Bootstrap to give get x positons to cover some volume
    NP = 100 # Guessed size of the trial solution population
    pos_solver = DifferentialEvolutionSolver2(number_of_params_pos, NP)
    pos_solver.SetInitialPoints(x_guess[6:])
    pos_solver.SetEvaluationLimits(generations=1500)
    pos_solver.Solve(lambda x: costx(x, list(anchorsmatrix2vec(anchors_est))), \
                 termination = ChangeOverGeneration(generations=300, tolerance=2), \
                 callback = cb)
    x_guess[6:] = pos_solver.bestSolution

    # Main solver
    solver = DifferentialEvolutionSolver2(number_of_params_pos+number_of_params_anch, NP)
    solver.SetInitialPoints(x_guess)
    solver.SetEvaluationLimits(generations=5000)
    solver.Solve(lambda x: costx(x[6:], x[0:6]), \
                 termination = ChangeOverGeneration(generations=300), \
                 callback = cb)

    return anchorsvec2matrix(solver.bestSolution[0:6])



# Do basic testing and output results
# if this is run like a script
# and not imported like a module or package
if __name__ == "__main__":
    l_long = 2000
    l_short = 1000
    n = 3
    anchors = irregular_anchors(l_long)
    pos = positions(n, l_short, fuzz = 5)
    u = np.shape(pos)[0]

    # Plot out real position and anchor
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()
    # Make anchor figure and position figure.
    # Put the right answers onto those figures
    fig_anch = plt.figure()
    fig_pos = plt.figure()
    ax_anch = fig_anch.add_subplot(111, projection='3d')
    ax_pos = fig_pos.add_subplot(111, projection='3d')
    scat_anch0 = ax_anch.scatter(anchors[:,0], anchors[:,1], anchors[:,2], 'ro')
    scat_pos0 = ax_pos.scatter(pos[:,0], pos[:,1], pos[:,2], 'k+')
    plt.pause(1)
    scat_anch = ax_anch.scatter(anchors[:,0], anchors[:,1], anchors[:,2], 'yx')
    scat_pos = ax_pos.scatter(pos[:,0], pos[:,1], pos[:,2], 'b+')

    iter = 0
    def replot(x):
        """Call while pos solver is running.
        """
        global iter, u
        if iter%30 == 0:
            if len(x) == 6 + 3*u:
                ps = posvec2matrix(x[6:], u)
                scat_pos._offsets3d = (ps[:,0], ps[:,1], ps[:,2])
                anch = anchorsvec2matrix(x[0:6])
                scat_anch._offsets3d = (anch[:,0], anch[:,1], anch[:,2])
                print("Anchor errors: ")
                print(anchorsvec2matrix(x[0:6]) - anchors)
            elif len(x) == 6:
                anch = anchorsvec2matrix(x[0:6])
                scat_anch._offsets3d = (anch[:,0], anch[:,1], anch[:,2])
            else:
                ps = posvec2matrix(x, u)
                scat_pos._offsets3d = (ps[:,0], ps[:,1], ps[:,2])
            plt.draw()
            plt.pause(0.1)
        iter += 1

    sample_fuzz = 0
    samp = samples(anchors, pos, fuzz = sample_fuzz)
    solution = solve(samp, replot)
    print("Anchor errors were: ")
    print(solution - anchors)
    print("Cost were: %f" % cost(solution, pos, samp))
