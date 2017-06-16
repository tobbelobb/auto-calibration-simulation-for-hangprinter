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

def cost_sq(anchors, pos, samp):
    return np.sum(pow((samples(anchors, pos, fuzz = 0) - samp), 2)) # Sum of squares

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

def solve(samp, cb, _cost = cost_sq):
    """Find reasonable positions and anchors given a set of samples.
    """
    def costx(posvec, anchvec):
        """Identical to cost, except the shape of inputs and capture of samp and u

        Parameters
        ----------
        x : [A_ay A_bx A_by A_cx A_cy A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """
        anchors = anchorsvec2matrix(anchvec)
        pos = np.reshape(posvec, (u,3))
        return _cost(anchors, pos, samp)

    l_anch = 1500.0
    l_pos = 750
    l_long = 5000.0
    l_short = 800.0
    u = np.shape(samp)[0]
    cos30 = np.cos(30*np.pi/180)
    sin30 = np.sin(30*np.pi/180)
    number_of_params_pos = 3*u
    number_of_params_anch = 6
    anchors_est = symmetric_anchors(l_anch)
    pos_est = np.random.rand(u,3)*l_short - [l_short/2, l_short/2, 0]
    x_guess = list(anchorsmatrix2vec(anchors_est)) + list(posmatrix2vec(pos_est))

    # Define bounds
    lb = [      -l_long, # A_ay > -5000.0
                      0, # A_bx > 0
                      0, # A_by > 0
          -l_long*cos30, # A_cx > -5000*cos(30)
                      0, # A_cy > 0
                      0, # A_dz > 0
                  -50.0, # x0   > -50.0
                  -50.0, # y0   > -50.0
                    -10, # z0   > -10
          ] + [-l_short, -l_short, -10]*(u-1)
    ub = [            0, # A_ay < 0
           l_long*cos30, # A_bx < 5000.0*cos(30)
           l_long*sin30, # A_by < 5000.0*sin(30)
                      0, # A_cx < 0
           l_long*sin30, # A_cy < 5000.0*sin(30)
               2*l_long, # A_dz < 10000.0
                   50.0, # x0   < 50.0
                   50.0, # y0   < 50.0
                l_short, # z0   < l_short
          ] + [l_short, l_short, 2*l_short]*(u-1)

    from mystic.termination import ChangeOverGeneration
    from mystic.solvers import DifferentialEvolutionSolver2

    # We know that our anchor guess is in the right directions but our
    # position guesses are still random. This optimization fixes that
    NP = 60 # Guessed size of the trial solution population
    pos_solver = DifferentialEvolutionSolver2(number_of_params_pos, NP)
    pos_solver.SetStrictRanges(lb[6:], ub[6:])
    pos_solver.SetInitialPoints(x_guess[6:])
    pos_solver.SetEvaluationLimits(generations=2000)
    pos_solver.Solve(lambda x: costx(x, list(anchorsmatrix2vec(anchors_est))), \
                 termination = ChangeOverGeneration(generations=300, tolerance=2), \
                 callback = cb, \
                 CrossProbability=0.8, ScalingFactor=0.8)
    x_guess[6:] = pos_solver.bestSolution

    def explode_points(x, fac, diffZ):
        new_x = list(x)
        for i in range(0,6):
            new_x[i] = new_x[i]*fac
        for i in range(8, len(x), 3):
            #new_x[i] = new_x[i] + diffZ
            new_x[i] = new_x[i]*fac
        return new_x

    def elevate(x, diffD, diffZ):
        new_x = list(x)
        new_x[5] = x[5] + diffD
        for i in range(8, len(x),3):
            new_x[i] = x[i] + diffZ
        return new_x

    # We know that our anchor guesses are along the right directions
    # but we don't know if their lengths are right. This optimization fixes that.
    percentage = 1.0
    while(percentage > 0.0005):
        solver = DifferentialEvolutionSolver2(number_of_params_pos+number_of_params_anch, NP)
        solver.SetStrictRanges(lb, ub)
        solver.SetInitialPoints(x_guess)
        solver.SetEvaluationLimits(generations=30)
        solver.Solve(lambda x: costx(x[6:], x[0:6]), \
                     termination = ChangeOverGeneration(generations=300), \
                     callback = cb, \
                     CrossProbability=0.8, ScalingFactor=0.8)

        z_diff = np.abs(x_guess[5] - solver.bestSolution[5])
        x_low = explode_points(solver.bestSolution, 1.0 - 0.01*percentage, 0)
        x_high = explode_points(solver.bestSolution, 1.0 + 0.01*percentage, 0)
        cost_low = costx(x_low[6:], x_low[0:6])
        cost_high = costx(x_high[6:], x_high[0:6])
        cost_mid = costx(solver.bestSolution[6:], solver.bestSolution[0:6])
        print( "cost_low: %f" %      cost_low )
        print( "cost_high:  %f" %     cost_high)
        print( "cost_mid: %f" %       cost_mid )

        # Try to set the right radius for the anchors...
        if (cost_low < 0.999*cost_mid) & (cost_low < cost_high):
            while((cost_low < 0.999*cost_mid) & (cost_low < cost_high)):
                x_low = explode_points(x_low, 1.0 - 0.01*percentage, 0)
                cost_low = costx(x_low[6:], x_low[0:6])
                print("Imploding and lowering")
            x_guess = x_low
        elif (cost_high < 0.999*cost_mid) & (cost_high < cost_low):
            while((cost_high < 0.999*cost_mid) & (cost_high < cost_low)):
                x_high = explode_points(x_high, 1.0 + 0.01*percentage, 0)
                cost_high = costx(x_high[6:], x_high[0:6])
                print("Exploding and heighering")
            x_guess = x_high
        else:
            x_guess = solver.bestSolution
            percentage = percentage/2
            print("Not highering or lowering. z_diff was %f" % z_diff)
        print("percentage = %f" % percentage)

    # Main optimization that narrows down the leftover fuzz
    x_guess = solver.bestSolution
    #NP = 10*(number_of_params_pos+number_of_params_anch) # Guessed size of the trial solution population
    NP = 80
    from mystic.strategy import Rand1Bin, Best1Exp, Best1Bin, Rand1Exp
    solver = DifferentialEvolutionSolver2(number_of_params_pos+number_of_params_anch, NP)
    solver.SetStrictRanges(lb, ub)
    solver.SetInitialPoints(x_guess)
    solver.SetEvaluationLimits(generations=1000000)
    solver.Solve(lambda x: costx(x[6:], x[0:6]), \
                 termination = ChangeOverGeneration(generations=300), \
                 callback = cb, \
                 CrossProbability=0.5, ScalingFactor=0.8, strategy = Best1Bin)
    x_guess = solver.bestSolution

    return anchorsvec2matrix(solver.bestSolution[0:6])

if __name__ == "__main__":
    l_long = 2500
    l_short = 800
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
    plt.close("all")
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
                print("cost: %f" % \
                    cost(anchorsvec2matrix(x[0:6]), np.reshape(x[6:], (u,3)), samp))

            elif len(x) == 6:
                anch = anchorsvec2matrix(x[0:6])
                scat_anch._offsets3d = (anch[:,0], anch[:,1], anch[:,2])
            else:
                ps = posvec2matrix(x, u)
                scat_pos._offsets3d = (ps[:,0], ps[:,1], ps[:,2])
            plt.draw()
            plt.pause(0.001)
        iter += 1

    sample_fuzz = 0
    samp = samples(anchors, pos, fuzz = sample_fuzz)
    solution = solve(samp, replot)
    print("Anchor errors were: ")
    print(solution - anchors)
    print("Real cost were: %f" % cost(solution, pos, samp))
