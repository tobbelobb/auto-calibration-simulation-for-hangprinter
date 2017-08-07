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
    pos[0] = [0.0, 0.0, 0.0]
    return np.sum(pow((samples(anchors, pos, fuzz = 0) - samp), 2)) # Sum of squares

def anchorsvec2matrix(anchorsvec, z = 0):
    """ Create a 4x3 anchors matrix from 6 element anchors vector.
    """
    anchors = np.array(np.zeros((4, 3)))
    anchors[A,Y] = anchorsvec[0];
    anchors[A,Z] = z;
    anchors[B,X] = anchorsvec[1];
    anchors[B,Y] = anchorsvec[2];
    anchors[B,Z] = z;
    anchors[C,X] = anchorsvec[3];
    anchors[C,Y] = anchorsvec[4];
    anchors[C,Z] = z;
    anchors[D,Z] = anchorsvec[5];
    return anchors

def anchorsmatrix2vec(a):
    return [a[A,Y], a[B, X], a[B,Y], a[C, X], a[C, Y], a[D, Z]]

def posvec2matrix(v, u):
    return np.reshape(v, (u,3))

def posmatrix2vec(m):
    return np.reshape(m, np.shape(m)[0]*3)

def solve(samp, cb, _cost = cost_sq, z = 0):
    """Find reasonable positions and anchors given a set of samples.
    """
    def costx(posvec, anchvec):
        """Identical to cost, except the shape of inputs and capture of samp and u

        Parameters
        ----------
        x : [A_ay A_bx A_by A_cx A_cy A_dz
               x1   y1   z1   x2   y2   z2   ...  xu   yu   zu
        """
        anchors = anchorsvec2matrix(anchvec, z)
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

    # Define bounds
    lb = [      -l_long, # A_ay > -5000.0
          -l_long*cos30, # A_bx > -5000*cos(30)
                      0, # A_by > 0
                      0, # A_cx > 0
                      0, # A_cy > 0
                      0, # A_dz > 0
                  -50.0, # x0   > -50.0
                  -50.0, # y0   > -50.0
                    -10, # z0   > -10
          ] + [-l_short, -l_short, -10]*(u-1)
    ub = [            0, # A_ay < 0
                      0, # A_bx < 0
           l_long*sin30, # A_by < 5000.0*sin(30)
           l_long*cos30, # A_cx < 5000.0*cos(30)
           l_long*sin30, # A_cy < 5000.0*sin(30)
                 l_long, # A_dz < 10000.0
                   50.0, # x0   < 50.0
                   50.0, # y0   < 50.0
                l_short, # z0   < l_short
          ] + [l_short, l_short, 2*l_short]*(u-1)

    from mystic.termination import ChangeOverGeneration, NormalizedChangeOverGeneration, VTR
    from mystic.solvers import DifferentialEvolutionSolver2, PowellDirectionalSolver

    #pos_est0 = np.random.rand(u,3)*l_short - [l_short/2, l_short/2, 0]
    #pos_est0 = positions(5*5*5, 0, fuzz = 0)
    pos_est0 = np.zeros((u,3))
    #anchors_est = symmetric_anchors(l_anch)

#    anchors_est = np.array([[0.0, -2163.0, -75.5],
#                            [-1841.0, 741.0, -75.5],
#                            [1639.0, 1404.0, -75.5],
#                            [0.0, 0.0, 3250.5]])
    anchors_est = np.array([[0.0, 0.0, z],
                            [0.0, 0.0, z],
                            [0.0, 0.0, z],
                            [0.0, 0.0, 0]])
    x_guess0 = list(anchorsmatrix2vec(anchors_est)) + list(posmatrix2vec(pos_est0))

    print("Solver 0")
    solver0 = PowellDirectionalSolver(number_of_params_pos+number_of_params_anch)
    solver0.SetInitialPoints(x_guess0)
    solver0.SetStrictRanges(lb, ub)
    solver0.Solve(lambda x: costx(x[6:], x[0:6]), callback = cb)
    x_guess0 = solver0.bestSolution

    for i in range(1,20):
        print("Solver %d" % i)
        solver0 = PowellDirectionalSolver(number_of_params_pos+number_of_params_anch)
        solver0.SetInitialPoints(x_guess0)
        solver0.SetStrictRanges(lb, ub)
        solver0.Solve(lambda x: costx(x[6:], x[0:6]), callback = cb)
        x_guess0 = solver0.bestSolution

    #anch_0 = anchorsvec2matrix(solver0.bestSolution[0:6], z)

    #return anch_0
    return solver0.bestSolution

if __name__ == "__main__":
    z = -75.5
    # Gotten from manual measuring
    anchors = np.array([[0.0, -2163.0, -75.5],
                        [-1841.0, 741.0, -75.5],
                        [1639.0, 1404.0, -75.5],
                        [0.0, 0.0, 3250.5]])


    # Calculated, not measured
    # Gives exact solution up to e-8 precision
    #samp = np.array([[   0.        ,    0.        ,    0.        ,    0.        ],
    #                 [-160.05450141,   -7.41578381,  508.81184094, -566.25890613],
    #                 [-240.31902526,  -88.76779973,  449.05946137, -269.65276306],
    #                 [-275.807475  , -124.76718348,  422.99492029,   27.57111729],
    #                 [ 121.48001868, -147.47176573,  327.9627796 , -583.0761023 ],
    #                 [-182.63383271,  234.22786546,  299.45677071, -583.0761023 ],
    #                 [  51.43751   , -235.32008868,  263.75505139, -284.78756451],
    #                 [  20.69285473, -274.3991353 ,  235.67481709,   13.81466774],
    #                 [-263.85173603,  162.04723258,  234.48460575, -284.78756451],
    #                 [-299.78804098,  130.31868502,  206.05693659,   13.81466774],
    #                 [ 407.1938119 , -246.58073095,  171.82074196, -566.25890613],
    #                 [-160.05450141,  488.67753676,  110.5331805 , -566.25890613],
    #                 [ 345.13869583, -339.709284  ,  103.18644535, -269.65276306],
    #                 [ 101.70771773,  110.37864749,  101.92650368, -600.        ],
    #                 [ 318.03428538, -381.3285948 ,   73.08753928,   27.57111729],
    #                 [-240.31902526,  424.12957408,   39.987215  , -269.65276306],
    #                 [  31.03446609,   33.77678835,   31.10338123, -300.        ],
    #                 [-275.807475  ,  395.89476307,    9.01126285,   27.57111729],
    #                 [ 389.63442186,   24.02428293,  -71.04756549, -583.0761023 ],
    #                 [ 121.48001868,  378.19573436, -105.08271722, -583.0761023 ],
    #                 [ 327.14199005,  -56.00201336, -147.95175501, -284.78756451],
    #                 [ 299.8396334 ,  -91.37870646, -181.86930811,   13.81466774],
    #                 [  51.43751   ,  310.5445596 , -183.31098627, -284.78756451],
    #                 [  20.69285473,  280.89543594, -217.84612468,   13.81466774],
    #                 [ 407.1938119 ,  301.97210143, -297.18630107, -566.25890613],
    #                 [ 345.13869583,  231.99721017, -383.85809756, -269.65276306],
    #                 [ 318.03428538,  201.28360206, -422.37578312,   27.57111729]])

    # Measured3
    samp = np.array([[ 0.0     ,  0.0   ,  0.0    ,  0.0],
                     [ -159.88 ,  -7.42 ,  498.96 ,  -566.56],
                     [ -240.01 ,  -88.81 ,  449.46 ,  -269.41],
                     [ -275.44 ,  -124.74 ,  427.20 ,  27.56],
                     [ 121.50  ,  -147.40 ,  316.38 ,  -583.50],
                     [ -182.43 ,  234.08 ,  303.25 ,  -583.50],
                     [ 51.45   ,  -235.10 ,  261.35 ,  -284.52],
                     [ 20.70   ,  -274.11 ,  234.92 ,  13.81],
                     [ -263.50 ,  162.03 ,  238.27 ,  -284.52],
                     [ -299.37 ,  130.35 ,  212.02 ,  13.81],
                     [ 406.84  ,  -246.34 ,  163.49 ,  -566.57],
                     [ -159.88 ,  488.26 ,  124.25 ,  -566.56],
                     [ 344.89  ,  -339.30 ,  104.47 ,  -269.41],
                     [ 101.72  ,  110.40 ,  101.92 ,  -600.54],
                     [ 317.83  ,  -380.85 ,  72.50 ,  27.56],
                     [ -240.01 ,  423.67 ,  58.79 ,  -269.41],
                     [ 31.05   ,  33.78 ,  35.93 ,  -299.71],
                     [ -275.44 ,  395.48 ,  26.22 ,  27.56],
                     [ 389.31  ,  24.03 ,  -75.26 ,  -583.50]])

    # Measured 3 and compensated with constant -2
    #samp = np.array([[   0.  ,    0.  ,   0.  ,    0.  ],
    #                 [-159.88,   -7.42,  496.96, -566.56],
    #                 [-240.01,  -88.81,  447.46, -269.41],
    #                 [-275.44, -124.74,  425.2 ,   27.56],
    #                 [ 121.5 , -147.4 ,  314.38, -583.5 ],
    #                 [-182.43,  234.08,  301.25, -583.5 ],
    #                 [  51.45, -235.1 ,  259.35, -284.52],
    #                 [  20.7 , -274.11,  232.92,   13.81],
    #                 [-263.5 ,  162.03,  236.27, -284.52],
    #                 [-299.37,  130.35,  210.02,   13.81],
    #                 [ 406.84, -246.34,  161.49, -566.57],
    #                 [-159.88,  488.26,  122.25, -566.56],
    #                 [ 344.89, -339.3 ,  102.47, -269.41],
    #                 [ 101.72,  110.4 ,   99.92, -600.54],
    #                 [ 317.83, -380.85,   70.5 ,   27.56],
    #                 [-240.01,  423.67,   56.79, -269.41],
    #                 [  31.05,   33.78,   33.93, -299.71],
    #                 [-275.44,  395.48,   24.22,   27.56],
    #                 [ 389.31,   24.03,  -77.26, -583.5 ]])
    #Compensating -2 mm out of every C measurement reduced errors from
    #[[   0.         -170.74775148    0.        ]
    # [-229.31816957  150.55807494    0.        ]
    # [-260.75172296 -196.38194866    0.        ]
    # [   0.            0.         -373.52002034]]
    # to
    #[[   0.         -139.74915668    0.        ]
    # [-211.47544526  122.72584696    0.        ]
    # [-204.44639263 -140.1629383     0.        ]
    # [   0.            0.         -425.38964298]]
    # All coordinates except D_z got better

    # Measured 3 and compensated with -0.64x +  3.85
    #samp = np.array([[   0.        ,    0.        ,    0.        ,    0.        ],
    #             [-159.88      ,   -7.42      ,  502.1720535 , -566.56      ],
    #             [-240.01      ,  -88.81      ,  452.03203292, -269.41      ],
    #             [-275.44      , -124.74      ,  429.13201234,   27.56      ],
    #             [ 121.5       , -147.4       ,  317.67199176, -583.5       ],
    #             [-182.43      ,  234.08      ,  303.90197117, -583.5       ],
    #             [  51.45      , -235.1       ,  261.36195059, -284.52      ],
    #             [  20.7       , -274.11      ,  234.29193001,   13.81      ],
    #             [-263.5       ,  162.03      ,  237.00190943, -284.52      ],
    #             [-299.37      ,  130.35      ,  210.11188885,   13.81      ],
    #             [ 406.84      , -246.34      ,  160.94186826, -566.57      ],
    #             [-159.88      ,  488.26      ,  121.06184768, -566.56      ],
    #             [ 344.89      , -339.3       ,  100.6418271 , -269.41      ],
    #             [ 101.72      ,  110.4       ,   97.45180652, -600.54      ],
    #             [ 317.83      , -380.85      ,   67.39178594,   27.56      ],
    #             [-240.01      ,  423.67      ,   53.04176536, -269.41      ],
    #             [  31.05      ,   33.78      ,   29.54174477, -299.71      ],
    #             [-275.44      ,  395.48      ,   19.19172419,   27.56      ],
    #             [ 389.31      ,   24.03      ,  -82.92829639, -583.5       ]])

    #Compensating the linear measurment error reduced anchor errors from
    #[[   0.         -170.74775148    0.        ]
    # [-229.31816957  150.55807494    0.        ]
    # [-260.75172296 -196.38194866    0.        ]
    # [   0.            0.         -373.52002034]]
    # to
    #[[   0.          -81.27827747    0.        ]
    # [-197.95233601  127.92453157    0.        ]
    # [-120.13780415  -76.13893988    0.        ]
    # [   0.            0.         -600.36145758]]
    # All coordinates except D_z got better. D_z got 22.5 cm worse.

    u = np.shape(samp)[0]
    pos = np.zeros((u, 3))

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
    #scat_pos0 = ax_pos.scatter(pos[:,0], pos[:,1], pos[:,2], 'k+')
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
                anch = anchorsvec2matrix(x[0:6], z)
                scat_anch._offsets3d = (anch[:,0], anch[:,1], anch[:,2])
                print("Anchor errors: ")
                print(anchorsvec2matrix(x[0:6], z) - anchors)
                print("cost: %f" % \
                    cost(anchorsvec2matrix(x[0:6], z), np.reshape(x[6:], (u,3)), samp))

            elif len(x) == 6:
                anch = anchorsvec2matrix(x[0:6], z)
                scat_anch._offsets3d = (anch[:,0], anch[:,1], anch[:,2])
            else:
                ps = posvec2matrix(x, u)
                #scat_pos._offsets3d = (ps[:,0], ps[:,1], ps[:,2])
            plt.draw()
            plt.pause(0.001)
        iter += 1


    solution = solve(samp, replot, z = -75.5)
    sol_anch = anchorsvec2matrix(solution[0:6], z = -75.5)
    print("Output Anchors were: ")
    print(sol_anch)
    print("Anchor errors were: ")
    print(sol_anch - anchors)
    print("cost: %f" % \
    cost(anchorsvec2matrix(solution[0:6], z), np.reshape(solution[6:], (u,3)), samp))
    #print("Real cost were: %f" % cost(solution, pos, samp))
    #plt.pause(2000)
