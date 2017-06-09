"""Simulation of Hangprinter auto-calibration
"""
from __future__ import division # Always want 3/2 = 1.5
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    |sqrt(sum for s from x to z (A_sk-s_i)^2) - sqrt(sum for s from x to z (A_sk-s_0)^2) - t_ik|

    or...
    sum for i from 1 to u
    |sqrt((A_xa-x_i)^2 + (A_ya-y_i)^2 + (A_za-z_i)^2) - sqrt((A_xa-x_0)^2 + (A_ya-y_0)^2 + (A_za-z_0)^2) - t_ib| +
    |sqrt((A_xb-x_i)^2 + (A_yb-y_i)^2 + (A_zb-z_i)^2) - sqrt((A_xb-x_0)^2 + (A_yb-y_0)^2 + (A_zb-z_0)^2) - t_ib| +
    |sqrt((A_xc-x_i)^2 + (A_yc-y_i)^2 + (A_zc-z_i)^2) - sqrt((A_xc-x_0)^2 + (A_yc-y_0)^2 + (A_zc-z_0)^2) - t_ic| +
    |sqrt((A_xd-x_i)^2 + (A_yd-y_i)^2 + (A_zd-z_i)^2) - sqrt((A_xd-x_0)^2 + (A_yd-y_0)^2 + (A_zd-z_0)^2) - t_id|

    Parameters
    ---------
    anchors : 4x3 matrix of anchor positions
    pos: ux3 matrix of positions
    samp : ux4 matrix of corresponding samples, starting with [0., 0., 0., 0.]
    """
    return np.sum(np.abs(samples(anchors, pos, fuzz = 0) - samp))

# Do basic testing and output results
# if this is run like a script
# and not imported like a module or package
if __name__ == "__main__":
    l_long = 1500
    l_short = 150
    n = 5
    sample_fuzz = 5
    anchors = irregular_anchors(l_long)
    pos = positions(n, l_short, fuzz = 5)
    samp = samples(anchors, pos, fuzz = sample_fuzz)

    cost = cost(anchors, pos, samp)
    print("Cost was %f" % cost)
    eps = 3*(n**3)*sample_fuzz # Linear effect of fuzz seems about right
    print("eps was %f" % eps)
    if cost > eps:
        print("Test fail")
    else:
        print("Test success")

# TODO:
#  * mystic

