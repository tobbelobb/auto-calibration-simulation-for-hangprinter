#!/usr/bin/python3

from __future__ import division
import numpy as np

from flex_distance import *
from simulation import *

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == "__main__":
    low_axis_max_force = 120
    low_axis_target_force = 20

    # Match the hangprinter-flex-compensation flex.hpp defined anchors
    anchors = np.array(
        [[16.4, -1610.98, -131.53], [1314.22, 128.14, -121.28], [-15.73, 1415.61, -121.82], [-1211.62, 18.14, -111.18], [10.0, -10.0, 2299.83]]
    )
    min_xy = -1611
    max_xy = 1601
    steps = 50
    X, Y = np.meshgrid(np.linspace(min_xy, max_xy, steps), np.linspace(min_xy, max_xy, steps))
    pos = np.c_[np.c_[np.ravel(X)[:, np.newaxis], np.ravel(Y)], np.zeros(steps * steps)]

    mechanical_advantage = np.array([2.0, 2.0, 2.0, 2.0, 4.0])
    springKPerUnitLength = 20000
    mover_weight = 1.0
    dist = flex_distance(
        low_axis_max_force,
        low_axis_target_force,
        anchors,
        pos,
        mechanical_advantage,
        springKPerUnitLength,
        mover_weight,
    )
    ZA = dist[:, 0]
    ZB = dist[:, 1]
    ZC = dist[:, 2]
    ZD = dist[:, 3]
    ZI = dist[:, 4]
    ZA = ZA.reshape(X.shape)
    ZB = ZB.reshape(X.shape)
    ZC = ZC.reshape(X.shape)
    ZD = ZD.reshape(X.shape)
    ZI = ZI.reshape(X.shape)

    axA = plt.figure().add_subplot(111, projection="3d")
    #axB = plt.figure().add_subplot(111, projection="3d")
    #axC = plt.figure().add_subplot(111, projection="3d")
    #axD = plt.figure().add_subplot(111, projection="3d")
    #axI = plt.figure().add_subplot(111, projection="3d")
    axA.plot_surface(X, Y, ZA, cmap=cm.coolwarm)
    #axB.plot_surface(X, Y, ZB, cmap=cm.coolwarm)
    #axC.plot_surface(X, Y, ZC, cmap=cm.coolwarm)
    #axD.plot_surface(X, Y, ZD, cmap=cm.coolwarm)
    #axI.plot_surface(X, Y, ZI, cmap=cm.coolwarm)

    plt.show()
