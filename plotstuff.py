#!/usr/bin/python3

from __future__ import division
import numpy as np

from flex_distance import static_forces_qp

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def anchor_labels(count):
    base = ["A", "B", "C", "D"]
    if count <= len(base):
        return base[:count]
    return base + ["I"]


if __name__ == "__main__":
    # Match the hangprinter-flex-compensation reference
    anchors = np.array(
        [
            [16.4, -1610.98, -131.53],
            [1314.22, 128.14, -121.28],
            [-15.73, 1415.61, -121.82],
            [-1211.62, 18.14, -111.18],
            [10.0, -10.0, 2299.83],
        ]
    )

    mass_kg = 1.0
    g = 9.81
    min_abcd = 3.0
    max_abcd = 120.0
    min_I = 3.0
    max_I = 120.0

    min_force = np.array([min_abcd, min_abcd, min_abcd, min_abcd, min_I])
    max_force = np.array([max_abcd, max_abcd, max_abcd, max_abcd, max_I])

    min_xy = -1611.0
    max_xy = 1601.0
    step = 35.0

    grid = np.arange(min_xy, max_xy, step)
    X, Y = np.meshgrid(grid, grid)
    flattened = np.c_[X.ravel(), Y.ravel(), np.zeros_like(X).ravel()]

    tensions = np.zeros((flattened.shape[0], anchors.shape[0]))
    supported_frac = np.zeros(flattened.shape[0])
    residual_z = np.zeros(flattened.shape[0])

    for idx, mover in enumerate(flattened):
        res = static_forces_qp(
            anchors,
            mover,
            min_force,
            max_force,
            mass_kg=mass_kg,
            g=g,
        )
        tensions[idx] = res["tensions"]
        supported_frac[idx] = res["supported_gravity_frac"]
        residual_z[idx] = res["residual"][2]

    Z = [tensions[:, i].reshape(X.shape) for i in range(anchors.shape[0])]
    Z_frac = supported_frac.reshape(X.shape)
    Z_res = residual_z.reshape(X.shape)

    labels = anchor_labels(anchors.shape[0])
    axes = []
    for i, label in enumerate(labels):
        ax = plt.figure().add_subplot(111, projection="3d", title=f"{label} (QP, plane XY)")
        ax.plot_surface(X, Y, Z[i], cmap=cm.viridis)
        axes.append(ax)

    ax_frac = plt.figure().add_subplot(111, projection="3d", title="Supported gravity fraction (QP, plane XY)")
    ax_frac.plot_surface(X, Y, Z_frac, cmap=cm.coolwarm)

    ax_res = plt.figure().add_subplot(111, projection="3d", title="Gravity residual Z (QP, plane XY)")
    ax_res.plot_surface(X, Y, Z_res, cmap=cm.coolwarm)

    plt.show()
