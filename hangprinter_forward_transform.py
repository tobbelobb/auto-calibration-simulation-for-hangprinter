import numpy as np

A = 0
B = 1
C = 2
D = 3
I = 4
X = 0
Y = 1
Z = 2


def det(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return a * (e * i - f * h) + b * (f * g - i * d) + c * (d * h - e * g)


def singular_3x3(matrix):
    threshold = 1e-1
    return abs(det(matrix)) < threshold


def forward_transform5(anchors, line_length_samp):
    anch_prim = anchors[:-1] - anchors[-1:]
    norms = np.linalg.norm(anchors, 2, 1)
    norms_sq = norms * norms
    l = line_length_samp + norms
    l_sq = l * l
    k = ((norms_sq[:-1] - norms_sq[-1]) - (l_sq[:-1] - l_sq[-1])) / 2.0

    m0 = anch_prim[[0, 1, 2]]
    m1 = anch_prim[[0, 1, 3]]
    m2 = anch_prim[[0, 2, 3]]
    m3 = anch_prim[[1, 2, 3]]

    p = np.array([0.0, 0.0, 0.0])
    ps = np.array([])
    c = 0
    if not singular_3x3(m0):
        p0 = np.linalg.solve(m0, k[[0, 1, 2]])
        ps = np.append(ps, p0)
        p += p0
        c += 1
    if not singular_3x3(m1):
        p1 = np.linalg.solve(m1, k[[0, 1, 3]])
        ps = np.append(ps, p1)
        p += p1
        c += 1
    if not singular_3x3(m2):
        p2 = np.linalg.solve(m2, k[[0, 2, 3]])
        ps = np.append(ps, p2)
        p += p2
        c += 1
    if not singular_3x3(m3):
        p3 = np.linalg.solve(m3, k[[1, 2, 3]])
        ps = np.append(ps, p3)
        p += p3
        c += 1

    spread = 0.0
    if c != 0:
        p /= c
        # p is a centroid point. However, since we have 5 lines, it's not for sure that we've gotten a valid set
        # of line lenghts. Since we've computed 4 different points, we can measure how far they are apart.
        # If they're all at the same point, then the line lengths were an entirely valid set of 5 points.
        # The further apart our 4 different points are, the less valid the line lengths were.
        # The spread tells the caller how many grains of salt they should interpret the returned p with.
        ps = ps.reshape((c, 3))
        diff = ps - p
        spread = np.sum(diff*diff)


    return p, spread


def forward_transform(anchors, line_length_samp):
    p0 = forward_transform_(anchors[(0, 1, 2, 4), :], line_length_samp[[0, 1, 2, 4]])
    p1 = forward_transform_(anchors[(1, 2, 3, 4), :], line_length_samp[[1, 2, 3, 4]])
    p2 = forward_transform_(anchors[(2, 3, 0, 4), :], line_length_samp[[2, 3, 0, 4]])
    p3 = forward_transform_(anchors[(3, 0, 1, 4), :], line_length_samp[[3, 0, 1, 4]])

    p = (p0 + p1 + p2 + p3) / 4.0
    ps = np.array([p0, p1, p2, p3])
    diff = ps - p
    spread = np.sum(diff*diff)

    return p, spread


# Doesn't do well when anchors B or C are exactly on the X-axis...
def forward_transform_(anchors, line_length_samp):
    distances_origin = np.linalg.norm(anchors, 2, 1)
    a = line_length_samp[A] + distances_origin[A]
    b = line_length_samp[B] + distances_origin[B]
    c = line_length_samp[C] + distances_origin[C]
    d = line_length_samp[D] + distances_origin[D]

    anchors_tmp0 = anchors
    # Force the anchor location norms Ax=0, Dx=0, Dy=0
    # through a series of rotations.
    x_angle = np.arctan2(anchors[D][Y], anchors[D][Z])
    rxt = np.array([[1, 0, 0], [0, np.cos(x_angle), np.sin(x_angle)], [0, -np.sin(x_angle), np.cos(x_angle)]])
    anchors_tmp0 = np.matmul(anchors, rxt)

    y_angle = np.arctan2(-anchors_tmp0[D][X], anchors_tmp0[D][Z])
    ryt = np.array([[np.cos(y_angle), 0, -np.sin(y_angle)], [0, 1, 0], [np.sin(y_angle), 0, np.cos(y_angle)]])
    anchors_tmp1 = np.matmul(anchors_tmp0, ryt)

    z_angle = np.arctan2(anchors_tmp1[A][X], anchors_tmp1[A][Y])
    rzt = np.array([[np.cos(z_angle), np.sin(z_angle), 0], [-np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])
    anchors_tmp0 = np.matmul(anchors_tmp1, rzt)

    Asq = distances_origin[A] * distances_origin[A]
    Bsq = distances_origin[B] * distances_origin[B]
    Csq = distances_origin[C] * distances_origin[C]
    Dsq = distances_origin[D] * distances_origin[D]
    aa = a * a
    dd = d * d

    k0b = (-b * b + Bsq - Dsq + dd) / (2.0 * anchors_tmp0[B][X]) + (
        anchors_tmp0[B][Y] / (2.0 * anchors_tmp0[A][Y] * anchors_tmp0[B][X])
    ) * (Dsq - Asq + aa - dd)
    k0c = (-c * c + Csq - Dsq + dd) / (2.0 * anchors_tmp0[C][X]) + (
        anchors_tmp0[C][Y] / (2.0 * anchors_tmp0[A][Y] * anchors_tmp0[C][X])
    ) * (Dsq - Asq + aa - dd)
    k1b = (anchors_tmp0[B][Y] * (anchors_tmp0[A][Z] - anchors_tmp0[D][Z])) / (
        anchors_tmp0[A][Y] * anchors_tmp0[B][X]
    ) + (anchors_tmp0[D][Z] - anchors_tmp0[B][Z]) / anchors_tmp0[B][X]
    k1c = (anchors_tmp0[C][Y] * (anchors_tmp0[A][Z] - anchors_tmp0[D][Z])) / (
        anchors_tmp0[A][Y] * anchors_tmp0[C][X]
    ) + (anchors_tmp0[D][Z] - anchors_tmp0[C][Z]) / anchors_tmp0[C][X]

    machinePos_tmp0 = np.zeros(3)
    if abs(k1c - k1b) > 0.000001:
        machinePos_tmp0[Z] = (k0b - k0c) / (k1c - k1b)
    else:
        return machinePos_tmp0
    machinePos_tmp0[X] = k0c + k1c * machinePos_tmp0[Z]
    machinePos_tmp0[Y] = (Asq - Dsq - aa + dd) / (2.0 * anchors_tmp0[A][Y]) + (
        (anchors_tmp0[D][Z] - anchors_tmp0[A][Z]) / anchors_tmp0[A][Y]
    ) * machinePos_tmp0[Z]

    # Rotate machinePos_tmp back to original coordinate system
    machinePos_tmp1 = np.matmul(rzt, machinePos_tmp0)
    machinePos_tmp0 = np.matmul(ryt, machinePos_tmp1)
    machinePos_tmp1 = np.matmul(rxt, machinePos_tmp0)

    return machinePos_tmp1


# void HangprinterKinematics::ForwardTransform(float const a, float const b, float const c, float const d, float machinePos[3]) const noexcept
# 	for (size_t row{0}; row < 3; ++row) {
# 		machinePos[row] = rxt[row][0]*machinePos_tmp0[0] + rxt[row][1]*machinePos_tmp0[1] + rxt[row][2]*machinePos_tmp0[2];
# 	}
# }
