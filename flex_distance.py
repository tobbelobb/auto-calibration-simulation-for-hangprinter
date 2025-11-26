import numpy as np

# Axes indexing
A = 0
B = 1
C = 2
D = 3
I = 4
X = 0
Y = 1
Z = 2


def _normalize_force_array(force, num_anchors, default_value):
    """Broadcast scalars/None to a per-anchor array."""
    if force is None:
        return np.full(num_anchors, default_value, dtype=float)
    arr = np.asarray(force, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.full(num_anchors, float(arr), dtype=float)
    if arr.shape[0] != num_anchors:
        raise ValueError(f"Expected {num_anchors} entries, got {arr.shape[0]}")
    return arr


def _build_direction_matrix(anchors, mover):
    """Return a 3xN matrix of unit direction vectors from mover to anchors."""
    num_anchors = anchors.shape[0]
    A_mat = np.zeros((3, num_anchors), dtype=float)
    for j in range(num_anchors):
        diff = anchors[j] - mover
        nrm = np.linalg.norm(diff)
        if nrm > 0.0:
            diff = diff / nrm
        else:
            diff = np.zeros(3, dtype=float)
        A_mat[:, j] = diff
    return A_mat


def _apply_A(A_mat, tensions):
    """Multiply direction matrix with tensions to get achieved force."""
    return A_mat @ tensions


def _solve_box_ridge_ls(A_mat, requested_force, lambda_reg, L, U, max_iters, tol):
    """
    Solve: min 0.5||A t - F||^2 + 0.5*lambda||t||^2 s.t. L <= t <= U
    Using a small active-set method mirroring the C++ reference.
    """
    A_mat = np.asarray(A_mat, dtype=float)
    requested_force = np.asarray(requested_force, dtype=float)
    num_anchors = A_mat.shape[1]

    H = A_mat.T @ A_mat
    H.flat[:: num_anchors + 1] += lambda_reg
    f = A_mat.T @ requested_force

    try:
        t = np.linalg.solve(H, f)
    except np.linalg.LinAlgError:
        t = np.zeros(num_anchors, dtype=float)
    t = np.clip(t, L, U)

    for _ in range(max_iters):
        g = H @ t - f

        # Projected gradient norm for convergence check
        pgn_sq = 0.0
        for i in range(num_anchors):
            gi = g[i]
            atL = t[i] <= L[i] + 1e-12
            atU = t[i] >= U[i] - 1e-12
            if (atL and gi > 0.0) or (atU and gi < 0.0):
                gi = 0.0
            pgn_sq += gi * gi
        if np.sqrt(pgn_sq) <= tol:
            break

        free_idx = []
        for i in range(num_anchors):
            atL = t[i] <= L[i] + 1e-12
            atU = t[i] >= U[i] - 1e-12
            violateL = atL and g[i] < -tol
            violateU = atU and g[i] > tol
            if (not atL and not atU) or violateL or violateU:
                free_idx.append(i)

        if not free_idx:
            best = -1.0
            free_idx.append(0)
            for i in range(num_anchors):
                atL = t[i] <= L[i] + 1e-12
                atU = t[i] >= U[i] - 1e-12
                viol = 0.0
                if atL:
                    viol = max(viol, -g[i])
                if atU:
                    viol = max(viol, g[i])
                if viol > best:
                    best = viol
                    free_idx[0] = i

        Hff = H[np.ix_(free_idx, free_idx)]
        gf = g[free_idx]
        try:
            pf = np.linalg.solve(Hff, -gf)
        except np.linalg.LinAlgError:
            pf = np.zeros_like(gf)

        alpha = 1.0
        for local_idx, anchor_idx in enumerate(free_idx):
            pi = pf[local_idx]
            if abs(pi) < 1e-16:
                continue
            if pi > 0.0:
                amax = (U[anchor_idx] - t[anchor_idx]) / pi
            else:
                amax = (L[anchor_idx] - t[anchor_idx]) / pi
            if amax < alpha:
                alpha = max(0.0, amax)

        t[free_idx] += alpha * pf
        t = np.clip(t, L, U)

    return t


def static_forces_qp(
    anchors,
    mover,
    min_force=None,
    max_force=None,
    *,
    ignore_gravity=False,
    ignore_pretension=False,
    mass_kg=0.0,
    g=9.81,
    lambda_reg=1e-3,
    tol=1e-3,
    max_iters_target=100,
):
    """Compute per-line tensions using the QP solver from the reference firmware."""
    anchors = np.asarray(anchors, dtype=float)
    mover = np.asarray(mover, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must be an (N, 3) array")
    if mover.shape != (3,):
        mover = mover.reshape(3)
    num_anchors = anchors.shape[0]

    max_force_arr = _normalize_force_array(max_force, num_anchors, np.inf)
    if min_force is None:
        if np.all(np.isinf(max_force_arr)):
            min_force_arr = np.zeros(num_anchors, dtype=float)
        else:
            min_force_arr = np.maximum(max_force_arr - 1.0, 0.0001)
    else:
        min_force_arr = _normalize_force_array(min_force, num_anchors, 0.0)

    if ignore_pretension:
        min_force_arr = np.zeros(num_anchors, dtype=float)

    max_force_arr = np.maximum(max_force_arr, min_force_arr)

    direction_matrix = _build_direction_matrix(anchors, mover)
    requested_force = np.array([0.0, 0.0, 0.0], dtype=float)
    if not ignore_gravity:
        requested_force[2] = mass_kg * g

    tensions = _solve_box_ridge_ls(
        direction_matrix,
        requested_force,
        lambda_reg,
        min_force_arr,
        max_force_arr,
        max_iters_target,
        tol,
    )

    achieved_force = _apply_A(direction_matrix, tensions)
    residual = requested_force - achieved_force
    supported_gravity_frac = 0.0
    if not ignore_gravity and requested_force[2] > 1e-9:
        supported_gravity_frac = achieved_force[2] / requested_force[2]

    return {
        "tensions": tensions,
        "achieved_force": achieved_force,
        "requested_force": requested_force,
        "residual": residual,
        "supported_gravity_frac": supported_gravity_frac,
    }


def flex_distance(
    anchors,
    pos,
    mechanical_advantage,
    springKPerUnitLength,
    mover_weight,
    min_force=None,
    max_force=None,
    *,
    ignore_gravity=False,
    ignore_pretension=False,
    lambda_reg=1e-3,
    tol=1e-3,
    max_iters_target=100,
    g=9.81,
    guy_wire_lengths=None,
):
    """
    Return flex compensation distances relative to the origin.

    The returned value can be added to geometric line deltas to get relaxed
    line lengths, matching the firmware's QP-based solver.
    """
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must be an (N, 3) array")
    pos = np.asarray(pos, dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
    mech_adv = np.asarray(mechanical_advantage, dtype=float)
    num_anchors = anchors.shape[0]
    if mech_adv.shape[0] != num_anchors:
        raise ValueError(f"Expected {num_anchors} mechanical advantage entries, got {mech_adv.shape[0]}")
    mech_adv_safe = np.maximum(mech_adv, 1e-9)
    guy_wire_lengths = (
        np.zeros(num_anchors, dtype=float)
        if guy_wire_lengths is None
        else _normalize_force_array(guy_wire_lengths, num_anchors, 0.0)
    )

    max_force_arr = _normalize_force_array(max_force, num_anchors, np.inf)
    if min_force is None:
        if np.all(np.isinf(max_force_arr)):
            min_force_arr = np.zeros(num_anchors, dtype=float)
        else:
            min_force_arr = np.maximum(max_force_arr - 1.0, 0.0001)
    else:
        min_force_arr = _normalize_force_array(min_force, num_anchors, 0.0)

    positions = np.vstack(([0.0, 0.0, 0.0], pos))
    flex_values = np.zeros((positions.shape[0], num_anchors), dtype=float)

    for idx, p in enumerate(positions):
        distances = np.linalg.norm(anchors - p, axis=1)
        spring_lengths = distances * mech_adv_safe + guy_wire_lengths
        safe_lengths = np.maximum(spring_lengths, 1e-9)
        springKs = springKPerUnitLength / safe_lengths
        springKs = np.maximum(springKs, 1e-9)

        forces = static_forces_qp(
            anchors,
            p,
            min_force_arr,
            max_force_arr,
            ignore_gravity=ignore_gravity,
            ignore_pretension=ignore_pretension,
            mass_kg=mover_weight,
            g=g,
            lambda_reg=lambda_reg,
            tol=tol,
            max_iters_target=max_iters_target,
        )["tensions"]

        flex_values[idx] = forces / (springKs * mech_adv_safe)

    return flex_values[0] - flex_values[1:]
