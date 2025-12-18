import numpy as np

A = 0
B = 1
C = 2
D = 3
I = 4


def _as_anchors(anchors) -> np.ndarray:
    a = np.asarray(anchors, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"anchors must have shape (N, 3); got {a.shape}")
    return a


def _as_line_positions(line_positions, n: int) -> np.ndarray:
    lp = np.asarray(line_positions, dtype=float).reshape(-1)
    if lp.size != n:
        raise ValueError(f"line_positions must have length {n}; got {lp.size}")
    return lp


def _initial_guess_lstsq(anchors: np.ndarray, distances: np.ndarray) -> np.ndarray:
    n = anchors.shape[0]
    if n < 4:
        return np.zeros(3, dtype=float)

    distances = np.maximum(distances, 1e-6)
    anchor_norm_sq = np.sum(np.square(anchors), axis=1)

    best = None
    best_cost = np.inf
    for ref in range(n):
        others = np.arange(n) != ref
        A = 2.0 * (anchors[others] - anchors[ref])
        b = (anchor_norm_sq[others] - anchor_norm_sq[ref]) - (np.square(distances[others]) - np.square(distances[ref]))
        try:
            guess, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 3 or not np.all(np.isfinite(guess)):
            continue
        err = np.linalg.norm(anchors - guess, axis=1) - distances
        cost = float(np.sum(np.square(err)))
        if cost < best_cost:
            best = guess
            best_cost = cost

    if best is None:
        return np.zeros(3, dtype=float)
    return best.astype(float)


def _residuals_and_derivatives(
    anchors: np.ndarray,
    origin_lengths: np.ndarray,
    line_positions_rel: np.ndarray,
    pos: np.ndarray,
    *,
    compute_flex=None,
    want_hessians: bool,
    impact_step: float = 1e-3,
):
    n = anchors.shape[0]

    if compute_flex is None:
        base_impact = np.zeros(n, dtype=float)
        impact_deriv = None
    else:
        base_impact = np.asarray(compute_flex(pos), dtype=float).reshape(-1)
        if base_impact.size != n:
            raise ValueError(f"compute_flex must return length {n}; got {base_impact.size}")

        impact_deriv = np.zeros((n, 3), dtype=float)
        for axis in range(3):
            shifted = pos.copy()
            shifted[axis] += impact_step
            plus = np.asarray(compute_flex(shifted), dtype=float).reshape(-1)
            shifted[axis] -= 2.0 * impact_step
            minus = np.asarray(compute_flex(shifted), dtype=float).reshape(-1)
            if plus.size != n or minus.size != n:
                raise ValueError(f"compute_flex must return length {n}; got {plus.size} and {minus.size}")
            impact_deriv[:, axis] = (plus - minus) / (2.0 * impact_step)

    diff = pos[None, :] - anchors
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-6)
    inv_len = 1.0 / lengths
    inv_len3 = inv_len * inv_len * inv_len

    found_line_pos = lengths - origin_lengths + base_impact
    residuals = found_line_pos - line_positions_rel

    J = diff * inv_len[:, None]
    if impact_deriv is not None:
        J = J + impact_deriv

    if want_hessians:
        H = np.zeros((n, 3, 3), dtype=float)
        for i in range(n):
            H[i] = np.eye(3, dtype=float) * inv_len[i] - np.outer(diff[i], diff[i]) * inv_len3[i]
    else:
        H = None

    cost = 0.5 * float(np.sum(np.square(residuals)))
    return residuals, J, H, cost


def forward_transform(
    anchors,
    line_positions,
    *,
    line_positions_are_absolute: bool = False,
    seed=None,
    eta: float = 1.0e-3,
    tol: float = 1.0e-3,
    halley_iters: int = 3,
    max_iters: int = 30,
    compute_flex=None,
    return_spread: bool = False,
):
    """
    Quadratic (Halley) forward transform for Hangprinter-like kinematics.

    Args:
        anchors: (N, 3) anchor positions in mm.
        line_positions: length-N array; by default, line lengths relative to the origin lengths (mm).
        line_positions_are_absolute: when True, `line_positions` are absolute anchor distances (mm).
        seed: optional initial (3,) guess in mm; defaults to a multilateration least-squares seed.
        compute_flex: optional callback(pos)->(N,) extra line position (mm) (e.g. stretch compensation).
        return_spread: when True, return (pos, spread) where spread is sum of squared residuals (mm^2).
    """

    anchors = _as_anchors(anchors)
    n = anchors.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 anchors; got {n}")

    lp = _as_line_positions(line_positions, n)
    origin_lengths = np.linalg.norm(anchors, axis=1)
    if line_positions_are_absolute:
        distances = lp
        line_positions_rel = distances - origin_lengths
    else:
        line_positions_rel = lp
        distances = line_positions_rel + origin_lengths

    distances = np.maximum(distances, 1e-6)

    if seed is None:
        pos = _initial_guess_lstsq(anchors, distances)
    else:
        pos = np.asarray(seed, dtype=float).reshape(3)

    converged = False
    iterations = 0

    eta = float(eta)
    tol = float(tol)
    halley_iters = int(halley_iters)
    max_iters = int(max_iters)

    for i in range(min(halley_iters, max_iters)):
        residuals, J, H, _ = _residuals_and_derivatives(
            anchors,
            origin_lengths,
            line_positions_rel,
            pos,
            compute_flex=compute_flex,
            want_hessians=True,
        )

        JTJ = J.T @ J
        JTJ.flat[::4] += eta
        grad = J.T @ residuals
        try:
            delta_lm = np.linalg.solve(JTJ, -grad)
        except np.linalg.LinAlgError:
            break

        Hbar = np.einsum("nij,j->ni", H, delta_lm)
        Jbar = J + 0.5 * Hbar

        JTJ2 = Jbar.T @ Jbar
        JTJ2.flat[::4] += eta
        grad2 = Jbar.T @ residuals
        try:
            delta = np.linalg.solve(JTJ2, -grad2)
        except np.linalg.LinAlgError:
            break

        pos = pos + delta
        iterations = i + 1
        if float(np.linalg.norm(delta)) < tol:
            converged = True
            break

    if not converged:
        for j in range(iterations, max_iters):
            residuals, J, _, _ = _residuals_and_derivatives(
                anchors,
                origin_lengths,
                line_positions_rel,
                pos,
                compute_flex=compute_flex,
                want_hessians=False,
            )
            JTJ = J.T @ J
            JTJ.flat[::4] += eta
            grad = J.T @ residuals
            try:
                delta = np.linalg.solve(JTJ, -grad)
            except np.linalg.LinAlgError:
                break

            pos = pos + delta
            iterations = j + 1
            if float(np.linalg.norm(delta)) < tol:
                converged = True
                break

    residuals, _, _, _ = _residuals_and_derivatives(
        anchors,
        origin_lengths,
        line_positions_rel,
        pos,
        compute_flex=compute_flex,
        want_hessians=False,
    )
    spread = float(np.sum(np.square(residuals)))

    if return_spread:
        return pos, spread
    return pos
# }
