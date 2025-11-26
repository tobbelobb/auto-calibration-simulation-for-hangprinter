#!/usr/bin/env python3
"""Construct intermediate JSONL datasets for multiple Hangprinter geometries.

Outputs go to synthetic_datasets/intermediate/<scenario>_<geometry>.jsonl
with empty motor_samples ready for populate via generate_synthetic_data.py.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from data import xyz_of_samp

RNG = np.random.default_rng(12345)


def hp5_anchor_sets() -> List[np.ndarray]:
    """Canonical HP5 anchors plus two size variants."""
    def make(l_xy: float, base_z: float, top_z: float) -> np.ndarray:
        return np.array(
            [
                [0.0, -l_xy, base_z],
                [l_xy, 0.0, base_z],
                [0.0, l_xy, base_z],
                [-l_xy, 0.0, base_z],
                [0.0, 0.0, top_z],
            ]
        )

    return [
        make(2000.0, -120.0, 2000.0),
        make(2500.0, -150.0, 2300.0),
        make(1800.0, -100.0, 2100.0),
    ]


def slideprinter_anchor_sets() -> List[np.ndarray]:
    """Three planar triangles centered near origin on z=0."""
    base = np.array([[-1000.0, -577.0, 0.0], [1000.0, -577.0, 0.0], [0.0, 1154.0, 0.0]])
    return [base * s for s in (1.0, 1.2, 1.5)]


def spidercam_anchor_sets() -> List[np.ndarray]:
    """Four high anchors forming rectangular footprints."""
    def make(span: float, height: float) -> np.ndarray:
        return np.array(
            [
                [span, span, height],
                [-span, span, height],
                [-span, -span, height],
                [span, -span, height],
            ]
        )

    return [make(1500.0, 3500.0), make(1800.0, 4000.0), make(1200.0, 3200.0)]


def cubecorners_anchor_sets() -> List[np.ndarray]:
    """Spidercam anchors plus matching low anchors at z=0."""
    sets = []
    for high in spidercam_anchor_sets():
        low = high.copy()
        low[:, 2] = 0.0
        sets.append(np.vstack((high, low)))
    return sets


def _hp5_boundary_l1(anchors: np.ndarray, z: float) -> float:
    top_z = anchors[:, 2].max()
    base_z = anchors[anchors[:, 2] < top_z][:, 2].mean()
    base_l1 = np.max(np.abs(anchors[:, 0]) + np.abs(anchors[:, 1]))
    t = np.clip((top_z - z) / max(top_z - base_z, 1e-6), 0.0, 1.0)
    return base_l1 * t


def _hp5_to_offset(points: np.ndarray, anchors: np.ndarray, delta: float) -> np.ndarray:
    out = points.copy()
    for i, p in enumerate(points):
        bound_l1 = _hp5_boundary_l1(anchors, p[2])
        cur_l1 = abs(p[0]) + abs(p[1])
        target_l1 = max(bound_l1 + delta, 0.0)
        if cur_l1 < 1e-9:
            out[i, 0] = target_l1
            out[i, 1] = 0.0
            continue
        scale = target_l1 / cur_l1
        out[i, 0] = p[0] * scale
        out[i, 1] = p[1] * scale
    return out


def _triangle_ray_to_boundary(point: np.ndarray, verts: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return boundary point along centroid->point ray for a CCW triangle."""
    centroid = verts.mean(axis=0)
    d = point - centroid
    if np.linalg.norm(d) < 1e-9:
        d = np.array([1.0, 0.0])
    t_max = np.inf
    for i in range(3):
        a = verts[i, :2]
        b = verts[(i + 1) % 3, :2]
        edge = b - a
        normal = np.array([edge[1], -edge[0]])
        if normal.dot(centroid[:2] - a) > 0:
            normal = -normal
        denom = normal.dot(d)
        if denom > 1e-12:
            t_edge = -normal.dot(centroid[:2] - a) / denom
            t_max = min(t_max, t_edge)
    return centroid[:2] + t_max * d, t_max, d


def _slide_to_offset(points: np.ndarray, anchors: np.ndarray, delta: float) -> np.ndarray:
    out = points.copy()
    verts = anchors[:, :2]
    for i, p in enumerate(points):
        boundary, t_max, d = _triangle_ray_to_boundary(p[:2], verts)
        d_norm = np.linalg.norm(d)
        step = delta / max(d_norm, 1e-9)
        t_target = t_max + step
        out[i, :2] = verts.mean(axis=0) + t_target * d
        out[i, 2] = 0.0
    return out


def _rectangle_to_offset(points: np.ndarray, anchors: np.ndarray, delta: float) -> np.ndarray:
    out = points.copy()
    min_x, max_x = anchors[:, 0].min(), anchors[:, 0].max()
    min_y, max_y = anchors[:, 1].min(), anchors[:, 1].max()
    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    half_w = (max_x - min_x) / 2.0
    half_h = (max_y - min_y) / 2.0
    for i, p in enumerate(points):
        d = p[:2] - center
        if np.linalg.norm(d) < 1e-9:
            d = np.array([1.0, 0.0])
        tx = np.inf
        if abs(d[0]) > 1e-9:
            tx = (half_w * np.sign(d[0])) / d[0]
        ty = np.inf
        if abs(d[1]) > 1e-9:
            ty = (half_h * np.sign(d[1])) / d[1]
        t_max = min(tx, ty)
        d_norm = np.linalg.norm(d)
        step = delta / max(d_norm, 1e-9)
        t_target = t_max + step
        out[i, :2] = center + t_target * d
    return out


def _hp5_random_positions(n: int, anchors: np.ndarray) -> np.ndarray:
    top_z = anchors[:, 2].max()
    base_z = anchors[anchors[:, 2] < top_z][:, 2].mean()
    base_l1 = np.max(np.abs(anchors[:, 0]) + np.abs(anchors[:, 1]))
    res = []
    for _ in range(n):
        z = RNG.uniform(base_z + 50.0, top_z - 300.0)
        t = (top_z - z) / max(top_z - base_z, 1e-6)
        limit = base_l1 * t
        x = RNG.uniform(-limit, limit)
        y_lim = limit - abs(x)
        y = RNG.uniform(-y_lim, y_lim)
        res.append([x, y, z])
    return np.array(res)


def _slide_random_positions(n: int, anchors: np.ndarray) -> np.ndarray:
    verts = anchors[:, :2]
    res = []
    for _ in range(n):
        w = RNG.dirichlet([1.0, 1.0, 1.0])
        xy = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
        res.append([xy[0], xy[1], 0.0])
    return np.array(res)


def _rect_random_positions(n: int, anchors: np.ndarray, z_min: float) -> np.ndarray:
    min_x, max_x = anchors[:, 0].min(), anchors[:, 0].max()
    min_y, max_y = anchors[:, 1].min(), anchors[:, 1].max()
    top_z = anchors[:, 2].max()
    z_hi = top_z - 600.0
    res = []
    for _ in range(n):
        x = RNG.uniform(min_x + 200.0, max_x - 200.0)
        y = RNG.uniform(min_y + 200.0, max_y - 200.0)
        z = RNG.uniform(z_min, z_hi)
        res.append([x, y, z])
    return np.array(res)


def _clean_positions(geometry: str, anchors: np.ndarray) -> np.ndarray:
    base = np.array(xyz_of_samp, dtype=float)
    if geometry == "HP5":
        return base
    if geometry == "Slideprinter":
        return np.column_stack((base[:, 0], base[:, 1], np.zeros(base.shape[0])))
    z_min = 200.0
    z_hi = anchors[:, 2].min() - 500.0
    clipped_z = np.clip(base[:, 2], z_min, z_hi)
    return np.column_stack((base[:, 0], base[:, 1], clipped_z))


def _scenario_positions(scenario: str, geometry: str, anchors: np.ndarray) -> Tuple[np.ndarray, Dict]:
    cfg: Dict = {}
    if scenario == "clean_baseline":
        return _clean_positions(geometry, anchors), cfg
    if scenario == "larger_baseline":
        if geometry == "HP5":
            return _hp5_random_positions(10, anchors), cfg
        if geometry == "Slideprinter":
            return _slide_random_positions(10, anchors), cfg
        return _rect_random_positions(10, anchors, 200.0), cfg
    if scenario in ("near_singularities", "at_singularities", "outside_singularities"):
        delta = 0.1
        if scenario == "at_singularities":
            delta = 0.0
        if scenario == "near_singularities":
            delta = -0.1
        base = _clean_positions(geometry, anchors)
        if geometry == "HP5":
            return _hp5_to_offset(base, anchors, delta), cfg
        if geometry == "Slideprinter":
            return _slide_to_offset(base, anchors, delta), cfg
        return _rectangle_to_offset(base, anchors, delta), cfg
    if scenario == "systematic_bias":
        cfg["min_force"] = 50.0
        return _clean_positions(geometry, anchors), cfg
    raise ValueError(f"Unknown scenario {scenario}")


def _write_jsonl(path: Path, entries: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


def main() -> None:
    geometries = {
        "HP5": hp5_anchor_sets(),
        "Slideprinter": slideprinter_anchor_sets(),
        "Spidercam": spidercam_anchor_sets(),
        "CubeCorners": cubecorners_anchor_sets(),
    }
    scenarios = [
        "clean_baseline",
        "larger_baseline",
        "near_singularities",
        "at_singularities",
        "outside_singularities",
        "systematic_bias",
    ]

    out_root = Path("synthetic_datasets") / "intermediate"
    files: Dict[Tuple[str, str], List[Dict]] = {}

    for scenario in scenarios:
        for geom, anchor_sets in geometries.items():
            # Only larger_baseline needs all three anchor sets; others use the first one.
            anchors_to_use = anchor_sets if scenario == "larger_baseline" else anchor_sets[:1]
            for idx, anchors in enumerate(anchors_to_use):
                pos, cfg = _scenario_positions(scenario, geom, anchors)
                entry = {
                    "dataset": scenario,
                    "geometry": geom,
                    "anchor_set": idx,
                    "anchors": anchors.tolist(),
                    "real_xyz": pos.tolist(),
                    "config": cfg,
                    "motor_samples": [],
                }
                key = (scenario, geom)
                files.setdefault(key, []).append(entry)

    for (scenario, geom), entries in files.items():
        path = out_root / f"{scenario}_{geom}.jsonl"
        _write_jsonl(path, entries)
        print(f"Wrote {len(entries)} entries to {path}")


if __name__ == "__main__":
    main()
