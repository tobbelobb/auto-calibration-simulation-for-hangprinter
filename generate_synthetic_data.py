#!/usr/bin/env python3
"""Fill synthetic motor samples for Hangprinter dataset JSONL files.

Each input line is expected to be a JSON object with at least:
  - "anchors": list of [x, y, z] anchor coordinates in millimeters
  - "real_xyz": list of mover poses, one [x, y, z] per sample
  - "config": optional dict overriding generation settings
  - "motor_samples": optional; left empty in intermediate files

Values placed under "config" (all optional):
  - spool_r
  - spool_buildup_factor
  - spool_gear_teeth
  - motor_gear_teeth
  - mechanical_advantage
  - lines_per_spool
  - spring_k_per_unit_length
  - mover_weight
  - min_force
  - max_force
  - guy_wire_lengths
  - use_flex (bool)
  - ignore_gravity (bool)
  - ignore_pretension (bool)

Outputs the same JSONL file shape, with "motor_samples" populated as
rotations in degrees (360 deg per full rotation), aligned with the
order of "real_xyz".
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from data import (
    constant_spool_buildup_factor,
    lines_per_spool,
    mechanical_advantage,
    motor_gear_teeth,
    mover_weight,
    spool_gear_teeth,
    spool_r_in_origin_first_guess,
    springKPerUnitLength,
)
from util import pos_to_motor_pos_samples

DEFAULT_MIN_FORCE = 3.0
DEFAULT_MAX_FORCE = 120.0


def _expand(
    value: Any,
    default: np.ndarray,
    count: int,
    name: str,
) -> np.ndarray:
    """
    Normalize scalars/lists to a per-anchor float array.

    If the provided array is shorter than needed, pad with its last
    value. If it's longer, truncate. A scalar is broadcast.
    """
    if value is None:
        arr = np.asarray(default, dtype=float).reshape(-1)
    else:
        arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size == 1:
        return np.full(count, float(arr[0]), dtype=float)
    if arr.size == count:
        return arr.astype(float)
    if arr.size > count:
        return arr[:count].astype(float)

    pad_width = count - arr.size
    if pad_width < 0:
        raise ValueError(f"Unexpected negative pad width for {name}")
    return np.pad(arr, (0, pad_width), mode="edge").astype(float)


def _normalize_force(value: Any, default: float, count: int, name: str) -> Optional[np.ndarray]:
    """Return per-anchor force array or None if value is explicitly None."""
    if value is None:
        value = default
    return _expand(value, np.array([value], dtype=float), count, name)


def _load_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def _synthesize(entry: Dict[str, Any]) -> Dict[str, Any]:
    anchors = np.asarray(entry["anchors"], dtype=float)
    poses = np.asarray(entry["real_xyz"], dtype=float)

    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must be an (N, 3) array")
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("real_xyz must be an (M, 3) array")

    cfg = entry.get("config", {})
    spool_buildup = cfg.get("spool_buildup_factor", constant_spool_buildup_factor)
    spool_r = _expand(cfg.get("spool_r"), spool_r_in_origin_first_guess, anchors.shape[0], "spool_r")

    spool_to_motor = float(cfg.get("spool_gear_teeth", spool_gear_teeth)) / float(
        cfg.get("motor_gear_teeth", motor_gear_teeth)
    )
    mech_adv = _expand(cfg.get("mechanical_advantage"), mechanical_advantage, anchors.shape[0], "mechanical_advantage")
    lines = _expand(cfg.get("lines_per_spool"), lines_per_spool, anchors.shape[0], "lines_per_spool")
    min_force = _normalize_force(cfg.get("min_force"), DEFAULT_MIN_FORCE, anchors.shape[0], "min_force")
    max_force = _normalize_force(cfg.get("max_force"), DEFAULT_MAX_FORCE, anchors.shape[0], "max_force")
    guy_wires = cfg.get("guy_wire_lengths")
    if guy_wires is not None:
        guy_wires = _expand(guy_wires, np.zeros(anchors.shape[0]), anchors.shape[0], "guy_wire_lengths")

    use_flex = bool(cfg.get("use_flex", False))
    ignore_gravity = bool(cfg.get("ignore_gravity", False))
    ignore_pretension = bool(cfg.get("ignore_pretension", False))

    motor_samples = pos_to_motor_pos_samples(
        anchors,
        poses,
        max_force,
        use_flex,
        spool_buildup_factor=spool_buildup,
        spool_r_in_origin=spool_r,
        spool_to_motor_gearing_factor=spool_to_motor,
        mech_adv_=mech_adv,
        lines_per_spool_=lines,
        min_force=min_force,
        spring_k_per_unit_length=cfg.get("spring_k_per_unit_length", springKPerUnitLength),
        mover_weight=cfg.get("mover_weight", mover_weight),
        ignore_gravity=ignore_gravity,
        ignore_pretension=ignore_pretension,
        guy_wire_lengths=guy_wires,
    )

    entry["motor_samples"] = motor_samples.tolist()
    return entry


def _write_jsonl(path: Path, entries: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate synthetic motor_samples for JSONL datasets.")
    parser.add_argument("input", type=Path, help="Path to intermediate JSONL file.")
    parser.add_argument("-o", "--output", type=Path, help="Output path (defaults to overwrite input).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep entries that already have motor_samples instead of recomputing.",
    )
    args = parser.parse_args()

    output_path = args.output or args.input
    entries: List[Dict[str, Any]] = []
    filled = 0
    skipped = 0

    for line in _load_lines(args.input):
        entry = json.loads(line)
        has_samples = bool(entry.get("motor_samples"))
        if has_samples and args.skip_existing:
            skipped += 1
        else:
            entry = _synthesize(entry)
            filled += 1
        entries.append(entry)

    _write_jsonl(output_path, entries)
    print(f"Wrote {len(entries)} entries to {output_path} ({filled} filled, {skipped} skipped).")


if __name__ == "__main__":
    main()
