#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild UAVDataset train/val/test info PKLs from a YAML frame-range plan.

This script is designed for the current CARLA-UAV -> BEVFusion dataset:
- scene structure stays unchanged;
- the base converter first produces whole-town train/val/test info PKLs;
- this script merges those base info files and repartitions keyframes;
- raw RGB/LiDAR files are never copied, moved, or deleted.

Important semantics
-------------------
* YAML range start/end are RAW frame_index values, inclusive.
* Explicit ranges override a scene's default_split.
* Historical sweeps are assigned by their RAW frame_index with exactly the
  same policy, so ranges must include any intended pre-roll raw frames.
* A symmetric guard removes keyframes around every split transition.
* Cross-split historical sweeps are rejected.
* Unless --allow-short-sweeps is set, samples that no longer have enough
  legal historical sweeps for the largest requested sweep variant are dropped
  from ALL variants. This keeps S0/S3/S6/S9 on identical keyframes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/uavdataset")
    p.add_argument(
        "--source-pattern",
        default="uavdataset_infos_{split}.pkl",
        help="Base converter output pattern; must contain {split}.",
    )
    p.add_argument("--plan", required=True)
    p.add_argument("--output-dir", default="data/uavdataset/balanced")
    p.add_argument(
        "--output-pattern",
        default="uavdataset_infos_{split}_s{sweeps}.pkl",
    )
    p.add_argument("--sweeps", nargs="+", type=int, default=[0, 3, 6, 9])
    p.add_argument(
        "--guard-keyframes",
        type=int,
        default=None,
        help="Override plan.settings.guard_keyframes.",
    )
    p.add_argument("--low-z-threshold", type=float, default=-5.0)
    p.add_argument("--upper-z-threshold", type=float, default=5.0)
    p.add_argument(
        "--allow-short-sweeps",
        action="store_true",
        help="Keep samples with fewer than max requested historical sweeps.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/repartition and write only split_audit_preview.json.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("infos"), list):
        raise RuntimeError(f"{path}: expected dict with list key 'infos'.")
    return data


def dump_pickle(data: dict, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def normalize_split(value: Any) -> str:
    value = str(value).strip().lower()
    if value not in SPLITS:
        raise ValueError(f"Invalid split {value!r}; expected {SPLITS}.")
    return value


class SplitPlan:
    def __init__(self, raw: Mapping[str, Any]):
        if int(raw.get("version", 1)) != 1:
            raise ValueError("Only plan version=1 is supported.")

        self.settings = dict(raw.get("settings", {}))
        scenes = raw.get("scenes")
        if not isinstance(scenes, Mapping) or not scenes:
            raise ValueError("Plan must contain a non-empty 'scenes' mapping.")

        self.scenes: Dict[str, dict] = {}
        for scene_name, spec in scenes.items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"{scene_name}: scene spec must be a mapping.")

            default_split = normalize_split(spec["default_split"])
            ranges = []
            for idx, item in enumerate(spec.get("ranges", []) or []):
                start = int(item["start"])
                end = int(item["end"])
                if end < start:
                    raise ValueError(
                        f"{scene_name}.ranges[{idx}]: end < start."
                    )
                ranges.append(
                    {
                        "start": start,
                        "end": end,
                        "split": normalize_split(item["split"]),
                        "name": str(item.get("name", f"range_{idx}")),
                    }
                )

            ranges.sort(key=lambda x: (x["start"], x["end"]))
            for a, b in zip(ranges, ranges[1:]):
                if b["start"] <= a["end"]:
                    raise ValueError(
                        f"{scene_name}: overlapping ranges "
                        f"{a['start']}-{a['end']} and {b['start']}-{b['end']}."
                    )

            self.scenes[str(scene_name)] = {
                "default_split": default_split,
                "ranges": ranges,
            }

    @classmethod
    def from_yaml(cls, path: Path) -> "SplitPlan":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("YAML root must be a mapping.")
        return cls(raw)

    def split_for(self, scene_name: str, frame_index: int) -> str:
        if scene_name not in self.scenes:
            raise KeyError(f"Scene {scene_name!r} missing from plan.")
        spec = self.scenes[scene_name]
        frame_index = int(frame_index)
        for item in spec["ranges"]:
            if item["start"] <= frame_index <= item["end"]:
                return item["split"]
        return spec["default_split"]

    def plain(self) -> dict:
        return {
            "version": 1,
            "settings": copy.deepcopy(self.settings),
            "scenes": copy.deepcopy(self.scenes),
        }


def combine_sources(
    dataset_root: Path, source_pattern: str
) -> Tuple[List[dict], dict, Dict[str, int]]:
    if "{split}" not in source_pattern:
        raise ValueError("--source-pattern must contain {split}.")

    metadata = None
    by_token: Dict[str, dict] = {}
    source_counts: Dict[str, int] = {}

    for split_name in SPLITS:
        path = dataset_root / source_pattern.format(split=split_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = load_pickle(path)
        source_counts[split_name] = len(data["infos"])

        if metadata is None:
            metadata = copy.deepcopy(data.get("metadata", {}))

        for info in data["infos"]:
            token = str(info.get("token", ""))
            if not token:
                raise RuntimeError(f"{path}: sample without token.")
            if token in by_token:
                raise RuntimeError(f"Duplicate token: {token}")
            if "scene_name" not in info or "frame_index" not in info:
                raise RuntimeError(
                    f"{token}: missing scene_name/frame_index."
                )
            by_token[token] = info

    infos = list(by_token.values())
    infos.sort(
        key=lambda x: (
            str(x["scene_name"]),
            int(x["frame_index"]),
        )
    )
    return infos, metadata or {}, source_counts


def validate_scene_coverage(plan: SplitPlan, infos: Sequence[dict]) -> None:
    dataset_scenes = sorted({str(x["scene_name"]) for x in infos})
    plan_scenes = sorted(plan.scenes)
    if dataset_scenes != plan_scenes:
        raise RuntimeError(
            "Plan/dataset scene mismatch:\n"
            f"dataset={dataset_scenes}\nplan={plan_scenes}"
        )


def parse_scene_frame(token: Any) -> Tuple[Optional[str], Optional[int]]:
    if token is None:
        return None, None
    text = str(token).replace("\\", "/")
    m = re.search(r"(Town[^/]+)/(\d+)(?:/[^/]+)?$", text)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def sweep_scene_frame(sweep: Mapping[str, Any]) -> Tuple[str, int]:
    scene, frame = parse_scene_frame(sweep.get("sample_data_token"))
    if scene is not None and frame is not None:
        return scene, frame

    path = str(sweep.get("data_path", "")).replace("\\", "/")
    m = re.search(r"(Town[^/]+)/(\d+)\.bin$", path)
    if m:
        return m.group(1), int(m.group(2))

    raise RuntimeError(f"Cannot resolve sweep scene/frame: {sweep}")


def find_transitions(
    plan: SplitPlan, infos: Sequence[dict]
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    by_scene: Dict[str, List[dict]] = defaultdict(list)
    for info in infos:
        by_scene[str(info["scene_name"])].append(info)

    transitions: List[dict] = []
    for scene_name, scene_infos in by_scene.items():
        scene_infos.sort(key=lambda x: int(x["frame_index"]))
        prev_split = None

        for idx, info in enumerate(scene_infos):
            cur_split = plan.split_for(scene_name, int(info["frame_index"]))
            if prev_split is not None and cur_split != prev_split:
                transitions.append(
                    {
                        "scene": scene_name,
                        "left_index": idx - 1,
                        "right_index": idx,
                        "left_frame": int(
                            scene_infos[idx - 1]["frame_index"]
                        ),
                        "right_frame": int(info["frame_index"]),
                        "left_split": prev_split,
                        "right_split": cur_split,
                    }
                )
            prev_split = cur_split

    return transitions, by_scene


def make_guard_drop_set(
    plan: SplitPlan,
    infos: Sequence[dict],
    guard_keyframes: int,
) -> Tuple[set, List[dict]]:
    transitions, by_scene = find_transitions(plan, infos)
    dropped = set()

    if guard_keyframes <= 0:
        return dropped, transitions

    for t in transitions:
        scene_infos = by_scene[t["scene"]]
        li = int(t["left_index"])
        ri = int(t["right_index"])

        # N keyframes on the left side.
        for idx in range(max(0, li - guard_keyframes + 1), li + 1):
            dropped.add(str(scene_infos[idx]["token"]))

        # N keyframes on the right side.
        for idx in range(ri, min(len(scene_infos), ri + guard_keyframes)):
            dropped.add(str(scene_infos[idx]["token"]))

    return dropped, transitions


def filter_sweeps(
    plan: SplitPlan,
    info: Mapping[str, Any],
    sample_split: str,
) -> Tuple[List[dict], List[dict]]:
    kept = []
    removed = []

    for sweep in info.get("sweeps", []):
        scene, frame = sweep_scene_frame(sweep)
        sweep_split = plan.split_for(scene, frame)

        if sweep_split == sample_split:
            kept.append(sweep)
        else:
            removed.append(
                {
                    "sample_data_token": sweep.get("sample_data_token"),
                    "data_path": sweep.get("data_path"),
                    "scene": scene,
                    "frame_index": frame,
                    "assigned_split": sweep_split,
                }
            )
    return kept, removed


def rebuild(
    plan: SplitPlan,
    infos: Sequence[dict],
    guard_keyframes: int,
    required_sweeps: int,
    allow_short_sweeps: bool,
) -> Tuple[Dict[str, List[dict]], dict]:
    guard_drop, transitions = make_guard_drop_set(
        plan, infos, guard_keyframes
    )

    out: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    guard_drop_split = Counter()
    guard_drop_scene = Counter()
    short_drop_split = Counter()
    short_drop_scene = Counter()
    removed_sweeps_total = 0
    removed_examples = []

    for original in infos:
        token = str(original["token"])
        scene = str(original["scene_name"])
        frame = int(original["frame_index"])
        split_name = plan.split_for(scene, frame)

        if token in guard_drop:
            guard_drop_split[split_name] += 1
            guard_drop_scene[scene] += 1
            continue

        kept_sweeps, removed = filter_sweeps(
            plan, original, split_name
        )
        removed_sweeps_total += len(removed)
        if removed and len(removed_examples) < 50:
            removed_examples.append(
                {
                    "token": token,
                    "scene": scene,
                    "frame_index": frame,
                    "split": split_name,
                    "removed": removed[:9],
                }
            )

        if (
            not allow_short_sweeps
            and required_sweeps > 0
            and len(kept_sweeps) < required_sweeps
        ):
            short_drop_split[split_name] += 1
            short_drop_scene[scene] += 1
            continue

        info = dict(original)
        info["sweeps"] = list(kept_sweeps)
        info["split"] = split_name
        out[split_name].append(info)

    for split_name in SPLITS:
        out[split_name].sort(
            key=lambda x: (
                str(x["scene_name"]),
                int(x["frame_index"]),
            )
        )

    safety = {
        "guard_keyframes": guard_keyframes,
        "guard_dropped_total": int(sum(guard_drop_split.values())),
        "guard_dropped_by_split": dict(guard_drop_split),
        "guard_dropped_by_scene": dict(guard_drop_scene),
        "short_sweep_dropped_total": int(
            sum(short_drop_split.values())
        ),
        "short_sweep_dropped_by_split": dict(short_drop_split),
        "short_sweep_dropped_by_scene": dict(short_drop_scene),
        "cross_split_sweeps_removed": int(removed_sweeps_total),
        "removed_sweep_examples": removed_examples,
        "transitions": [
            {
                k: v
                for k, v in t.items()
                if k not in ("left_index", "right_index")
            }
            for t in transitions
        ],
    }
    return out, safety


def valid_gt(
    info: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    if raw_boxes.size == 0:
        boxes = np.zeros((0, 7), dtype=np.float64)
    else:
        boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])[:, :7]

    names = np.asarray(info.get("gt_names", []), dtype=object)
    valid = np.asarray(info.get("valid_flag", []), dtype=bool)

    if len(valid) == 0:
        valid = np.ones(len(boxes), dtype=bool)

    if len(names) != len(boxes) or len(valid) != len(boxes):
        raise RuntimeError(
            f"{info.get('token')}: annotation array length mismatch."
        )

    num_pts = np.asarray(
        info.get("num_lidar_pts", np.full(len(boxes), np.nan)),
        dtype=np.float64,
    )
    if len(num_pts) != len(boxes):
        num_pts = np.full(len(boxes), np.nan)

    return boxes[valid], names[valid], num_pts[valid]


def summarize(
    infos: Sequence[dict],
    low_z_threshold: float,
    upper_z_threshold: float,
) -> dict:
    frames = 0
    empty_frames = 0
    gt_total = 0
    class_counts = Counter()
    low_z = 0
    upper_z = 0
    distance_bins = Counter()
    point_bins = Counter()
    per_scene = defaultdict(
        lambda: {"frames": 0, "empty_frames": 0, "gt": 0, "low_z": 0}
    )

    for info in infos:
        frames += 1
        scene = str(info["scene_name"])
        per_scene[scene]["frames"] += 1

        boxes, names, npts = valid_gt(info)
        if len(boxes) == 0:
            empty_frames += 1
            per_scene[scene]["empty_frames"] += 1
            continue

        gt_total += len(boxes)
        per_scene[scene]["gt"] += len(boxes)

        for box, name, pts in zip(boxes, names, npts):
            class_counts[str(name)] += 1

            z = float(box[2])
            if z < low_z_threshold:
                low_z += 1
                per_scene[scene]["low_z"] += 1
            if z > upper_z_threshold:
                upper_z += 1

            r = math.hypot(float(box[0]), float(box[1]))
            if r < 20.0:
                distance_bins["0-20"] += 1
            elif r < 35.0:
                distance_bins["20-35"] += 1
            elif r < 51.2:
                distance_bins["35-51.2"] += 1
            else:
                distance_bins[">=51.2"] += 1

            if not np.isfinite(pts):
                point_bins["unknown"] += 1
            elif pts < 30:
                point_bins["15-29"] += 1
            elif pts < 60:
                point_bins["30-59"] += 1
            elif pts < 120:
                point_bins["60-119"] += 1
            else:
                point_bins[">=120"] += 1

    dg = max(gt_total, 1)
    df = max(frames, 1)

    scene_stats = {}
    for scene, d in sorted(per_scene.items()):
        d = dict(d)
        d["low_z_fraction"] = (
            d["low_z"] / d["gt"] if d["gt"] else 0.0
        )
        d["empty_frame_fraction"] = (
            d["empty_frames"] / d["frames"] if d["frames"] else 0.0
        )
        scene_stats[scene] = d

    return {
        "frames": frames,
        "empty_frames": empty_frames,
        "empty_frame_fraction": empty_frames / df,
        "gt": int(gt_total),
        "class_counts": dict(class_counts),
        "class_fractions": {
            k: v / dg for k, v in sorted(class_counts.items())
        },
        "low_z_count": int(low_z),
        "low_z_fraction": low_z / dg,
        "upper_z_count": int(upper_z),
        "upper_z_fraction": upper_z / dg,
        "distance_bins": dict(distance_bins),
        "distance_fractions": {
            k: distance_bins[k] / dg
            for k in ("0-20", "20-35", "35-51.2", ">=51.2")
        },
        "lidar_point_bins": dict(point_bins),
        "lidar_point_fractions": {
            k: point_bins[k] / dg
            for k in ("15-29", "30-59", "60-119", ">=120", "unknown")
        },
        "by_scene": scene_stats,
    }


def retained_ranges(infos: Sequence[dict]) -> List[dict]:
    by_scene = defaultdict(list)
    for info in infos:
        by_scene[str(info["scene_name"])].append(
            int(info["frame_index"])
        )

    ranges = []
    for scene, frames in sorted(by_scene.items()):
        frames = sorted(frames)
        if not frames:
            continue
        start = prev = frames[0]
        for frame in frames[1:]:
            if frame - prev != 5:
                ranges.append(
                    {
                        "scene": scene,
                        "start_frame": start,
                        "end_frame": prev,
                        "keyframes": (prev - start) // 5 + 1,
                    }
                )
                start = frame
            prev = frame

        ranges.append(
            {
                "scene": scene,
                "start_frame": start,
                "end_frame": prev,
                "keyframes": (prev - start) // 5 + 1,
            }
        )
    return ranges


def write_variants(
    split_infos: Mapping[str, Sequence[dict]],
    metadata: Mapping[str, Any],
    output_dir: Path,
    output_pattern: str,
    sweeps_values: Sequence[int],
    plan_path: Path,
    overwrite: bool,
) -> Dict[str, Dict[str, str]]:
    if "{split}" not in output_pattern or "{sweeps}" not in output_pattern:
        raise ValueError(
            "--output-pattern must contain {split} and {sweeps}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Dict[str, str]] = {s: {} for s in SPLITS}

    for split_name in SPLITS:
        for n in sweeps_values:
            variant = []
            for original in split_infos[split_name]:
                info = dict(original)
                info["sweeps"] = list(
                    original.get("sweeps", [])
                )[:n]
                variant.append(info)

            meta = copy.deepcopy(dict(metadata))
            meta["max_sweeps"] = int(n)
            meta["split_rebuilt"] = True
            meta["split_plan"] = str(plan_path)
            meta["sweep_selection"] = "nearest_N_after_split_safety"

            path = output_dir / output_pattern.format(
                split=split_name, sweeps=n
            )
            dump_pickle(
                {"infos": variant, "metadata": meta},
                path,
                overwrite,
            )
            paths[split_name][str(n)] = str(path)

    return paths


def print_summary(audit: Mapping[str, Any]) -> None:
    print("\n=== UAV balanced split ===")
    print(
        "{:<7} {:>7} {:>8} {:>9} {:>9} {:>10}".format(
            "split", "frames", "GT", "empty%", "low-Z%", ">=35m%"
        )
    )

    total_frames = sum(
        audit["splits"][s]["frames"] for s in SPLITS
    )

    for split_name in SPLITS:
        s = audit["splits"][split_name]
        far = (
            s["distance_fractions"].get("35-51.2", 0.0)
            + s["distance_fractions"].get(">=51.2", 0.0)
        )
        print(
            "{:<7} {:>7d} {:>8d} {:>8.2f}% {:>8.2f}% {:>9.2f}%".format(
                split_name,
                s["frames"],
                s["gt"],
                100.0 * s["empty_frame_fraction"],
                100.0 * s["low_z_fraction"],
                100.0 * far,
            )
        )

    print("\nFrame fractions:")
    for split_name in SPLITS:
        n = audit["splits"][split_name]["frames"]
        print(
            f"  {split_name}: {n}/{total_frames} = "
            f"{n / max(total_frames, 1):.2%}"
        )

    print("\nSafety:")
    safety = audit["safety"]
    for key in (
        "guard_keyframes",
        "guard_dropped_total",
        "short_sweep_dropped_total",
        "cross_split_sweeps_removed",
    ):
        print(f"  {key}: {safety[key]}")


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    sweeps_values = sorted(set(int(x) for x in args.sweeps))
    if not sweeps_values or min(sweeps_values) < 0:
        raise SystemExit("--sweeps must be non-negative.")

    plan = SplitPlan.from_yaml(plan_path)
    guard = (
        int(args.guard_keyframes)
        if args.guard_keyframes is not None
        else int(plan.settings.get("guard_keyframes", 2))
    )
    if guard < 0:
        raise SystemExit("--guard-keyframes must be >= 0.")

    infos, metadata, source_counts = combine_sources(
        dataset_root, args.source_pattern
    )
    validate_scene_coverage(plan, infos)

    split_infos, safety = rebuild(
        plan=plan,
        infos=infos,
        guard_keyframes=guard,
        required_sweeps=max(sweeps_values),
        allow_short_sweeps=args.allow_short_sweeps,
    )

    audit = {
        "plan": plan.plain(),
        "plan_path": str(plan_path),
        "dataset_root": str(dataset_root),
        "source_pattern": args.source_pattern,
        "source_counts": source_counts,
        "input_total_frames": len(infos),
        "requested_sweeps": sweeps_values,
        "low_z_threshold": args.low_z_threshold,
        "upper_z_threshold": args.upper_z_threshold,
        "safety": safety,
        "splits": {
            split_name: summarize(
                split_infos[split_name],
                args.low_z_threshold,
                args.upper_z_threshold,
            )
            for split_name in SPLITS
        },
        "retained_ranges": {
            split_name: retained_ranges(split_infos[split_name])
            for split_name in SPLITS
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        audit_path = output_dir / "split_audit_preview.json"
        if audit_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{audit_path} exists; use --overwrite."
            )
        audit["output_paths"] = {}
        audit_path.write_text(
            json.dumps(
                audit,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            ),
            encoding="utf-8",
        )
        print_summary(audit)
        print(f"\nDRY RUN: no PKLs written.")
        print(f"Preview audit: {audit_path}")
        return

    output_paths = write_variants(
        split_infos=split_infos,
        metadata=metadata,
        output_dir=output_dir,
        output_pattern=args.output_pattern,
        sweeps_values=sweeps_values,
        plan_path=plan_path,
        overwrite=args.overwrite,
    )
    audit["output_paths"] = output_paths

    audit_path = output_dir / "split_audit.json"
    if audit_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{audit_path} exists; use --overwrite."
        )
    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print_summary(audit)
    print(f"\nAudit: {audit_path}")
    for split_name in SPLITS:
        for n in sweeps_values:
            print(
                f"{split_name} S{n}: "
                f"{output_paths[split_name][str(n)]}"
            )


if __name__ == "__main__":
    main()
