#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full UAVDataset + BEVFusion pre-training audit.

This script is intended to run AFTER UAVDataset conversion and BEFORE training.
It combines the useful parts of:
  * tools/analyze_uav_gt_z.py
  * tools/audit_uav_pipeline.py
  * tools/profile_uav_sweeps_voxels.py

Design goals
------------
1. No frame sampling: train/val/test are scanned in full.
2. Read the maximum requested sweep history once per keyframe, then reuse it
   for S0/S3/S6/S9-style statistics.
3. Keep recorder-side filtering separate. Raw labels are audited, but this
   script never changes recorder filtering policy.
4. Treat point_cloud_range as XYZ physical model ROI:
      [xmin, ymin, zmin, xmax, ymax, zmax]
   GT range membership is based on GT CENTER XYZ. Box-partly-outside is an
   audit signal only; it is not used to decide whether a GT belongs to ROI.
5. Write only two files by default:
      runs/uav_dataset_analysis_<timestamp>/analysis.json
      runs/uav_dataset_analysis_<timestamp>/recommended_changes.yaml
   The YAML contains only parameters that are recommended to change.

Recommended invocation (from BEVFusion project root)
------------------------------------------------------
python tools/analyze_uav_dataset.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --raw-root /path/to/raw/dataset \
  --converted-root data/uavdataset \
  --sweeps 0 3 6 9

python tools/analyze_uav_dataset.py \
    configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
    --raw-root data/uavdataset/raw \
    --converted-root data/uavdataset

默认会全量分析：

train + val + test
所有 frame
所有 valid/invalid GT metadata
全部点云 frame
0 / 3 / 6 / 9 sweeps
train identity
train deterministic augmentation × 2

Notes

-----
* The base uavdataset_infos_{split}.pkl produced with --max-sweeps 9 is the
  preferred source. Derived _s3/_s6/_s9 PKLs contain the same GT, but a _s6
  source cannot reconstruct S9 history.
* Frustum filtering is simulated from info['cams'] calibration and image_size,
  matching LiDARCameraFrustumFilter geometry. If on_threshold_failure='keep',
  this script also matches its fallback behavior. If it is 'raise', failures
  are reported while analysis continues with the geometrically visible points.
* Train augmentation is stochastic by nature. Every train frame is audited in
  identity form plus N deterministic augmentation repeats (default 2). This is
  still a full-frame audit; no frame is sampled away.
"""


from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm


SPLITS = ("train", "val", "test")
DEFAULT_CLASSES = ("car", "van", "truck", "bus")
EPS = 1e-9


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full UAVDataset/BEVFusion data, range, frustum and voxel audit."
    )
    p.add_argument("config", help="Recursive BEVFusion YAML config.")
    p.add_argument(
        "--raw-root",
        default=None,
        help=(
            "Optional raw recorder dataset root containing scene directories. "
            "Raw annotation_stats and saved-object distributions are scanned in full."
        ),
    )
    p.add_argument(
        "--converted-root",
        default=None,
        help="Converted dataset root. Default: dataset_root resolved from config.",
    )
    p.add_argument(
        "--source-pattern",
        default="uavdataset_infos_{split}.pkl",
        help=(
            "Converted info source pattern. Prefer the base S9-capable infos. "
            "Must contain {split}."
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Splits to scan. Default: train val test.",
    )
    p.add_argument(
        "--sweeps",
        nargs="+",
        type=int,
        default=[0, 3, 6, 9],
        help="Historical sweep counts. Current frame is extra.",
    )
    p.add_argument(
        "--train-augment-repeats",
        type=int,
        default=2,
        help=(
            "Deterministic random train augmentation repeats per train frame. "
            "Identity is always audited separately."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--z-bin-width",
        type=float,
        default=0.5,
        help="Histogram width in metres for GT/point Z distributions.",
    )
    p.add_argument(
        "--xy-bin-width",
        type=float,
        default=2.0,
        help="Histogram width in metres for GT center X/Y distributions.",
    )
    p.add_argument(
        "--radius-bin-width",
        type=float,
        default=2.0,
        help="Histogram width in metres for GT radial-distance distributions.",
    )
    p.add_argument(
        "--z-span-thresholds",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 8.0, 10.0],
        help="Frame GT center-Z span thresholds to count.",
    )
    p.add_argument(
        "--gt-point-thresholds",
        nargs="+",
        type=int,
        default=[1, 5, 10, 15],
        help="Per-GT current-frame retained point thresholds.",
    )
    p.add_argument(
        "--capacity-percentile",
        type=float,
        default=99.5,
        help="Percentile used to recommend max_voxels/max_num_points increases.",
    )
    p.add_argument(
        "--cap-step",
        type=int,
        default=5000,
        help="Round max_voxels recommendations upward to this step.",
    )
    p.add_argument(
        "--max-point-cap-drop-ratio",
        type=float,
        default=0.01,
        help=(
            "Only recommend increasing max_num_points when the aggregate point "
            "drop ratio caused by the per-voxel point cap exceeds this value."
        ),
    )
    p.add_argument(
        "--max-recommended-points-per-voxel",
        type=int,
        default=32,
        help="Safety ceiling for automatic max_num_points recommendation.",
    )
    p.add_argument(
        "--range-margin",
        type=float,
        default=0.5,
        help="Extra margin for a data-driven ROI expansion suggestion.",
    )
    p.add_argument(
        "--range-robust-percentile",
        type=float,
        default=99.9,
        help="Central train+val GT-center coverage used for ROI expansion suggestions.",
    )
    p.add_argument(
        "--range-expand-min-outside-fraction",
        type=float,
        default=0.001,
        help=(
            "Minimum train+val valid-GT center outside fraction before an automatic "
            "point_cloud_range expansion is recommended. 0.001 = 0.1%%."
        ),
    )
    p.add_argument(
        "--output-root",
        default="runs",
        help="Parent output directory. Default: runs/",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Default: uav_dataset_analysis_<timestamp>.",
    )
    p.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw recorder label audit even if --raw-root is provided.",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run lightweight helper checks and exit.",
    )
    return p.parse_args()


def load_cfg(path: str):
    """Load the project's recursive Torchpack YAML exactly like existing tools."""
    try:
        from mmcv import Config
        from torchpack.utils.config import configs
        from mmdet3d.utils import recursive_eval
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Failed to import BEVFusion config dependencies. Run this script from "
            "the BEVFusion project environment/root. Original error: {}".format(exc)
        ) from exc

    configs.load(path, recursive=True)
    return Config(recursive_eval(configs), filename=path)


def as_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain(v) for v in value]
    return value


def nested_get(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        try:
            if isinstance(cur, Mapping):
                cur = cur[key]
            else:
                cur = getattr(cur, key)
        except (KeyError, AttributeError, TypeError):
            return default
    return cur


def set_nested(root: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    cur = root
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = as_plain(value)


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_infos(path: Path) -> Tuple[List[dict], dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "infos" not in data:
        raise ValueError("Expected {'infos', 'metadata'} in {}".format(path))
    return list(data["infos"]), dict(data.get("metadata", {}))


def strict_center_mask(boxes: np.ndarray, pcr: Sequence[float]) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    low = np.asarray(pcr[:3], dtype=np.float64)
    high = np.asarray(pcr[3:], dtype=np.float64)
    return ((boxes[:, :3] > low) & (boxes[:, :3] < high)).all(axis=1)


def point_range_mask(points: np.ndarray, pcr: Sequence[float]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    low = np.asarray(pcr[:3], dtype=np.float64)
    high = np.asarray(pcr[3:], dtype=np.float64)
    return ((points[:, :3] > low) & (points[:, :3] < high)).all(axis=1)


def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def percentile_summary(values: Iterable[float]) -> Optional[dict]:
    values = list(values)
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "min": float(np.min(a)),
        "p01": float(np.percentile(a, 1)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "p99_5": float(np.percentile(a, 99.5)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def ceil_step(value: float, step: int) -> int:
    return int(math.ceil(float(value) / float(step)) * step)


def snap_down(value: float, step: float) -> float:
    return float(math.floor((value + EPS) / step) * step)


def snap_up(value: float, step: float) -> float:
    return float(math.ceil((value - EPS) / step) * step)


def close_list(a: Any, b: Any, atol: float = 1e-6) -> bool:
    try:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=atol, rtol=0.0))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Streaming histograms / weighted occupancy summaries
# -----------------------------------------------------------------------------


@dataclass
class Histogram1D:
    width: float
    counts: Counter = field(default_factory=Counter)
    count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def update(self, values: np.ndarray) -> None:
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        if not len(a):
            return
        self.count += int(len(a))
        current_min = float(np.min(a))
        current_max = float(np.max(a))
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)
        idx = np.floor(a / self.width).astype(np.int64)
        unique, c = np.unique(idx, return_counts=True)
        for i, n in zip(unique.tolist(), c.tolist()):
            self.counts[int(i)] += int(n)

    def to_dict(self) -> dict:
        if not self.counts:
            return {
                "bin_width": float(self.width),
                "count": 0,
                "min": None,
                "max": None,
                "bins": [],
            }
        bins = []
        for idx in sorted(self.counts):
            low = idx * self.width
            high = (idx + 1) * self.width
            c = int(self.counts[idx])
            bins.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "count": c,
                    "fraction": safe_rate(c, self.count),
                }
            )
        return {
            "bin_width": float(self.width),
            "count": int(self.count),
            "min": self.min_value,
            "max": self.max_value,
            "bins": bins,
        }


@dataclass
class OccupancyHistogram:
    counts: Counter = field(default_factory=Counter)
    num_voxels: int = 0

    def update_counts(self, occupancies: np.ndarray) -> None:
        occ = np.asarray(occupancies, dtype=np.int64).reshape(-1)
        if not len(occ):
            return
        freq = np.bincount(occ)
        for occupancy, n in enumerate(freq):
            if occupancy > 0 and n:
                self.counts[int(occupancy)] += int(n)
                self.num_voxels += int(n)

    def percentile(self, q: float) -> Optional[float]:
        if self.num_voxels == 0:
            return None
        target = (q / 100.0) * self.num_voxels
        running = 0
        for occupancy in sorted(self.counts):
            running += self.counts[occupancy]
            if running >= target:
                return float(occupancy)
        return float(max(self.counts))

    def to_dict(self) -> dict:
        if not self.counts:
            return {"num_voxels": 0, "p50": None, "p95": None, "p99": None, "p99_5": None, "max": None, "counts": {}}
        return {
            "num_voxels": int(self.num_voxels),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "p99_5": self.percentile(99.5),
            "max": int(max(self.counts)),
            "counts": {str(k): int(v) for k, v in sorted(self.counts.items())},
        }


# -----------------------------------------------------------------------------
# Config extraction / consistency
# -----------------------------------------------------------------------------


def extract_config(cfg: Any, config_path: Path) -> dict:
    pcr = list(map(float, cfg.point_cloud_range))
    voxel_size = list(map(float, cfg.voxel_size))
    load_dim = int(cfg.load_dim)
    max_sweeps = int(getattr(cfg, "max_sweeps", 0))
    classes = list(getattr(cfg, "object_classes", DEFAULT_CLASSES))

    frustum = dict(getattr(cfg, "frustum_filter", {}))
    augment3d = dict(getattr(cfg, "augment3d", {}))

    max_num_points = nested_get(cfg, ["model", "encoders", "lidar", "voxelize", "max_num_points"], None)
    max_voxels = nested_get(cfg, ["model", "encoders", "lidar", "voxelize", "max_voxels"], None)
    sparse_shape = nested_get(cfg, ["model", "encoders", "lidar", "backbone", "sparse_shape"], None)

    camera_xbound = nested_get(cfg, ["model", "encoders", "camera", "vtransform", "xbound"], None)
    camera_ybound = nested_get(cfg, ["model", "encoders", "camera", "vtransform", "ybound"], None)
    camera_zbound = nested_get(cfg, ["model", "encoders", "camera", "vtransform", "zbound"], None)
    camera_dbound = nested_get(cfg, ["model", "encoders", "camera", "vtransform", "dbound"], None)

    train_grid = nested_get(cfg, ["model", "heads", "object", "train_cfg", "grid_size"], None)
    train_pcr = nested_get(cfg, ["model", "heads", "object", "train_cfg", "point_cloud_range"], None)
    test_grid = nested_get(cfg, ["model", "heads", "object", "test_cfg", "grid_size"], None)
    test_pc_range = nested_get(cfg, ["model", "heads", "object", "test_cfg", "pc_range"], None)
    post_center_range = nested_get(cfg, ["model", "heads", "object", "bbox_coder", "post_center_range"], None)

    return {
        "config_file": str(config_path.resolve()),
        "dataset_root": str(Path(str(cfg.dataset_root)).expanduser()),
        "point_cloud_range": pcr,
        "voxel_size": voxel_size,
        "load_dim": load_dim,
        "max_sweeps": max_sweeps,
        "object_classes": classes,
        "frustum_filter": as_plain(frustum),
        "augment3d": as_plain(augment3d),
        "model": {
            "lidar_voxelize": {
                "max_num_points": None if max_num_points is None else int(max_num_points),
                "max_voxels": None if max_voxels is None else list(map(int, max_voxels)),
            },
            "sparse_shape": as_plain(sparse_shape),
            "camera_vtransform": {
                "xbound": as_plain(camera_xbound),
                "ybound": as_plain(camera_ybound),
                "zbound": as_plain(camera_zbound),
                "dbound": as_plain(camera_dbound),
            },
            "head": {
                "train_grid_size": as_plain(train_grid),
                "train_point_cloud_range": as_plain(train_pcr),
                "test_grid_size": as_plain(test_grid),
                "test_pc_range": as_plain(test_pc_range),
                "post_center_range": as_plain(post_center_range),
            },
        },
    }


def project_expected_grid(pcr: Sequence[float], voxel_size: Sequence[float]) -> dict:
    low = np.asarray(pcr[:3], dtype=np.float64)
    high = np.asarray(pcr[3:], dtype=np.float64)
    voxel = np.asarray(voxel_size, dtype=np.float64)
    raw = (high - low) / voxel
    rounded = np.rint(raw).astype(np.int64)
    divisible = bool(np.allclose(raw, rounded, atol=1e-6, rtol=0.0))
    bins_xyz = rounded.tolist()
    # This fork currently uses [x_bins, y_bins, z_bins + 1] for both head grid
    # and SparseEncoder sparse_shape (e.g. 1024,1024,131 for 130 z bins).
    project_shape = [int(rounded[0]), int(rounded[1]), int(rounded[2] + 1)]
    return {
        "divisible_by_voxel_size": divisible,
        "continuous_bins_xyz": raw.tolist(),
        "voxel_bins_xyz": bins_xyz,
        "project_grid_sparse_convention": project_shape,
    }


def config_consistency_report(config: dict) -> dict:
    pcr = config["point_cloud_range"]
    voxel = config["voxel_size"]
    expected = project_expected_grid(pcr, voxel)
    expected_shape = expected["project_grid_sparse_convention"]
    model = config["model"]
    cam = model["camera_vtransform"]
    head = model["head"]

    checks = {}
    checks["voxel_range_divisible"] = expected["divisible_by_voxel_size"]
    checks["lidar_train_pcr_matches"] = close_list(head["train_point_cloud_range"], pcr) if head["train_point_cloud_range"] is not None else None
    checks["head_train_grid_matches_project_convention"] = close_list(head["train_grid_size"], expected_shape) if head["train_grid_size"] is not None else None
    checks["head_test_grid_matches_project_convention"] = close_list(head["test_grid_size"], expected_shape) if head["test_grid_size"] is not None else None
    checks["sparse_shape_matches_project_convention"] = close_list(model["sparse_shape"], expected_shape) if model["sparse_shape"] is not None else None
    checks["head_test_xy_origin_matches"] = close_list(head["test_pc_range"], pcr[:2]) if head["test_pc_range"] is not None else None
    checks["post_center_range_matches_pcr"] = close_list(head["post_center_range"], pcr) if head["post_center_range"] is not None else None

    if cam["xbound"] is not None:
        checks["camera_xbound_endpoints_match"] = close_list(cam["xbound"][:2], [pcr[0], pcr[3]])
    else:
        checks["camera_xbound_endpoints_match"] = None
    if cam["ybound"] is not None:
        checks["camera_ybound_endpoints_match"] = close_list(cam["ybound"][:2], [pcr[1], pcr[4]])
    else:
        checks["camera_ybound_endpoints_match"] = None
    if cam["zbound"] is not None:
        checks["camera_zbound_endpoints_match"] = close_list(cam["zbound"][:2], [pcr[2], pcr[5]])
    else:
        checks["camera_zbound_endpoints_match"] = None

    return {"expected": expected, "checks": checks}


# -----------------------------------------------------------------------------
# Raw recorder audit (no recorder policy changes)
# -----------------------------------------------------------------------------


def audit_raw_labels(raw_root: Path, classes: Sequence[str]) -> dict:
    root = raw_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Raw root does not exist: {}".format(root))

    scenes = [
        p
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / "labels").is_dir()
    ]
    report = {
        "root": str(root),
        "scene_count": len(scenes),
        "frames": 0,
        "annotation_stats_totals": {},
        "saved_class_counts": {},
        "saved_num_lidar_points": None,
        "saved_num_rgb_visible_pixels": None,
        "scenes": {},
    }
    all_stats = Counter()
    all_classes = Counter()
    all_lidar_points: List[float] = []
    all_rgb_pixels: List[float] = []

    for scene in scenes:
        label_files = sorted((scene / "labels").glob("*.json"))
        scene_stats = Counter()
        scene_classes = Counter()
        scene_lidar: List[float] = []
        scene_rgb: List[float] = []

        for path in tqdm(label_files, desc="raw {}".format(scene.name), unit="frame"):
            data = load_json(path)
            stats = data.get("annotation_stats", {})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    scene_stats[str(key)] += int(value)
                    all_stats[str(key)] += int(value)
            for obj in data.get("objects", []):
                name = str(obj.get("class", "unknown"))
                if classes and name not in classes:
                    continue
                scene_classes[name] += 1
                all_classes[name] += 1
                if obj.get("num_lidar_points") is not None:
                    v = float(obj["num_lidar_points"])
                    scene_lidar.append(v)
                    all_lidar_points.append(v)
                if obj.get("num_rgb_visible_pixels") is not None:
                    v = float(obj["num_rgb_visible_pixels"])
                    scene_rgb.append(v)
                    all_rgb_pixels.append(v)

        metadata = {}
        metadata_path = scene / "metadata.json"
        if metadata_path.is_file():
            raw_meta = load_json(metadata_path)
            metadata = {
                "route_map": raw_meta.get("route_map"),
                "route_name": raw_meta.get("route_name"),
                "actual_num_frames": raw_meta.get("actual_num_frames"),
                "annotations": raw_meta.get("annotations"),
            }

        report["scenes"][scene.name] = {
            "frames": len(label_files),
            "metadata": metadata,
            "annotation_stats_totals": dict(scene_stats),
            "saved_class_counts": dict(scene_classes),
            "saved_num_lidar_points": percentile_summary(scene_lidar),
            "saved_num_rgb_visible_pixels": percentile_summary(scene_rgb),
        }
        report["frames"] += len(label_files)

    report["annotation_stats_totals"] = dict(all_stats)
    report["saved_class_counts"] = dict(all_classes)
    report["saved_num_lidar_points"] = percentile_summary(all_lidar_points)
    report["saved_num_rgb_visible_pixels"] = percentile_summary(all_rgb_pixels)
    return report


# -----------------------------------------------------------------------------
# Converted GT audit
# -----------------------------------------------------------------------------


def rotated_xy_extents(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned XY bounds of upright rotated boxes.

    boxes are [cx, cy, cz, dim_x, dim_y, dim_z, yaw]. Naming of dim_x/dim_y
    does not matter for this bound calculation.
    """
    if len(boxes) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty, empty
    cx, cy = boxes[:, 0], boxes[:, 1]
    dx, dy = boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6]
    c = np.abs(np.cos(yaw))
    s = np.abs(np.sin(yaw))
    half_x = 0.5 * (c * dx + s * dy)
    half_y = 0.5 * (s * dx + c * dy)
    return cx - half_x, cy - half_y, cx + half_x, cy + half_y


def box_range_masks(boxes: np.ndarray, pcr: Sequence[float]) -> dict:
    n = len(boxes)
    if n == 0:
        z = np.zeros(0, dtype=bool)
        return {
            "center_inside": z,
            "outside_x": z,
            "outside_y": z,
            "outside_z": z,
            "partly_or_fully_outside": z,
            "partly_outside_not_fully": z,
            "fully_outside": z,
        }

    pcr = np.asarray(pcr, dtype=np.float64)
    cx, cy, cz = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    center_inside = strict_center_mask(boxes, pcr)
    outside_x = (cx <= pcr[0]) | (cx >= pcr[3])
    outside_y = (cy <= pcr[1]) | (cy >= pcr[4])
    outside_z = (cz <= pcr[2]) | (cz >= pcr[5])

    min_x, min_y, max_x, max_y = rotated_xy_extents(boxes)
    half_h = 0.5 * boxes[:, 5]
    min_z = cz - half_h
    max_z = cz + half_h

    partly_or_fully = (
        (min_x <= pcr[0])
        | (min_y <= pcr[1])
        | (min_z <= pcr[2])
        | (max_x >= pcr[3])
        | (max_y >= pcr[4])
        | (max_z >= pcr[5])
    )
    fully = (
        (max_x <= pcr[0])
        | (max_y <= pcr[1])
        | (max_z <= pcr[2])
        | (min_x >= pcr[3])
        | (min_y >= pcr[4])
        | (min_z >= pcr[5])
    )
    return {
        "center_inside": center_inside,
        "outside_x": outside_x,
        "outside_y": outside_y,
        "outside_z": outside_z,
        "partly_or_fully_outside": partly_or_fully,
        "partly_outside_not_fully": partly_or_fully & ~fully,
        "fully_outside": fully,
    }


def make_gt_row(split: str, info: dict, name: str, box: np.ndarray, lidar_points: float) -> dict:
    cx, cy, cz, dx, dy, dz, yaw = map(float, box[:7])
    return {
        "split": split,
        "scene": str(info.get("scene_name", "unknown")),
        "location": str(info.get("location", "unknown")),
        "class": name,
        "x": cx,
        "y": cy,
        "center_z": cz,
        "bottom_z": cz - 0.5 * dz,
        "top_z": cz + 0.5 * dz,
        "dim_x": dx,
        "dim_y": dy,
        "height": dz,
        "yaw": yaw,
        "radius": float(np.hypot(cx, cy)),
        "num_lidar_pts": float(lidar_points),
    }


def summarize_gt_rows(
    rows: Sequence[dict],
    z_bin_width: float,
    xy_bin_width: float,
    radius_bin_width: float,
) -> dict:
    if not rows:
        return {"count": 0}

    keys = (
        "x",
        "y",
        "center_z",
        "bottom_z",
        "top_z",
        "dim_x",
        "dim_y",
        "height",
        "radius",
        "num_lidar_pts",
    )
    out = {"count": len(rows), "summary": {}}
    for key in keys:
        out["summary"][key] = percentile_summary(row[key] for row in rows)

    hist_specs = {
        "x": xy_bin_width,
        "y": xy_bin_width,
        "center_z": z_bin_width,
        "bottom_z": z_bin_width,
        "top_z": z_bin_width,
        "radius": radius_bin_width,
    }
    out["histograms"] = {}
    for key, width in hist_specs.items():
        hist = Histogram1D(width)
        hist.update(np.asarray([row[key] for row in rows], dtype=np.float64))
        out["histograms"][key] = hist.to_dict()
    return out


def audit_gt(
    infos_by_split: Mapping[str, Sequence[dict]],
    pcr: Sequence[float],
    classes: Sequence[str],
    z_bin_width: float,
    xy_bin_width: float,
    radius_bin_width: float,
    z_span_thresholds: Sequence[float],
) -> Tuple[dict, Dict[str, List[dict]]]:
    report: Dict[str, Any] = {"splits": {}}
    rows_by_split: Dict[str, List[dict]] = {}

    for split, infos in infos_by_split.items():
        rows: List[dict] = []
        valid_class_counts = Counter()
        invalid_class_counts = Counter()
        class_rows: Dict[str, List[dict]] = defaultdict(list)
        scene_rows: Dict[str, List[dict]] = defaultdict(list)
        scene_class_counts: Dict[str, Counter] = defaultdict(Counter)
        frame_z_spans: List[float] = []
        empty_valid = 0
        range_counts = Counter()
        total_valid = 0

        for info in tqdm(infos, desc="GT {}".format(split), unit="frame"):
            boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)
            names = np.asarray(info.get("gt_names", []), dtype=object)
            valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
            lidar_counts = np.asarray(info.get("num_lidar_pts", np.zeros(len(boxes))), dtype=np.float64)
            if not (len(boxes) == len(names) == len(valid) == len(lidar_counts)):
                raise ValueError("GT array length mismatch in {}".format(info.get("token")))

            for name, is_valid in zip(names.tolist(), valid.tolist()):
                name = str(name)
                (valid_class_counts if is_valid else invalid_class_counts)[name] += 1

            valid_boxes = boxes[valid]
            valid_names = names[valid]
            valid_lidar = lidar_counts[valid]
            if len(valid_boxes) == 0:
                empty_valid += 1
            else:
                frame_z_spans.append(float(np.max(valid_boxes[:, 2]) - np.min(valid_boxes[:, 2])))

            masks = box_range_masks(valid_boxes, pcr)
            total_valid += len(valid_boxes)
            range_counts["center_outside_x"] += int(masks["outside_x"].sum())
            range_counts["center_outside_y"] += int(masks["outside_y"].sum())
            range_counts["center_outside_z"] += int(masks["outside_z"].sum())
            range_counts["center_outside_xy"] += int((masks["outside_x"] | masks["outside_y"]).sum())
            range_counts["center_outside_xyz"] += int((~masks["center_inside"]).sum())
            range_counts["box_partly_or_fully_outside"] += int(masks["partly_or_fully_outside"].sum())
            range_counts["box_partly_outside_not_fully"] += int(masks["partly_outside_not_fully"].sum())
            range_counts["box_fully_outside"] += int(masks["fully_outside"].sum())

            scene = str(info.get("scene_name", "unknown"))
            for box, raw_name, lidar_points in zip(valid_boxes, valid_names, valid_lidar):
                name = str(raw_name)
                if classes and name not in classes:
                    continue
                row = make_gt_row(split, info, name, box, lidar_points)
                rows.append(row)
                class_rows[name].append(row)
                scene_rows[scene].append(row)
                scene_class_counts[scene][name] += 1

        rows_by_split[split] = rows
        z_span_hist = Histogram1D(z_bin_width)
        z_span_hist.update(np.asarray(frame_z_spans, dtype=np.float64))
        z_span_counts = {
            ">={:g}m".format(threshold): int(np.sum(np.asarray(frame_z_spans) >= threshold))
            for threshold in z_span_thresholds
        }
        range_report = {
            key: {
                "count": int(value),
                "fraction_of_valid_gt": safe_rate(value, total_valid),
            }
            for key, value in range_counts.items()
        }

        report["splits"][split] = {
            "frames": len(infos),
            "empty_or_no_valid_gt_frames": int(empty_valid),
            "valid_gt": int(total_valid),
            "valid_class_counts": dict(valid_class_counts),
            "invalid_class_counts": dict(invalid_class_counts),
            "valid_class_counts_by_scene": {
                scene: dict(counter) for scene, counter in sorted(scene_class_counts.items())
            },
            "range_audit": range_report,
            "distribution": summarize_gt_rows(rows, z_bin_width, xy_bin_width, radius_bin_width),
            "by_class": {
                name: summarize_gt_rows(class_rows[name], z_bin_width, xy_bin_width, radius_bin_width)
                for name in sorted(class_rows)
            },
            "by_scene": {
                scene: summarize_gt_rows(scene_rows[scene], z_bin_width, xy_bin_width, radius_bin_width)
                for scene in sorted(scene_rows)
            },
            "frame_center_z_span": {
                "summary": percentile_summary(frame_z_spans),
                "histogram": z_span_hist.to_dict(),
                "threshold_counts": z_span_counts,
                "threshold_fractions": {
                    key: safe_rate(value, len(infos)) for key, value in z_span_counts.items()
                },
            },
        }

    report["split_comparison"] = compare_split_distributions(rows_by_split, z_bin_width, classes)
    return report, rows_by_split


def rows_to_hist_counter(rows: Sequence[dict], key: str, width: float) -> Counter:
    c = Counter()
    if not rows:
        return c
    values = np.asarray([r[key] for r in rows], dtype=np.float64)
    idx, counts = np.unique(np.floor(values / width).astype(np.int64), return_counts=True)
    for i, n in zip(idx.tolist(), counts.tolist()):
        c[int(i)] += int(n)
    return c


def js_divergence(a: Counter, b: Counter) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    pa = np.asarray([a.get(k, 0) for k in keys], dtype=np.float64)
    pb = np.asarray([b.get(k, 0) for k in keys], dtype=np.float64)
    if pa.sum() == 0 or pb.sum() == 0:
        return 0.0
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    mask_a = pa > 0
    mask_b = pb > 0
    kl_a = np.sum(pa[mask_a] * np.log(pa[mask_a] / m[mask_a]))
    kl_b = np.sum(pb[mask_b] * np.log(pb[mask_b] / m[mask_b]))
    return float(0.5 * (kl_a + kl_b))


def top_bin_fraction_differences(a: Counter, b: Counter, width: float, limit: int = 12) -> List[dict]:
    total_a = sum(a.values())
    total_b = sum(b.values())
    rows = []
    for idx in set(a) | set(b):
        fa = safe_rate(a.get(idx, 0), total_a)
        fb = safe_rate(b.get(idx, 0), total_b)
        rows.append(
            {
                "low": float(idx * width),
                "high": float((idx + 1) * width),
                "fraction_a": fa,
                "fraction_b": fb,
                "absolute_fraction_difference": abs(fa - fb),
            }
        )
    rows.sort(key=lambda r: r["absolute_fraction_difference"], reverse=True)
    return rows[:limit]


def compare_split_distributions(
    rows_by_split: Mapping[str, Sequence[dict]], z_bin_width: float, classes: Sequence[str]
) -> dict:
    result = {}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a_name, b_name in pairs:
        if a_name not in rows_by_split or b_name not in rows_by_split:
            continue
        a_rows = rows_by_split[a_name]
        b_rows = rows_by_split[b_name]
        a_hist = rows_to_hist_counter(a_rows, "center_z", z_bin_width)
        b_hist = rows_to_hist_counter(b_rows, "center_z", z_bin_width)
        a_classes = Counter(r["class"] for r in a_rows)
        b_classes = Counter(r["class"] for r in b_rows)
        class_delta = {}
        for name in sorted(set(classes) | set(a_classes) | set(b_classes)):
            fa = safe_rate(a_classes[name], len(a_rows))
            fb = safe_rate(b_classes[name], len(b_rows))
            class_delta[name] = {
                "fraction_a": fa,
                "fraction_b": fb,
                "absolute_fraction_difference": abs(fa - fb),
            }
        per_class_z = {}
        for name in sorted(set(classes) | set(a_classes) | set(b_classes)):
            aa = [r for r in a_rows if r["class"] == name]
            bb = [r for r in b_rows if r["class"] == name]
            ah = rows_to_hist_counter(aa, "center_z", z_bin_width)
            bh = rows_to_hist_counter(bb, "center_z", z_bin_width)
            per_class_z[name] = {
                "gt_a": len(aa),
                "gt_b": len(bb),
                "js_divergence_natural_log": js_divergence(ah, bh),
                "top_z_bins_by_fraction_difference": top_bin_fraction_differences(ah, bh, z_bin_width, 8),
            }
        result["{}_vs_{}".format(a_name, b_name)] = {
            "gt_a": len(a_rows),
            "gt_b": len(b_rows),
            "center_z_js_divergence_natural_log": js_divergence(a_hist, b_hist),
            "top_center_z_bins_by_fraction_difference": top_bin_fraction_differences(a_hist, b_hist, z_bin_width),
            "class_fraction_difference": class_delta,
            "per_class_center_z": per_class_z,
        }
    return result


# -----------------------------------------------------------------------------
# Point cloud/frustum/augmentation/voxel helpers
# -----------------------------------------------------------------------------


def load_cloud(path: Path, load_dim: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % load_dim:
        raise ValueError("{}: {} floats not divisible by load_dim={}".format(path, arr.size, load_dim))
    return arr.reshape(-1, load_dim).copy()


def transform_sweep_to_current(points: np.ndarray, sweep: dict, current_ts_us: int) -> np.ndarray:
    out = points.copy()
    r = np.asarray(sweep["sensor2lidar_rotation"], dtype=np.float32)
    t = np.asarray(sweep["sensor2lidar_translation"], dtype=np.float32)
    out[:, :3] = out[:, :3] @ r.T
    out[:, :3] += t
    if out.shape[1] >= 5:
        out[:, 4] = float(current_ts_us - sweep["timestamp"]) / 1e6
    return out


def camera_lidar2image(cam: dict) -> np.ndarray:
    r = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
    camera2lidar = np.eye(4, dtype=np.float64)
    camera2lidar[:3, :3] = r
    camera2lidar[:3, 3] = t
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(cam["camera_intrinsics"], dtype=np.float64)
    return intrinsic @ np.linalg.inv(camera2lidar)


def frustum_mask(points: np.ndarray, cams: Mapping[str, dict], frustum_cfg: Mapping[str, Any]) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not cams:
        raise RuntimeError("No camera calibration in info['cams']")

    xyz1 = np.ones((n, 4), dtype=np.float64)
    xyz1[:, :3] = points[:, :3]
    mode = str(frustum_cfg.get("mode", "union"))
    keep = np.zeros(n, dtype=bool) if mode == "union" else np.ones(n, dtype=bool)
    if mode not in ("union", "intersection"):
        raise ValueError("Unsupported frustum mode: {}".format(mode))

    min_depth = float(frustum_cfg.get("min_depth", 0.05))
    max_depth = frustum_cfg.get("max_depth", None)
    max_depth = None if max_depth is None else float(max_depth)
    margin = float(frustum_cfg.get("margin_px", 2.0))

    for cam in cams.values():
        mat = camera_lidar2image(cam)
        projected = xyz1 @ mat.T
        depth = projected[:, 2]
        valid = np.isfinite(depth) & (depth > min_depth)
        if max_depth is not None:
            valid &= depth <= max_depth
        u = np.full(n, np.nan, dtype=np.float64)
        v = np.full(n, np.nan, dtype=np.float64)
        u[valid] = projected[valid, 0] / depth[valid]
        v[valid] = projected[valid, 1] / depth[valid]
        width, height = map(int, cam["image_size"])
        visible = (
            valid
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= margin)
            & (u < width - margin)
            & (v >= margin)
            & (v < height - margin)
        )
        if mode == "union":
            keep |= visible
        else:
            keep &= visible
    return keep


def effective_frustum_selection(
    before_count: int,
    visible_count: int,
    frustum_cfg: Mapping[str, Any],
) -> Tuple[bool, bool]:
    """Return (threshold_failed, use_visible_points).

    The transform behavior is:
      * no failure -> visible points
      * failure + keep -> original points
      * failure + raise -> training would raise; analyzer continues with visible
        points so downstream capacity can still be inspected.
    """
    min_points = int(frustum_cfg.get("min_points", 0))
    min_keep_ratio = float(frustum_cfg.get("min_keep_ratio", 0.0))
    ratio = safe_rate(visible_count, before_count)
    failed = visible_count < min_points or ratio < min_keep_ratio
    if failed and str(frustum_cfg.get("on_threshold_failure", "raise")) == "keep":
        return True, False
    return failed, True


def sample_train_augmentation(rng: np.random.Generator, aug_cfg: Mapping[str, Any]) -> dict:
    scale_lim = list(aug_cfg.get("scale", [1.0, 1.0]))
    rot_lim = list(aug_cfg.get("rotate", [0.0, 0.0]))
    trans_lim = float(aug_cfg.get("translate", 0.0))
    return {
        "scale": float(rng.uniform(float(scale_lim[0]), float(scale_lim[1]))),
        "theta": float(rng.uniform(float(rot_lim[0]), float(rot_lim[1]))),
        "translation": rng.normal(0.0, trans_lim, size=3).astype(np.float64),
        "flip_horizontal": bool(rng.integers(0, 2)),
        "flip_vertical": bool(rng.integers(0, 2)),
    }


def apply_train_augmentation_xyz(xyz: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    if len(xyz) == 0:
        return np.asarray(xyz, dtype=np.float64).copy()
    out = np.asarray(xyz, dtype=np.float64).copy()
    theta = float(params["theta"])
    c, s = math.cos(-theta), math.sin(-theta)
    r = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    out[:, :2] = out[:, :2] @ r.T
    out[:, :3] += np.asarray(params["translation"], dtype=np.float64)
    out[:, :3] *= float(params["scale"])
    if params["flip_horizontal"]:
        out[:, 1] *= -1.0
    if params["flip_vertical"]:
        out[:, 0] *= -1.0
    return out


def points_in_upright_box(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    center = np.asarray(box[:3], dtype=np.float64)
    dim_x, dim_y, height, yaw = map(float, box[3:7])

    # Cheap AABB broad phase before rotating the much smaller candidate set.
    c_abs, s_abs = abs(math.cos(yaw)), abs(math.sin(yaw))
    half_x = 0.5 * (c_abs * dim_x + s_abs * dim_y) + 1e-5
    half_y = 0.5 * (s_abs * dim_x + c_abs * dim_y) + 1e-5
    half_z = 0.5 * height + 1e-5
    broad = (
        (np.abs(points_xyz[:, 0] - center[0]) <= half_x)
        & (np.abs(points_xyz[:, 1] - center[1]) <= half_y)
        & (np.abs(points_xyz[:, 2] - center[2]) <= half_z)
    )
    idx = np.flatnonzero(broad)
    mask = np.zeros(len(points_xyz), dtype=bool)
    if not len(idx):
        return mask
    candidate = points_xyz[idx]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world_to_local = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    local_xy = (candidate[:, :2] - center[:2]) @ world_to_local
    inside = (
        (np.abs(local_xy[:, 0]) <= dim_x * 0.5 + 1e-5)
        & (np.abs(local_xy[:, 1]) <= dim_y * 0.5 + 1e-5)
        & (np.abs(candidate[:, 2] - center[2]) <= height * 0.5 + 1e-5)
    )
    mask[idx[inside]] = True
    return mask


def voxel_occupancies(
    points: np.ndarray,
    pcr: Sequence[float],
    voxel_size: Sequence[float],
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    low = np.asarray(pcr[:3], dtype=np.float64)
    high = np.asarray(pcr[3:], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    dims = np.rint((high - low) / size).astype(np.int64)
    coords = np.floor((points[:, :3] - low) / size).astype(np.int64)
    if np.any(coords < 0) or np.any(coords >= dims):
        # points should already have passed strict point_range_mask.
        valid = ((coords >= 0) & (coords < dims)).all(axis=1)
        coords = coords[valid]
    if not len(coords):
        return np.zeros(0, dtype=np.int64)
    linear = coords[:, 0] + dims[0] * (coords[:, 1] + dims[1] * coords[:, 2])
    _, counts = np.unique(linear, return_counts=True)
    return counts.astype(np.int64)


@dataclass
class PipelineBucket:
    z_bin_width: float
    frame_points_before: List[int] = field(default_factory=list)
    frame_points_after_frustum: List[int] = field(default_factory=list)
    frame_points_after_range: List[int] = field(default_factory=list)
    frame_frustum_keep_ratio: List[float] = field(default_factory=list)
    frame_range_keep_ratio_after_frustum: List[float] = field(default_factory=list)
    candidate_voxels: List[int] = field(default_factory=list)
    voxel_cap_overflow_ratio: List[float] = field(default_factory=list)
    point_cap_drop_ratio: List[float] = field(default_factory=list)
    history_lag_s: List[float] = field(default_factory=list)
    full_history: List[int] = field(default_factory=list)
    frustum_threshold_failures: int = 0
    points_dropped_by_point_cap: int = 0
    points_before_point_cap: int = 0
    occupancy_hist: OccupancyHistogram = field(default_factory=OccupancyHistogram)
    z_before: Histogram1D = field(init=False)
    z_after_frustum: Histogram1D = field(init=False)
    z_after_range: Histogram1D = field(init=False)

    def __post_init__(self):
        self.z_before = Histogram1D(self.z_bin_width)
        self.z_after_frustum = Histogram1D(self.z_bin_width)
        self.z_after_range = Histogram1D(self.z_bin_width)

    def add(
        self,
        before: np.ndarray,
        after_frustum: np.ndarray,
        after_range: np.ndarray,
        occupancies: np.ndarray,
        max_num_points: int,
        max_voxels: int,
        threshold_failed: bool,
        full_history: bool,
        oldest_lag_s: float,
    ) -> None:
        b = len(before)
        f = len(after_frustum)
        r = len(after_range)
        self.frame_points_before.append(b)
        self.frame_points_after_frustum.append(f)
        self.frame_points_after_range.append(r)
        self.frame_frustum_keep_ratio.append(safe_rate(f, b))
        self.frame_range_keep_ratio_after_frustum.append(safe_rate(r, f))
        self.z_before.update(before[:, 2] if len(before) else np.zeros(0))
        self.z_after_frustum.update(after_frustum[:, 2] if len(after_frustum) else np.zeros(0))
        self.z_after_range.update(after_range[:, 2] if len(after_range) else np.zeros(0))

        candidate = int(len(occupancies))
        self.candidate_voxels.append(candidate)
        self.voxel_cap_overflow_ratio.append(safe_rate(max(candidate - max_voxels, 0), candidate))
        dropped = int(np.maximum(occupancies - max_num_points, 0).sum()) if len(occupancies) else 0
        self.points_dropped_by_point_cap += dropped
        self.points_before_point_cap += r
        self.point_cap_drop_ratio.append(safe_rate(dropped, r))
        self.occupancy_hist.update_counts(occupancies)
        self.history_lag_s.append(float(oldest_lag_s))
        self.full_history.append(int(bool(full_history)))
        self.frustum_threshold_failures += int(bool(threshold_failed))

    def to_dict(self, configured_max_num_points: int, configured_max_voxels: int) -> dict:
        candidate = percentile_summary(self.candidate_voxels)
        return {
            "frames_or_augmentations": len(self.frame_points_before),
            "points_before_frustum_per_frame": percentile_summary(self.frame_points_before),
            "points_after_frustum_per_frame": percentile_summary(self.frame_points_after_frustum),
            "points_after_range_per_frame": percentile_summary(self.frame_points_after_range),
            "frustum_keep_ratio_per_frame": percentile_summary(self.frame_frustum_keep_ratio),
            "range_keep_ratio_after_frustum_per_frame": percentile_summary(self.frame_range_keep_ratio_after_frustum),
            "frustum_threshold_failures": int(self.frustum_threshold_failures),
            "full_requested_history_rate": float(np.mean(self.full_history)) if self.full_history else 0.0,
            "oldest_requested_history_lag_seconds": percentile_summary(self.history_lag_s),
            "point_z_histograms": {
                "before_frustum": self.z_before.to_dict(),
                "after_frustum": self.z_after_frustum.to_dict(),
                "after_range": self.z_after_range.to_dict(),
            },
            "voxelization": {
                "configured_max_num_points": int(configured_max_num_points),
                "configured_max_voxels": int(configured_max_voxels),
                "candidate_nonempty_voxels_per_frame": candidate,
                "voxel_cap_overflow_ratio_per_frame": percentile_summary(self.voxel_cap_overflow_ratio),
                "frames_over_max_voxels": int(np.sum(np.asarray(self.candidate_voxels) > configured_max_voxels)),
                "frames_over_max_voxels_fraction": float(np.mean(np.asarray(self.candidate_voxels) > configured_max_voxels)) if self.candidate_voxels else 0.0,
                "point_cap_drop_ratio_per_frame": percentile_summary(self.point_cap_drop_ratio),
                "aggregate_point_cap_drop_ratio": safe_rate(self.points_dropped_by_point_cap, self.points_before_point_cap),
                "occupied_voxel_point_count_distribution": self.occupancy_hist.to_dict(),
            },
        }


@dataclass
class SupportBucket:
    thresholds: Sequence[int]
    stored: List[float] = field(default_factory=list)
    before: List[float] = field(default_factory=list)
    after_frustum: List[float] = field(default_factory=list)
    after_range: List[float] = field(default_factory=list)
    frustum_retention: List[float] = field(default_factory=list)
    range_retention_vs_frustum: List[float] = field(default_factory=list)
    total_retention_vs_before: List[float] = field(default_factory=list)
    stored_minus_recounted: List[float] = field(default_factory=list)

    def add(self, stored: float, before: float, after_frustum: float, after_range: float) -> None:
        self.stored.append(stored)
        self.before.append(before)
        self.after_frustum.append(after_frustum)
        self.after_range.append(after_range)
        self.frustum_retention.append(safe_rate(after_frustum, before))
        self.range_retention_vs_frustum.append(safe_rate(after_range, after_frustum))
        self.total_retention_vs_before.append(safe_rate(after_range, before))
        self.stored_minus_recounted.append(float(stored - before))

    def to_dict(self) -> dict:
        n = len(self.before)
        out = {
            "gt": n,
            "stored_num_lidar_pts": percentile_summary(self.stored),
            "recounted_current_points_before_filters": percentile_summary(self.before),
            "current_points_after_frustum": percentile_summary(self.after_frustum),
            "current_points_after_frustum_and_range": percentile_summary(self.after_range),
            "frustum_retention_vs_before": percentile_summary(self.frustum_retention),
            "range_retention_vs_after_frustum": percentile_summary(self.range_retention_vs_frustum),
            "total_retention_vs_before": percentile_summary(self.total_retention_vs_before),
            "stored_minus_recounted": percentile_summary(self.stored_minus_recounted),
            "thresholds_after_frustum_and_range": {},
        }
        a = np.asarray(self.after_range, dtype=np.float64)
        for threshold in self.thresholds:
            count = int(np.sum(a < threshold)) if len(a) else 0
            out["thresholds_after_frustum_and_range"]["<{}".format(threshold)] = {
                "count": count,
                "fraction": safe_rate(count, n),
            }
        zero = int(np.sum(a == 0)) if len(a) else 0
        out["zero_after_frustum_and_range"] = {"count": zero, "fraction": safe_rate(zero, n)}
        return out


class SupportGroups:
    def __init__(self, thresholds: Sequence[int], z_bin_width: float):
        self.thresholds = thresholds
        self.z_bin_width = z_bin_width
        self.overall = SupportBucket(thresholds)
        self.by_class: Dict[str, SupportBucket] = defaultdict(lambda: SupportBucket(thresholds))
        self.by_scene: Dict[str, SupportBucket] = defaultdict(lambda: SupportBucket(thresholds))
        self.by_z_bin: Dict[str, SupportBucket] = defaultdict(lambda: SupportBucket(thresholds))
        self.outside_supported: List[dict] = []

    def add(
        self,
        split: str,
        scene: str,
        name: str,
        box: np.ndarray,
        stored: float,
        before: float,
        after_frustum: float,
        after_range: float,
        center_inside: bool,
    ) -> None:
        low = math.floor(float(box[2]) / self.z_bin_width) * self.z_bin_width
        z_label = "[{:.3f},{:.3f})".format(low, low + self.z_bin_width)
        for bucket in (self.overall, self.by_class[name], self.by_scene[scene], self.by_z_bin[z_label]):
            bucket.add(stored, before, after_frustum, after_range)
        if (not center_inside) and after_frustum > 0:
            self.outside_supported.append(
                {
                    "split": split,
                    "scene": scene,
                    "class": name,
                    "center_xyz": [float(box[0]), float(box[1]), float(box[2])],
                    "stored_num_lidar_pts": float(stored),
                    "points_before_filters": float(before),
                    "points_after_frustum_before_range": float(after_frustum),
                    "points_after_range": float(after_range),
                }
            )

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "by_class": {k: v.to_dict() for k, v in sorted(self.by_class.items())},
            "by_scene": {k: v.to_dict() for k, v in sorted(self.by_scene.items())},
            "by_center_z_bin": {k: v.to_dict() for k, v in sorted(self.by_z_bin.items())},
            "center_outside_range_but_frustum_supported": {
                "count": len(self.outside_supported),
                "records": self.outside_supported,
            },
        }


def audit_point_pipeline(
    infos_by_split: Mapping[str, Sequence[dict]],
    converted_root: Path,
    pcr: Sequence[float],
    voxel_size: Sequence[float],
    load_dim: int,
    frustum_cfg: Mapping[str, Any],
    aug_cfg: Mapping[str, Any],
    sweep_counts: Sequence[int],
    train_augment_repeats: int,
    seed: int,
    z_bin_width: float,
    gt_point_thresholds: Sequence[int],
    max_num_points: int,
    max_voxels_train: int,
    max_voxels_test: int,
) -> Tuple[dict, Dict[str, Dict[int, PipelineBucket]], Dict[str, SupportGroups]]:
    report: Dict[str, Any] = {"splits": {}}
    buckets: Dict[str, Dict[int, PipelineBucket]] = {}
    support_groups: Dict[str, SupportGroups] = {}
    max_history = max(sweep_counts) if sweep_counts else 0

    for split, infos in infos_by_split.items():
        is_train = split == "train"
        cap = max_voxels_train if is_train else max_voxels_test
        identity = {n: PipelineBucket(z_bin_width) for n in sweep_counts}
        augmented = {n: PipelineBucket(z_bin_width) for n in sweep_counts} if is_train and train_augment_repeats > 0 else {}
        support = SupportGroups(gt_point_thresholds, z_bin_width)
        rng = np.random.default_rng(seed + sum(ord(c) for c in split))
        insufficient_sweep_samples = Counter()
        aug_gt_counts = Counter()
        aug_gt_by_class: Dict[str, Counter] = defaultdict(Counter)
        aug_gt_frame_outside_fraction: List[float] = []
        aug_gt_z_hist = Histogram1D(z_bin_width)

        for info in tqdm(infos, desc="points {}".format(split), unit="frame"):
            current = load_cloud(resolve(converted_root, info["lidar_path"]), load_dim)
            if current.shape[1] >= 5:
                current[:, 4] = 0.0

            history_meta = list(info.get("sweeps", []))[:max_history]
            clouds = [current]
            for sweep in history_meta:
                cloud = load_cloud(resolve(converted_root, sweep["data_path"]), load_dim)
                clouds.append(transform_sweep_to_current(cloud, sweep, int(info["timestamp"])))

            # Frustum is point-wise. Compute once per cloud and reuse for all N.
            visible_masks = [frustum_mask(cloud, info["cams"], frustum_cfg) for cloud in clouds]

            # Per-GT support is based on CURRENT frame, matching stored num_lidar_pts.
            current_frustum_mask = visible_masks[0]
            current_range_mask = point_range_mask(current, pcr)
            boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)
            names = np.asarray(info.get("gt_names", []), dtype=object)
            valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
            stored = np.asarray(info.get("num_lidar_pts", np.zeros(len(boxes))), dtype=np.float64)
            valid_boxes = boxes[valid]
            valid_names = names[valid]
            valid_stored = stored[valid]
            center_inside = strict_center_mask(valid_boxes, pcr)
            scene = str(info.get("scene_name", "unknown"))
            for box, raw_name, stored_count, inside in zip(valid_boxes, valid_names, valid_stored, center_inside):
                name = str(raw_name)
                inside_box = points_in_upright_box(current[:, :3], box)
                before_count = int(inside_box.sum())
                after_frustum_count = int((inside_box & current_frustum_mask).sum())
                after_range_count = int((inside_box & current_frustum_mask & current_range_mask).sum())
                support.add(
                    split,
                    scene,
                    name,
                    box,
                    float(stored_count),
                    float(before_count),
                    float(after_frustum_count),
                    float(after_range_count),
                    bool(inside),
                )

            # Pre-sample augmentation params once per frame so every sweep count
            # in one repeat sees the same transform.
            aug_params = [sample_train_augmentation(rng, aug_cfg) for _ in range(train_augment_repeats)] if is_train else []

            # ObjectRangeFilter happens AFTER GlobalRotScaleTrans + RandomFlip3D.
            # Audit the exact center-range consequence separately from raw GT range.
            if is_train and aug_params and len(valid_boxes):
                low = np.asarray(pcr[:3], dtype=np.float64)
                high = np.asarray(pcr[3:], dtype=np.float64)
                for params in aug_params:
                    aug_centers = apply_train_augmentation_xyz(valid_boxes[:, :3], params)
                    aug_gt_z_hist.update(aug_centers[:, 2])
                    outside_x = (aug_centers[:, 0] <= low[0]) | (aug_centers[:, 0] >= high[0])
                    outside_y = (aug_centers[:, 1] <= low[1]) | (aug_centers[:, 1] >= high[1])
                    outside_z = (aug_centers[:, 2] <= low[2]) | (aug_centers[:, 2] >= high[2])
                    outside_xy = outside_x | outside_y
                    outside_xyz = outside_xy | outside_z
                    aug_gt_counts["gt_observations"] += len(aug_centers)
                    aug_gt_counts["outside_x"] += int(outside_x.sum())
                    aug_gt_counts["outside_y"] += int(outside_y.sum())
                    aug_gt_counts["outside_z"] += int(outside_z.sum())
                    aug_gt_counts["outside_xy_current_filter"] += int(outside_xy.sum())
                    aug_gt_counts["outside_xyz_proposed_filter"] += int(outside_xyz.sum())
                    aug_gt_counts["additional_z_only_vs_current_xy"] += int((outside_z & ~outside_xy).sum())
                    aug_gt_frame_outside_fraction.append(float(outside_xyz.mean()))
                    for raw_name, ox, oy, oz, oxy, oxyz in zip(
                        valid_names.tolist(), outside_x, outside_y, outside_z, outside_xy, outside_xyz
                    ):
                        c = aug_gt_by_class[str(raw_name)]
                        c["gt_observations"] += 1
                        c["outside_x"] += int(ox)
                        c["outside_y"] += int(oy)
                        c["outside_z"] += int(oz)
                        c["outside_xy_current_filter"] += int(oxy)
                        c["outside_xyz_proposed_filter"] += int(oxyz)
                        c["additional_z_only_vs_current_xy"] += int(oz and not oxy)

            for n in sweep_counts:
                use_hist = min(n, len(clouds) - 1)
                if len(history_meta) < n:
                    insufficient_sweep_samples[str(n)] += 1
                selected_clouds = clouds[: 1 + use_hist]
                selected_visible = visible_masks[: 1 + use_hist]
                before = np.concatenate(selected_clouds, axis=0) if selected_clouds else np.zeros((0, load_dim), dtype=np.float32)
                visible_mask = np.concatenate(selected_visible, axis=0) if selected_visible else np.zeros(0, dtype=bool)
                visible = before[visible_mask]
                threshold_failed, use_visible = effective_frustum_selection(len(before), len(visible), frustum_cfg)
                after_frustum = visible if use_visible else before
                after_range = after_frustum[point_range_mask(after_frustum, pcr)]
                occupancies = voxel_occupancies(after_range, pcr, voxel_size)
                if n == 0:
                    lag = 0.0
                elif len(history_meta) >= n:
                    lag = float(info["timestamp"] - history_meta[n - 1]["timestamp"]) / 1e6
                elif history_meta:
                    lag = float(info["timestamp"] - history_meta[-1]["timestamp"]) / 1e6
                else:
                    lag = 0.0
                identity[n].add(
                    before,
                    after_frustum,
                    after_range,
                    occupancies,
                    max_num_points,
                    cap,
                    threshold_failed,
                    len(history_meta) >= n,
                    lag,
                )

                if is_train and train_augment_repeats > 0:
                    for params in aug_params:
                        aug_xyz = apply_train_augmentation_xyz(after_frustum[:, :3], params)
                        aug_points = after_frustum.copy()
                        aug_points[:, :3] = aug_xyz
                        aug_after_range = aug_points[point_range_mask(aug_points, pcr)]
                        aug_occupancies = voxel_occupancies(aug_after_range, pcr, voxel_size)
                        # before/after-frustum counts/hists are repeated intentionally
                        # so each augmented observation has a complete denominator.
                        augmented[n].add(
                            before,
                            after_frustum,
                            aug_after_range,
                            aug_occupancies,
                            max_num_points,
                            cap,
                            threshold_failed,
                            len(history_meta) >= n,
                            lag,
                        )

        split_report = {
            "frames": len(infos),
            "insufficient_requested_history_frames": dict(insufficient_sweep_samples),
            "per_gt_current_frame_support": support.to_dict(),
            "sweeps": {},
        }
        if is_train and train_augment_repeats > 0:
            total_aug = int(aug_gt_counts.get("gt_observations", 0))
            split_report["train_augmented_gt_center_range"] = {
                "augment_repeats_per_frame": int(train_augment_repeats),
                "gt_observations": total_aug,
                "counts": dict(aug_gt_counts),
                "fractions": {
                    key: safe_rate(value, total_aug)
                    for key, value in aug_gt_counts.items()
                    if key != "gt_observations"
                },
                "per_frame_outside_xyz_fraction": percentile_summary(aug_gt_frame_outside_fraction),
                "center_z_histogram_after_augmentation": aug_gt_z_hist.to_dict(),
                "by_class": {
                    name: {
                        "counts": dict(counter),
                        "fractions": {
                            key: safe_rate(value, counter.get("gt_observations", 0))
                            for key, value in counter.items()
                            if key != "gt_observations"
                        },
                    }
                    for name, counter in sorted(aug_gt_by_class.items())
                },
                "interpretation": {
                    "outside_xy_current_filter": "what current ObjectRangeFilter removes after train augmentation",
                    "outside_xyz_proposed_filter": "what UAVObjectRangeFilter3D would remove after train augmentation",
                    "additional_z_only_vs_current_xy": "extra GT removed only because proposed filter also checks Z",
                },
            }
        for n in sweep_counts:
            split_report["sweeps"][str(n)] = {
                "historical_sweeps": n,
                "total_lidar_frames_per_prediction": n + 1,
                "identity": identity[n].to_dict(max_num_points, cap),
            }
            if n in augmented:
                split_report["sweeps"][str(n)]["train_augmented"] = augmented[n].to_dict(max_num_points, cap)
        report["splits"][split] = split_report
        buckets[split] = identity
        if augmented:
            # Keep augmented buckets under a synthetic key for recommendation logic.
            buckets[split + "__augmented"] = augmented
        support_groups[split] = support

    return report, buckets, support_groups


# -----------------------------------------------------------------------------
# Recommendation logic
# -----------------------------------------------------------------------------


def central_percentile_bounds(values: np.ndarray, coverage: float) -> Tuple[float, float]:
    tail = (100.0 - coverage) / 2.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


def recommend_range(
    rows_by_split: Mapping[str, Sequence[dict]],
    supported_outside_records: Sequence[dict],
    pcr: Sequence[float],
    voxel_size: Sequence[float],
    camera_xbound: Optional[Sequence[float]],
    camera_ybound: Optional[Sequence[float]],
    margin: float,
    coverage: float,
    min_outside_fraction: float,
) -> Tuple[List[float], dict]:
    design = list(rows_by_split.get("train", [])) + list(rows_by_split.get("val", []))
    current = list(map(float, pcr))
    if not design:
        return current, {"reason": "no train+val GT"}

    xyz = np.asarray([[r["x"], r["y"], r["center_z"]] for r in design], dtype=np.float64)
    low = np.asarray(current[:3], dtype=np.float64)
    high = np.asarray(current[3:], dtype=np.float64)
    outside = ~((xyz > low) & (xyz < high)).all(axis=1)
    outside_fraction = float(np.mean(outside))
    supported_fraction = safe_rate(len(supported_outside_records), len(xyz))
    detail = {
        "design_gt": int(len(xyz)),
        "center_outside_current_range": int(outside.sum()),
        "center_outside_fraction": outside_fraction,
        "center_outside_and_camera_frustum_supported": int(len(supported_outside_records)),
        "center_outside_and_camera_frustum_supported_fraction": supported_fraction,
        "coverage_percentile": float(coverage),
        "margin_m": float(margin),
    }
    if supported_fraction < min_outside_fraction:
        detail["reason"] = (
            "camera-frustum-supported outside fraction below automatic expansion threshold; "
            "keep ROI and let the unified evaluator exclude out-of-ROI GT"
        )
        return current, detail

    desired_low = low.copy()
    desired_high = high.copy()
    for axis in range(3):
        q_low, q_high = central_percentile_bounds(xyz[:, axis], coverage)
        desired_low[axis] = min(low[axis], q_low - margin)
        desired_high[axis] = max(high[axis], q_high + margin)

    # XY must stay compatible with camera BEV step if available. Z only needs
    # LiDAR voxel alignment because the current camera zbound intentionally uses
    # one vertical bin spanning the full Z range.
    xy_steps = [float(voxel_size[0]), float(voxel_size[1])]
    if camera_xbound is not None and len(camera_xbound) >= 3:
        xy_steps[0] = max(xy_steps[0], float(camera_xbound[2]))
    if camera_ybound is not None and len(camera_ybound) >= 3:
        xy_steps[1] = max(xy_steps[1], float(camera_ybound[2]))
    steps = [xy_steps[0], xy_steps[1], float(voxel_size[2])]
    for axis in range(3):
        desired_low[axis] = snap_down(float(desired_low[axis]), steps[axis])
        desired_high[axis] = snap_up(float(desired_high[axis]), steps[axis])

    recommended = desired_low.tolist() + desired_high.tolist()
    detail["recommended"] = recommended
    return recommended, detail


def bucket_for_recommendation(
    buckets: Mapping[str, Dict[int, PipelineBucket]],
    split: str,
    sweep: int,
    prefer_augmented: bool,
) -> Optional[PipelineBucket]:
    if prefer_augmented and split + "__augmented" in buckets:
        return buckets[split + "__augmented"].get(sweep)
    return buckets.get(split, {}).get(sweep)


def merge_values(buckets: Sequence[PipelineBucket], attr: str) -> List[Any]:
    out = []
    for bucket in buckets:
        if bucket is not None:
            out.extend(getattr(bucket, attr))
    return out


def merge_occupancy_hist(buckets: Sequence[PipelineBucket]) -> OccupancyHistogram:
    out = OccupancyHistogram()
    for bucket in buckets:
        if bucket is None:
            continue
        for occupancy, n in bucket.occupancy_hist.counts.items():
            out.counts[int(occupancy)] += int(n)
            out.num_voxels += int(n)
    return out


def build_recommendations(
    config: dict,
    consistency: dict,
    rows_by_split: Mapping[str, Sequence[dict]],
    point_buckets: Mapping[str, Dict[int, PipelineBucket]],
    support_groups: Mapping[str, SupportGroups],
    args: argparse.Namespace,
) -> Tuple[dict, dict]:
    changes: Dict[str, Any] = {}
    rationale: Dict[str, Any] = {}
    pcr = config["point_cloud_range"]
    voxel = config["voxel_size"]
    cam = config["model"]["camera_vtransform"]
    head = config["model"]["head"]

    supported_outside_records = []
    for split in ("train", "val"):
        if split in support_groups:
            supported_outside_records.extend(support_groups[split].outside_supported)

    recommended_pcr, range_detail = recommend_range(
        rows_by_split,
        supported_outside_records,
        pcr,
        voxel,
        cam.get("xbound"),
        cam.get("ybound"),
        args.range_margin,
        args.range_robust_percentile,
        args.range_expand_min_outside_fraction,
    )
    rationale["point_cloud_range"] = range_detail

    effective_pcr = pcr
    if not close_list(recommended_pcr, pcr):
        effective_pcr = recommended_pcr
        set_nested(changes, ["point_cloud_range"], recommended_pcr)

        expected = project_expected_grid(effective_pcr, voxel)["project_grid_sparse_convention"]
        set_nested(changes, ["model", "encoders", "lidar", "backbone", "sparse_shape"], expected)
        set_nested(changes, ["model", "heads", "object", "train_cfg", "grid_size"], expected)
        set_nested(changes, ["model", "heads", "object", "test_cfg", "grid_size"], expected)

        if cam.get("xbound") is not None:
            xb = list(map(float, cam["xbound"]))
            set_nested(changes, ["model", "encoders", "camera", "vtransform", "xbound"], [effective_pcr[0], effective_pcr[3], xb[2]])
        if cam.get("ybound") is not None:
            yb = list(map(float, cam["ybound"]))
            set_nested(changes, ["model", "encoders", "camera", "vtransform", "ybound"], [effective_pcr[1], effective_pcr[4], yb[2]])
        if cam.get("zbound") is not None:
            # Preserve the current design: a single vertical camera BEV bin.
            set_nested(
                changes,
                ["model", "encoders", "camera", "vtransform", "zbound"],
                [effective_pcr[2], effective_pcr[5], effective_pcr[5] - effective_pcr[2]],
            )

    # Explicit post_center_range is currently wider than the model input range.
    # Keep it identical to the final physical ROI.
    if head.get("post_center_range") is not None and not close_list(head["post_center_range"], effective_pcr):
        set_nested(changes, ["model", "heads", "object", "bbox_coder", "post_center_range"], effective_pcr)
        rationale["post_center_range"] = {
            "current": head["post_center_range"],
            "recommended": effective_pcr,
            "reason": "prediction center ROI should match the unified physical detection/evaluation ROI",
        }

    # If camera endpoints are already inconsistent even without a range change,
    # recommend only endpoint correction while preserving their resolution step.
    for axis_name, pcr_pair, path in (
        ("xbound", [effective_pcr[0], effective_pcr[3]], ["model", "encoders", "camera", "vtransform", "xbound"]),
        ("ybound", [effective_pcr[1], effective_pcr[4]], ["model", "encoders", "camera", "vtransform", "ybound"]),
    ):
        bound = cam.get(axis_name)
        if bound is not None and not close_list(bound[:2], pcr_pair):
            set_nested(changes, path, [pcr_pair[0], pcr_pair[1], float(bound[2])])
    zbound = cam.get("zbound")
    if zbound is not None and not close_list(zbound[:2], [effective_pcr[2], effective_pcr[5]]):
        set_nested(
            changes,
            ["model", "encoders", "camera", "vtransform", "zbound"],
            [effective_pcr[2], effective_pcr[5], effective_pcr[5] - effective_pcr[2]],
        )

    # Grid/sparse consistency (project convention: z bins + 1).
    expected_shape = project_expected_grid(effective_pcr, voxel)["project_grid_sparse_convention"]
    if config["model"].get("sparse_shape") is not None and not close_list(config["model"]["sparse_shape"], expected_shape):
        set_nested(changes, ["model", "encoders", "lidar", "backbone", "sparse_shape"], expected_shape)
    if head.get("train_grid_size") is not None and not close_list(head["train_grid_size"], expected_shape):
        set_nested(changes, ["model", "heads", "object", "train_cfg", "grid_size"], expected_shape)
    if head.get("test_grid_size") is not None and not close_list(head["test_grid_size"], expected_shape):
        set_nested(changes, ["model", "heads", "object", "test_cfg", "grid_size"], expected_shape)

    # Voxel capacities are recommended for the sweep count currently configured
    # for the model. Train uses augmented full-frame observations; val+test use
    # identity observations.
    configured_sweep = int(config["max_sweeps"])
    max_voxels = config["model"]["lidar_voxelize"].get("max_voxels")
    max_num_points = config["model"]["lidar_voxelize"].get("max_num_points")
    if max_voxels is not None and configured_sweep in args.sweeps:
        train_bucket = bucket_for_recommendation(point_buckets, "train", configured_sweep, True)
        eval_buckets = [
            bucket_for_recommendation(point_buckets, split, configured_sweep, False)
            for split in ("val", "test")
        ]
        train_candidates = [] if train_bucket is None else train_bucket.candidate_voxels
        eval_candidates = merge_values([b for b in eval_buckets if b is not None], "candidate_voxels")
        current_train, current_test = map(int, max_voxels)
        rec_train = current_train
        rec_test = current_test
        if train_candidates:
            target = float(np.percentile(np.asarray(train_candidates), args.capacity_percentile))
            target = ceil_step(target, args.cap_step)
            if target > current_train:
                rec_train = target
        if eval_candidates:
            target = float(np.percentile(np.asarray(eval_candidates), args.capacity_percentile))
            target = ceil_step(target, args.cap_step)
            if target > current_test:
                rec_test = target
        if [rec_train, rec_test] != [current_train, current_test]:
            set_nested(changes, ["model", "encoders", "lidar", "voxelize", "max_voxels"], [rec_train, rec_test])
        rationale["max_voxels"] = {
            "configured_sweep": configured_sweep,
            "capacity_percentile": args.capacity_percentile,
            "current": [current_train, current_test],
            "recommended": [rec_train, rec_test],
            "train_candidate_voxels": percentile_summary(train_candidates),
            "eval_candidate_voxels": percentile_summary(eval_candidates),
        }

        if max_num_points is not None:
            occupancy_buckets = [b for b in [train_bucket] + eval_buckets if b is not None]
            occupancy = merge_occupancy_hist(occupancy_buckets)
            dropped = sum(b.points_dropped_by_point_cap for b in occupancy_buckets)
            before = sum(b.points_before_point_cap for b in occupancy_buckets)
            drop_ratio = safe_rate(dropped, before)
            target_occ = occupancy.percentile(args.capacity_percentile)
            recommended_max_points = int(max_num_points)
            if (
                target_occ is not None
                and drop_ratio > args.max_point_cap_drop_ratio
                and target_occ > max_num_points
            ):
                recommended_max_points = min(
                    int(math.ceil(target_occ)), args.max_recommended_points_per_voxel
                )
                if recommended_max_points > max_num_points:
                    set_nested(
                        changes,
                        ["model", "encoders", "lidar", "voxelize", "max_num_points"],
                        recommended_max_points,
                    )
            rationale["max_num_points"] = {
                "current": int(max_num_points),
                "recommended": int(recommended_max_points),
                "aggregate_point_drop_ratio": drop_ratio,
                "capacity_percentile": args.capacity_percentile,
                "occupancy_at_percentile": target_occ,
                "automatic_ceiling": args.max_recommended_points_per_voxel,
            }
    elif max_voxels is not None:
        rationale["max_voxels"] = {
            "reason": "configured max_sweeps={} was not included in --sweeps; no automatic capacity recommendation".format(configured_sweep)
        }

    return changes, rationale


# -----------------------------------------------------------------------------
# Warnings / terminal summary
# -----------------------------------------------------------------------------


def build_warnings(
    config: dict,
    consistency: dict,
    gt_report: dict,
    point_report: dict,
    recommendations: dict,
) -> List[str]:
    warnings: List[str] = []
    for name, ok in consistency["checks"].items():
        if ok is False:
            warnings.append("config inconsistency: {}".format(name))

    # Strong split-distribution signals; thresholds are diagnostic only.
    comparisons = gt_report.get("split_comparison", {})
    for pair, data in comparisons.items():
        js = data.get("center_z_js_divergence_natural_log", 0.0)
        if js >= 0.05:
            warnings.append("{} center-Z distribution shift is notable (JS={:.4f})".format(pair, js))
        for cls, cls_data in data.get("per_class_center_z", {}).items():
            cls_js = cls_data.get("js_divergence_natural_log", 0.0)
            if cls_js >= 0.10 and cls_data.get("gt_a", 0) and cls_data.get("gt_b", 0):
                warnings.append("{} / {} center-Z shift is strong (JS={:.4f})".format(pair, cls, cls_js))

    for split, data in gt_report.get("splits", {}).items():
        outside = data["range_audit"]["center_outside_xyz"]
        if outside["count"]:
            warnings.append(
                "{} has {} valid GT centers outside point_cloud_range ({:.4%})".format(
                    split, outside["count"], outside["fraction_of_valid_gt"]
                )
            )

    for split, data in point_report.get("splits", {}).items():
        supported = data["per_gt_current_frame_support"]["center_outside_range_but_frustum_supported"]["count"]
        if supported:
            warnings.append(
                "{} has {} GT centers outside point_cloud_range that still have current LiDAR points after camera-frustum filtering".format(split, supported)
            )
        for sweep, sweep_data in data.get("sweeps", {}).items():
            identity = sweep_data["identity"]
            if identity["frustum_threshold_failures"]:
                warnings.append(
                    "{} S{} would trigger LiDARCameraFrustumFilter threshold failure on {} full-frame observations".format(
                        split, sweep, identity["frustum_threshold_failures"]
                    )
                )

    if recommendations:
        warnings.append("recommended_changes.yaml contains parameter changes; inspect analysis.json rationale before applying them")
    return warnings


def print_terminal_summary(report: dict, changes: dict, output_dir: Path) -> None:
    print("\n=== UAV full audit summary ===")
    gt = report["converted_gt"]["splits"]
    for split in report["run"]["splits"]:
        if split not in gt:
            continue
        r = gt[split]
        outside = r["range_audit"]["center_outside_xyz"]
        z = r["distribution"].get("summary", {}).get("center_z")
        z_text = "no GT" if not z else "z=[{:.3f}, {:.3f}], median={:.3f}".format(z["min"], z["max"], z["p50"])
        print(
            "{}: frames={}, valid_gt={}, {}, center_outside_xyz={} ({:.4%})".format(
                split,
                r["frames"],
                r["valid_gt"],
                z_text,
                outside["count"],
                outside["fraction_of_valid_gt"],
            )
        )

    print("\nRecommended parameter changes:")
    if changes:
        print(yaml.safe_dump(changes, sort_keys=False, allow_unicode=True).rstrip())
    else:
        print("  none")
    print("\nOutput directory: {}".format(output_dir.resolve()))
    print("  analysis.json")
    print("  recommended_changes.yaml")


# -----------------------------------------------------------------------------
# Self-check
# -----------------------------------------------------------------------------


def self_check() -> None:
    pcr = [-1, -1, -1, 1, 1, 1]
    boxes = np.asarray(
        [
            [0, 0, 0, 1, 1, 1, 0],
            [2, 0, 0, 1, 1, 1, 0],
            [0.8, 0, 0, 1, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    masks = box_range_masks(boxes, pcr)
    assert masks["center_inside"].tolist() == [True, False, True]
    assert masks["partly_or_fully_outside"].tolist() == [False, True, True]
    assert masks["fully_outside"].tolist() == [False, True, False]

    points = np.asarray([[0, 0, 0], [0.9, 0.9, 0.9], [1.0, 0, 0]], dtype=np.float64)
    assert point_range_mask(points, pcr).tolist() == [True, True, False]

    occ = voxel_occupancies(
        np.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float64),
        [0, 0, 0, 2, 2, 2],
        [1, 1, 1],
    )
    assert sorted(occ.tolist()) == [1, 2]

    hist = Histogram1D(0.5)
    hist.update(np.asarray([-0.1, 0.1, 0.6]))
    assert hist.count == 3
    print("self-check passed")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return

    if "{split}" not in args.source_pattern:
        raise SystemExit("--source-pattern must contain {split}")
    if any(n < 0 for n in args.sweeps):
        raise SystemExit("--sweeps values must be >= 0")
    if args.train_augment_repeats < 0:
        raise SystemExit("--train-augment-repeats must be >= 0")
    if args.z_bin_width <= 0 or args.xy_bin_width <= 0 or args.radius_bin_width <= 0:
        raise SystemExit("histogram bin widths must be > 0")
    if not 50.0 < args.range_robust_percentile < 100.0:
        raise SystemExit("--range-robust-percentile must be in (50, 100)")
    if not 50.0 < args.capacity_percentile <= 100.0:
        raise SystemExit("--capacity-percentile must be in (50, 100]")
    if args.cap_step <= 0:
        raise SystemExit("--cap-step must be > 0")

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_cfg(str(config_path))
    config = extract_config(cfg, config_path)
    consistency = config_consistency_report(config)

    converted_root = Path(args.converted_root or config["dataset_root"]).expanduser().resolve()
    if not converted_root.is_dir():
        raise FileNotFoundError("Converted root does not exist: {}".format(converted_root))

    sweep_counts = sorted(set(int(n) for n in args.sweeps))
    max_requested_sweep = max(sweep_counts) if sweep_counts else 0
    infos_by_split: Dict[str, List[dict]] = {}
    info_meta_by_split: Dict[str, dict] = {}
    info_paths: Dict[str, str] = {}
    run_warnings: List[str] = []

    for split in args.splits:
        path = converted_root / args.source_pattern.format(split=split)
        if not path.is_file():
            raise FileNotFoundError("Missing info file for {}: {}".format(split, path))
        infos, metadata = load_infos(path)
        infos_by_split[split] = infos
        info_meta_by_split[split] = metadata
        info_paths[split] = str(path)
        source_max = metadata.get("max_sweeps")
        if source_max is not None and int(source_max) < max_requested_sweep:
            run_warnings.append(
                "{} source metadata.max_sweeps={} < requested S{}; S{} statistics will use only available history".format(
                    split, source_max, max_requested_sweep, max_requested_sweep
                )
            )

    run_name = args.run_name or "uav_dataset_analysis_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = Path(args.output_root).expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    report: Dict[str, Any] = {
        "run": {
            "config": str(config_path),
            "raw_root": None if args.raw_root is None else str(Path(args.raw_root).expanduser().resolve()),
            "converted_root": str(converted_root),
            "source_pattern": args.source_pattern,
            "info_files": info_paths,
            "info_metadata": info_meta_by_split,
            "splits": list(args.splits),
            "sweeps": sweep_counts,
            "sweep_semantics": "N historical sweeps + current frame",
            "frame_sampling": "none; every info frame is scanned",
            "train_identity_audit": True,
            "train_augment_repeats": args.train_augment_repeats,
            "seed": args.seed,
        },
        "resolved_config": config,
        "config_consistency": consistency,
    }

    if args.raw_root and not args.skip_raw:
        report["raw_recorder"] = audit_raw_labels(
            Path(args.raw_root).expanduser(), config["object_classes"]
        )

    gt_report, rows_by_split = audit_gt(
        infos_by_split,
        config["point_cloud_range"],
        config["object_classes"],
        args.z_bin_width,
        args.xy_bin_width,
        args.radius_bin_width,
        args.z_span_thresholds,
    )
    report["converted_gt"] = gt_report

    max_num_points = config["model"]["lidar_voxelize"].get("max_num_points")
    max_voxels = config["model"]["lidar_voxelize"].get("max_voxels")
    if max_num_points is None or max_voxels is None:
        raise RuntimeError("Config must contain model.encoders.lidar.voxelize.max_num_points/max_voxels")

    point_report, point_buckets, support_groups = audit_point_pipeline(
        infos_by_split,
        converted_root,
        config["point_cloud_range"],
        config["voxel_size"],
        config["load_dim"],
        config["frustum_filter"],
        config["augment3d"],
        sweep_counts,
        args.train_augment_repeats,
        args.seed,
        args.z_bin_width,
        args.gt_point_thresholds,
        int(max_num_points),
        int(max_voxels[0]),
        int(max_voxels[1]),
    )
    report["point_pipeline"] = point_report

    changes, rationale = build_recommendations(
        config,
        consistency,
        rows_by_split,
        point_buckets,
        support_groups,
        args,
    )
    report["recommendation_rationale"] = rationale
    report["warnings"] = run_warnings + build_warnings(
        config, consistency, gt_report, point_report, changes
    )

    analysis_path = output_dir / "analysis.json"
    recommendation_path = output_dir / "recommended_changes.yaml"
    analysis_path.write_text(
        json.dumps(as_plain(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    recommendation_path.write_text(
        yaml.safe_dump(as_plain(changes), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print_terminal_summary(report, changes, output_dir)


if __name__ == "__main__":
    main()
