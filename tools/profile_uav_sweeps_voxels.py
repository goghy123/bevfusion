#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile UAV multi-sweep point clouds and estimate reasonable max_voxels values.

Run this script from the BEVFusion project root, e.g.:

python tools/profile_uav_sweeps_voxels.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --converted-root data/uavdataset \
  --sweeps 3 6 9 \
  --train-augment-repeats 2 \
  --output reports/uav_sweep_voxel_profile.json

Notes
-----
- sweeps=N means N historical sweeps PLUS the current LiDAR frame.
- By default all samples in every matching info pkl are scanned.
- The script discovers info files with --info-glob, so future appended/rebuilt
  train/val/test info files are picked up without hard-coding sample counts.
- It uses the point-cloud range, voxel size, frustum settings, load_dim,
  train augmentation range, and current max_voxels from the supplied config.
- It always selects the nearest N historical sweeps. This matches test-time
  behavior and is the recommended behavior for a clean 3/6/9 sweep ablation.
"""

import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.utils import recursive_eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Recursive BEVFusion YAML config.")
    p.add_argument(
        "--converted-root",
        default=None,
        help="Converted dataset root. Default: dataset_root from config.",
    )
    p.add_argument(
        "--info-glob",
        default="uavdataset_infos_*.pkl",
        help="Info-file glob below converted root.",
    )
    p.add_argument(
        "--sweeps",
        type=int,
        nargs="+",
        default=[3, 6, 9],
        help="Historical sweep counts to profile. Current frame is extra.",
    )
    p.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="0 means all samples; positive values evenly sample each info file.",
    )
    p.add_argument(
        "--train-augment-repeats",
        type=int,
        default=2,
        help="Monte-Carlo repeats for train 3D augmentation. 0 disables train augmentation.",
    )
    p.add_argument(
        "--candidate-caps",
        type=int,
        nargs="*",
        default=None,
        help="Optional max_voxels candidates. If omitted, candidates are auto-generated.",
    )
    p.add_argument(
        "--cap-step",
        type=int,
        default=5000,
        help="Round recommended capacities up to this step.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default="uav_sweep_voxel_profile.json",
        help="Output JSON path.",
    )
    return p.parse_args()


def load_cfg(path):
    configs.load(path, recursive=True)
    return Config(recursive_eval(configs), filename=path)


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def choose_evenly(items, limit):
    items = list(items)
    if limit <= 0 or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[i] for i in idx]


def percentile_summary(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "p99_5": float(np.percentile(a, 99.5)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def ceil_step(value, step):
    return int(math.ceil(float(value) / step) * step)


def load_cloud(path, load_dim):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % load_dim:
        raise ValueError(f"{path}: {arr.size} floats not divisible by load_dim={load_dim}")
    return arr.reshape(-1, load_dim).copy()


def camera_lidar2image(cam):
    # info stores camera -> current lidar/reference transform.
    r = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)

    camera2lidar = np.eye(4, dtype=np.float64)
    camera2lidar[:3, :3] = r
    camera2lidar[:3, 3] = t

    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(cam["camera_intrinsics"], dtype=np.float64)
    return intrinsic @ np.linalg.inv(camera2lidar)


def frustum_mask(points, cams, frustum_cfg):
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    xyz1 = np.ones((n, 4), dtype=np.float64)
    xyz1[:, :3] = points[:, :3]

    mode = str(frustum_cfg.get("mode", "union"))
    if mode == "union":
        keep = np.zeros(n, dtype=bool)
    elif mode == "intersection":
        keep = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"Unsupported frustum mode: {mode}")

    min_depth = float(frustum_cfg.get("min_depth", 0.05))
    max_depth = frustum_cfg.get("max_depth", None)
    max_depth = None if max_depth is None else float(max_depth)
    margin = float(frustum_cfg.get("margin_px", 2.0))

    for cam in cams.values():
        mat = camera_lidar2image(cam)
        proj = xyz1 @ mat.T
        depth = proj[:, 2]

        valid = np.isfinite(depth) & (depth > min_depth)
        if max_depth is not None:
            valid &= depth <= max_depth

        u = np.full(n, np.nan, dtype=np.float64)
        v = np.full(n, np.nan, dtype=np.float64)
        u[valid] = proj[valid, 0] / depth[valid]
        v[valid] = proj[valid, 1] / depth[valid]

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


def range_mask(points, pcr):
    low = np.asarray(pcr[:3], dtype=np.float64)
    high = np.asarray(pcr[3:], dtype=np.float64)
    return ((points[:, :3] >= low) & (points[:, :3] < high)).all(axis=1)


def unique_voxel_count(points, pcr, voxel_size):
    if len(points) == 0:
        return 0
    low = np.asarray(pcr[:3], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    coors = np.floor((points[:, :3] - low) / size).astype(np.int32)
    return int(np.unique(coors, axis=0).shape[0])


def transform_sweep_to_current(points, sweep, current_ts_us):
    points = points.copy()
    r = np.asarray(sweep["sensor2lidar_rotation"], dtype=np.float32)
    t = np.asarray(sweep["sensor2lidar_translation"], dtype=np.float32)
    points[:, :3] = points[:, :3] @ r.T
    points[:, :3] += t
    if points.shape[1] >= 5:
        points[:, 4] = float(current_ts_us - sweep["timestamp"]) / 1e6
    return points


def load_visible_clouds(info, root, load_dim, frustum_cfg, max_history):
    current = load_cloud(resolve(root, info["lidar_path"]), load_dim)
    if current.shape[1] >= 5:
        current[:, 4] = 0.0

    clouds = [current]
    sweep_meta = list(info.get("sweeps", []))[:max_history]
    for sweep in sweep_meta:
        p = load_cloud(resolve(root, sweep["data_path"]), load_dim)
        clouds.append(transform_sweep_to_current(p, sweep, info["timestamp"]))

    # Frustum filtering happens after multi-sweep loading, but it is point-wise;
    # filtering each cloud separately gives the same retained union.
    visible = []
    for cloud in clouds:
        mask = frustum_mask(cloud, info["cams"], frustum_cfg)
        visible.append(cloud[mask])

    lags = [
        float(info["timestamp"] - s["timestamp"]) / 1e6
        for s in sweep_meta
    ]
    return visible, lags, len(info.get("sweeps", []))


def apply_train_aug(points, rng, aug_cfg):
    if len(points) == 0:
        return points
    out = points.copy()

    scale_lim = list(aug_cfg.get("scale", [1.0, 1.0]))
    rot_lim = list(aug_cfg.get("rotate", [0.0, 0.0]))
    trans_lim = float(aug_cfg.get("translate", 0.0))

    scale = rng.uniform(float(scale_lim[0]), float(scale_lim[1]))
    theta = rng.uniform(float(rot_lim[0]), float(rot_lim[1]))
    translation = rng.normal(0.0, trans_lim, size=3)

    # Project code: points.rotate(-theta), translate, scale.
    c, s = math.cos(-theta), math.sin(-theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    out[:, :2] = out[:, :2] @ r.T
    out[:, :3] += translation
    out[:, :3] *= scale

    # RandomFlip3D: horizontal and vertical independently with p=0.5.
    if rng.integers(0, 2):
        out[:, 1] *= -1.0
    if rng.integers(0, 2):
        out[:, 0] *= -1.0
    return out


def split_name_from_path(path):
    stem = path.stem
    marker = "_infos_"
    return stem.split(marker, 1)[1] if marker in stem else stem


def make_candidates(stats, configured_cap, user_caps, step):
    if user_caps:
        caps = sorted(set(int(x) for x in user_caps if x > 0))
    else:
        q = stats
        caps = {
            int(configured_cap),
            ceil_step(q["p90"], step),
            ceil_step(q["p95"], step),
            ceil_step(q["p99"], step),
            ceil_step(q["p99_5"], step),
            ceil_step(q["max"], step),
        }
        caps = sorted(x for x in caps if x > 0)
    return caps


def cap_table(values, caps):
    a = np.asarray(values, dtype=np.int64)
    out = {}
    for cap in caps:
        excess = np.maximum(a - cap, 0)
        out[str(cap)] = {
            "samples_over_cap": int((a > cap).sum()),
            "sample_overflow_rate": float((a > cap).mean()),
            "mean_candidate_voxel_drop_ratio": float(
                np.mean(excess / np.maximum(a, 1))
            ),
            "p99_candidate_voxel_drop_ratio": float(
                np.percentile(excess / np.maximum(a, 1), 99)
            ),
        }
    return out


def main():
    args = parse_args()
    if any(x < 0 for x in args.sweeps):
        raise SystemExit("--sweeps values must be >= 0")
    if args.cap_step <= 0:
        raise SystemExit("--cap-step must be > 0")

    cfg = load_cfg(args.config)
    root = Path(args.converted_root or cfg.dataset_root).resolve()
    info_paths = sorted(root.glob(args.info_glob))
    if not info_paths:
        raise FileNotFoundError(
            f"No info files matched {args.info_glob!r} under {root}"
        )

    pcr = list(map(float, cfg.point_cloud_range))
    voxel_size = list(map(float, cfg.voxel_size))
    load_dim = int(cfg.load_dim)
    frustum_cfg = dict(cfg.frustum_filter)
    aug_cfg = dict(cfg.augment3d)

    max_voxels_cfg = cfg.model.encoders.lidar.voxelize.max_voxels
    configured_train_cap = int(max_voxels_cfg[0])
    configured_test_cap = int(max_voxels_cfg[1])

    sweep_counts = sorted(set(args.sweeps))
    max_history = max(sweep_counts) if sweep_counts else 0

    report = {
        "config": {
            "config_file": str(Path(args.config).resolve()),
            "converted_root": str(root),
            "info_glob": args.info_glob,
            "info_files": [str(p) for p in info_paths],
            "sweeps_profiled": sweep_counts,
            "sweep_semantics": "N historical sweeps + current frame",
            "sweep_selection": "nearest_N",
            "point_cloud_range": pcr,
            "voxel_size": voxel_size,
            "load_dim": load_dim,
            "frustum_filter": frustum_cfg,
            "augment3d": aug_cfg,
            "configured_max_voxels": [
                configured_train_cap,
                configured_test_cap,
            ],
            "max_samples_per_split": args.max_samples_per_split,
            "train_augment_repeats": args.train_augment_repeats,
            "seed": args.seed,
        },
        "splits": {},
        "combined": {},
    }

    combined = {
        n: {
            "candidate_voxels": [],
            "points_after_filters": [],
            "oldest_history_lag_s": [],
            "full_history": [],
        }
        for n in sweep_counts
    }

    for info_path in info_paths:
        split = split_name_from_path(info_path)
        with info_path.open("rb") as f:
            raw = pickle.load(f)
        infos = list(raw["infos"] if isinstance(raw, dict) and "infos" in raw else raw)
        selected_infos = choose_evenly(infos, args.max_samples_per_split)

        per_sweep = {
            n: {
                "candidate_voxels": [],
                "points_after_filters": [],
                "oldest_history_lag_s": [],
                "full_history": [],
            }
            for n in sweep_counts
        }

        is_train = split.lower().startswith("train")
        aug_repeats = args.train_augment_repeats if is_train else 1
        if aug_repeats <= 0:
            aug_repeats = 1
        rng = np.random.default_rng(args.seed + sum(ord(c) for c in split))

        for sample_idx, info in enumerate(selected_infos):
            visible_clouds, lags, available = load_visible_clouds(
                info, root, load_dim, frustum_cfg, max_history
            )

            # Reuse the same augmentation parameters across sweep counts within
            # one repeat, so differences primarily reflect sweep count.
            for rep in range(aug_repeats):
                aug_params_seed = int(rng.integers(0, 2**31 - 1))
                for n in sweep_counts:
                    use_hist = min(n, len(visible_clouds) - 1)
                    points = np.concatenate(
                        visible_clouds[: 1 + use_hist], axis=0
                    )

                    if is_train and args.train_augment_repeats > 0:
                        local_rng = np.random.default_rng(aug_params_seed)
                        points = apply_train_aug(points, local_rng, aug_cfg)

                    points = points[range_mask(points, pcr)]
                    voxels = unique_voxel_count(points, pcr, voxel_size)

                    d = per_sweep[n]
                    d["candidate_voxels"].append(voxels)
                    d["points_after_filters"].append(int(len(points)))
                    d["full_history"].append(int(available >= n))

                    if n == 0:
                        lag = 0.0
                    elif len(lags) >= n:
                        lag = float(lags[n - 1])
                    elif lags:
                        lag = float(lags[-1])
                    else:
                        lag = 0.0
                    d["oldest_history_lag_s"].append(lag)

        split_report = {
            "info_file": str(info_path),
            "total_samples_in_info": len(infos),
            "profiled_samples": len(selected_infos),
            "sweeps": {},
        }

        for n in sweep_counts:
            d = per_sweep[n]
            voxel_stats = percentile_summary(d["candidate_voxels"])
            point_stats = percentile_summary(d["points_after_filters"])
            lag_stats = percentile_summary(d["oldest_history_lag_s"])
            full_history_rate = float(np.mean(d["full_history"])) if d["full_history"] else 0.0

            configured_cap = configured_train_cap if is_train else configured_test_cap
            caps = make_candidates(
                voxel_stats, configured_cap, args.candidate_caps, args.cap_step
            )

            recommended = {
                "p95_rounded": ceil_step(voxel_stats["p95"], args.cap_step),
                "p99_rounded": ceil_step(voxel_stats["p99"], args.cap_step),
                "p99_5_rounded": ceil_step(voxel_stats["p99_5"], args.cap_step),
                "max_rounded": ceil_step(voxel_stats["max"], args.cap_step),
                "interpretation": (
                    "p95 is efficiency-oriented; p99 is the default starting point; "
                    "p99.5/max are conservative and consume more memory/compute."
                ),
            }

            split_report["sweeps"][str(n)] = {
                "historical_sweeps": n,
                "total_lidar_frames_per_prediction": n + 1,
                "full_requested_history_rate": full_history_rate,
                "oldest_history_lag_seconds": lag_stats,
                "points_after_frustum_range": point_stats,
                "candidate_nonempty_voxels": voxel_stats,
                "configured_cap": configured_cap,
                "cap_scenarios": cap_table(d["candidate_voxels"], caps),
                "recommended_capacity_markers": recommended,
            }

            combined[n]["candidate_voxels"].extend(d["candidate_voxels"])
            combined[n]["points_after_filters"].extend(d["points_after_filters"])
            combined[n]["oldest_history_lag_s"].extend(d["oldest_history_lag_s"])
            combined[n]["full_history"].extend(d["full_history"])

        report["splits"][split] = split_report

    for n in sweep_counts:
        d = combined[n]
        voxel_stats = percentile_summary(d["candidate_voxels"])
        caps = make_candidates(
            voxel_stats,
            max(configured_train_cap, configured_test_cap),
            args.candidate_caps,
            args.cap_step,
        )
        report["combined"][str(n)] = {
            "historical_sweeps": n,
            "total_lidar_frames_per_prediction": n + 1,
            "full_requested_history_rate": float(np.mean(d["full_history"]))
            if d["full_history"]
            else 0.0,
            "oldest_history_lag_seconds": percentile_summary(
                d["oldest_history_lag_s"]
            ),
            "points_after_frustum_range": percentile_summary(
                d["points_after_filters"]
            ),
            "candidate_nonempty_voxels": voxel_stats,
            "cap_scenarios": cap_table(d["candidate_voxels"], caps),
            "recommended_capacity_markers": {
                "p95_rounded": ceil_step(voxel_stats["p95"], args.cap_step),
                "p99_rounded": ceil_step(voxel_stats["p99"], args.cap_step),
                "p99_5_rounded": ceil_step(voxel_stats["p99_5"], args.cap_step),
                "max_rounded": ceil_step(voxel_stats["max"], args.cap_step),
            },
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
