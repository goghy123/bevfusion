#!/usr/bin/env python3
"""Audit raw and converted UAV data without MMDetection3D imports.

Exact GT statistics come from converted info files. Point/frustum/voxel
statistics are sampled because reading every ten-sweep cloud is expensive.
"""

# 抽帧分析

# python tools/audit_uav_pipeline.py \
#   --converted-root data/uavdataset/raw \
#   --splits train val \
#   --scene train:Town04_Opt \
#   --scene val:Town05_Opt \
#   --converted-mode stride \
#   --sample-every 20 \
#   --sweeps-num 9 \
#   --point-cloud-range -51.2 -51.2 -12 51.2 51.2 14 \
#   --output uav_audit_s9_town04_vs_town05_stride20.json

# 全量分析
# python tools/audit_uav_pipeline.py \
#   --converted-root data/uavdataset \
#   --splits train val test \
#   --converted-mode all \
#   --sweeps-num 9 \
#   --point-cloud-range -51.2 -51.2 -12 51.2 51.2 14 \
#   --output uav_audit_s9_all.json


import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from data_converter.uavdataset_converter import (
    S3,
    camera_to_reference_transform,
    reference_height_m,
    world_reference_transform,
)


CLASSES = ("car", "van", "truck", "bus")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def choose_evenly(items, limit):
    items = list(items)
    if limit <= 0 or len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[index] for index in indices]



def parse_scene_filters(values):
    """Parse repeated SPLIT:SCENE filters into {split: {scene, ...}}."""
    filters = defaultdict(set)
    for value in values or []:
        if ":" not in value:
            raise ValueError(
                "Invalid --scene {!r}; expected SPLIT:SCENE, e.g. train:Town04_Opt".format(value)
            )
        split, scene = value.split(":", 1)
        split = split.strip()
        scene = scene.strip()
        if split not in ("train", "val", "test") or not scene:
            raise ValueError(
                "Invalid --scene {!r}; expected train|val|test:SCENE".format(value)
            )
        filters[split].add(scene)
    return filters


def filter_infos_by_scene(infos, split, scene_filters):
    """Filter only when this split has explicit scene filters."""
    allowed = scene_filters.get(split)
    if not allowed:
        return list(infos)
    available = {str(info.get("scene_name", "unknown")) for info in infos}
    missing = sorted(allowed - available)
    if missing:
        raise ValueError(
            "Requested scene(s) not found in {}: {}. Available: {}".format(
                split, ", ".join(missing), ", ".join(sorted(available))
            )
        )
    return [
        info for info in infos
        if str(info.get("scene_name", "unknown")) in allowed
    ]


def choose_every_n_frames(items, every, offset=0):
    """Select every Nth frame independently inside each scene."""
    if every <= 0:
        raise ValueError("--sample-every must be >= 1")
    if offset < 0 or offset >= every:
        raise ValueError("--sample-offset must satisfy 0 <= offset < sample_every")
    selected = []
    scene_positions = defaultdict(int)
    for item in items:
        scene = str(item.get("scene_name", "unknown"))
        position = scene_positions[scene]
        scene_positions[scene] += 1
        frame_index = item.get("frame_index")
        try:
            frame_index = int(frame_index)
        except (TypeError, ValueError):
            frame_index = position
        if frame_index >= offset and (frame_index - offset) % every == 0:
            selected.append(item)
    return selected


def select_point_audit_infos(infos, args):
    if args.converted_mode == "all":
        return list(infos)
    selected = choose_every_n_frames(infos, args.sample_every, args.sample_offset)
    if args.max_converted_samples_per_split > 0:
        selected = choose_evenly(selected, args.max_converted_samples_per_split)
    return selected


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else (Path(root) / path).resolve()


def project(points, lidar2image, width, height, margin=2.0):
    homogeneous = np.ones((len(points), 4), dtype=np.float64)
    homogeneous[:, :3] = points[:, :3]
    projected = homogeneous @ np.asarray(lidar2image, dtype=np.float64).T
    depth = projected[:, 2]
    valid = np.isfinite(depth) & (depth > 0.05)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid] = projected[valid, :2] / depth[valid, None]
    visible = (
        valid
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= margin)
        & (uv[:, 0] < width - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] < height - margin)
    )
    return depth, uv, visible


def range_mask(points, point_range):
    low = np.asarray(point_range[:3], dtype=np.float64)
    high = np.asarray(point_range[3:], dtype=np.float64)
    return ((points[:, :3] >= low) & (points[:, :3] < high)).all(axis=1)


def voxel_stats(points, point_range, voxel_size, max_points, max_voxels):
    if not len(points):
        return {"points": 0, "voxels": 0, "point_drop_ratio": 0.0, "voxel_overflow_ratio": 0.0}
    low = np.asarray(point_range[:3], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    coordinates = np.floor((points[:, :3] - low) / size).astype(np.int64)
    _, counts = np.unique(coordinates, axis=0, return_counts=True)
    dropped = np.maximum(counts - max_points, 0).sum()
    overflow = max(len(counts) - max_voxels, 0)
    return {
        "points": int(len(points)),
        "voxels": int(len(counts)),
        "points_per_voxel_p50": float(np.percentile(counts, 50)),
        "points_per_voxel_p95": float(np.percentile(counts, 95)),
        "points_per_voxel_p99": float(np.percentile(counts, 99)),
        "points_per_voxel_max": int(counts.max()),
        "point_drop_ratio": float(dropped / len(points)),
        "voxel_overflow_ratio": float(overflow / max(len(counts), 1)),
    }


def raw_object_point_count(raw_xyz, obj):
    box = obj.get("bbox3d", {}).get("lidar")
    size = obj.get("bbox3d", {}).get("size_xyz_m")
    if not box or size is None:
        return None
    center = np.asarray(box["center_xyz_m"], dtype=np.float64)
    rotation = np.asarray(box["orientation_matrix"], dtype=np.float64)
    local = (raw_xyz - center) @ rotation
    half = np.asarray(size, dtype=np.float64) * 0.5 + 1e-5
    return int((np.abs(local) <= half).all(axis=1).sum())


def points_in_upright_box(points, box):
    center = np.asarray(box[:3], dtype=np.float64)
    width, length, height, yaw = map(float, box[3:7])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    world_to_local = np.asarray([[cosine, sine], [-sine, cosine]])
    local_xy = (points[:, :2] - center[:2]) @ world_to_local
    return (
        (np.abs(local_xy[:, 0]) <= width * 0.5 + 1e-5)
        & (np.abs(local_xy[:, 1]) <= length * 0.5 + 1e-5)
        & (np.abs(points[:, 2] - center[2]) <= height * 0.5 + 1e-5)
    )


def camera_matrices(camera_info):
    rotation = np.asarray(camera_info["sensor2lidar_rotation"], dtype=np.float64)
    translation = np.asarray(camera_info["sensor2lidar_translation"], dtype=np.float64)
    camera2lidar = np.eye(4, dtype=np.float64)
    camera2lidar[:3, :3] = rotation
    camera2lidar[:3, 3] = translation
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(camera_info["camera_intrinsics"], dtype=np.float64)
    return camera2lidar, intrinsic @ np.linalg.inv(camera2lidar)


def load_multisweep(info, root, sweeps_num):
    current = np.fromfile(resolve_path(root, info["lidar_path"]), dtype=np.float32).reshape(-1, 5)
    current = current.copy()
    current[:, 4] = 0.0
    clouds = [current]
    timestamp = float(info["timestamp"]) / 1e6
    for sweep in info.get("sweeps", [])[:sweeps_num]:
        points = np.fromfile(resolve_path(root, sweep["data_path"]), dtype=np.float32).reshape(-1, 5).copy()
        points[:, :3] = points[:, :3] @ np.asarray(sweep["sensor2lidar_rotation"]).T
        points[:, :3] += np.asarray(sweep["sensor2lidar_translation"])
        points[:, 4] = timestamp - float(sweep["timestamp"]) / 1e6
        clouds.append(points)
    return np.concatenate(clouds, axis=0)


def audit_raw(args):
    root = Path(args.raw_root).resolve()
    report = {
        "root": str(root),
        "scenes": {},
        "sampled_frames": 0,
        "sensor_timestamp_mismatches": 0,
        "image_size_mismatches": 0,
        "stored_lidar_count_mismatches": 0,
    }
    calibration_errors = []
    frustum_ratios = []
    range_ratios = []
    depth_ratios = []
    intensity_values = []
    lidar_count_differences = []

    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    for scene in tqdm(scenes, desc="raw scenes", unit="scene", disable=args.no_progress):
        calibration_path = scene / "calibration.json"
        metadata_path = scene / "metadata.json"
        if not calibration_path.is_file() or not metadata_path.is_file():
            continue
        calibration = load_json(calibration_path)
        metadata = load_json(metadata_path)
        indices = sorted(
            int(path.stem)
            for path in (scene / "lidar").glob("*.bin")
            if path.stem.isdigit()
        )
        sampled = choose_evenly(indices, args.max_raw_frames_per_scene)
        scene_counts = Counter()

        for index in tqdm(sampled, desc="raw {}".format(scene.name), unit="frame", leave=False, disable=args.no_progress):
            stem = "{:06d}".format(index)
            pose = load_json(scene / "pose" / (stem + ".json"))
            label = load_json(scene / "labels" / (stem + ".json"))
            raw = np.fromfile(scene / "lidar" / (stem + ".bin"), dtype=np.float32).reshape(-1, 4)
            width, height = map(int, calibration["camera_resolution"])
            with Image.open(scene / "rgb" / (stem + ".png")) as image:
                if image.size != (width, height):
                    report["image_size_mismatches"] += 1
            if abs(float(label["timestamp"]) - float(pose["carla_timestamp"])) > 1e-6:
                report["sensor_timestamp_mismatches"] += 1

            height_shift = reference_height_m(pose, metadata["uav_altitude_above_road_m"])
            world_reference = world_reference_transform(pose, height_shift)
            camera2reference = camera_to_reference_transform(pose, calibration, world_reference)
            lidar2camera = np.linalg.inv(camera2reference)
            intrinsic = np.eye(4, dtype=np.float64)
            intrinsic[:3, :3] = np.asarray(calibration["K"], dtype=np.float64)
            lidar2image = intrinsic @ lidar2camera

            converted = np.empty((len(raw), 5), dtype=np.float64)
            converted[:, :3] = raw[:, :3] @ S3.T
            converted[:, 2] += height_shift
            converted[:, 3] = raw[:, 3] * args.intensity_scale
            converted[:, 4] = 0.0
            depth, _, frustum = project(converted, lidar2image, width, height, args.margin_px)
            inside_range = range_mask(converted, args.point_cloud_range)
            used = frustum & inside_range
            frustum_ratios.append(float(frustum.mean()))
            range_ratios.append(float(inside_range.mean()))
            depth_ratios.append(float(((depth >= args.depth_min) & (depth < args.depth_max) & used).sum() / max(used.sum(), 1)))
            intensity_values.extend(converted[:: max(1, len(converted) // 512), 3].tolist())

            direct = np.asarray(calibration["T_camera_cv_from_lidar"], dtype=np.float64)
            check_indices = np.linspace(0, len(raw) - 1, min(256, len(raw)), dtype=int)
            raw_h = np.ones((len(check_indices), 4), dtype=np.float64)
            raw_h[:, :3] = raw[check_indices, :3]
            converted_h = np.ones((len(check_indices), 4), dtype=np.float64)
            converted_h[:, :3] = converted[check_indices, :3]
            direct_camera = raw_h @ direct.T
            converted_camera = converted_h @ lidar2camera.T
            calibration_errors.append(float(np.max(np.abs(direct_camera - converted_camera))))

            for obj in label.get("objects", []):
                name = str(obj.get("class", ""))
                if name not in CLASSES:
                    continue
                scene_counts[name] += 1
                counted = raw_object_point_count(raw[:, :3], obj)
                stored = int(obj.get("num_lidar_points", 0))
                if counted is not None:
                    lidar_count_differences.append(counted - stored)
                if counted is not None and counted != stored:
                    report["stored_lidar_count_mismatches"] += 1

            report["sampled_frames"] += 1

        report["scenes"][scene.name] = {
            "available_frames": len(indices),
            "metadata_frames": int(metadata.get("actual_num_frames", len(indices))),
            "sampled_frames": len(sampled),
            "sampled_class_counts": dict(scene_counts),
        }

    report["calibration_max_abs_error"] = quantiles(calibration_errors)
    report["frustum_keep_ratio"] = quantiles(frustum_ratios)
    report["raw_point_range_keep_ratio"] = quantiles(range_ratios)
    report["depth_bound_coverage_after_filters"] = quantiles(depth_ratios)
    report["scaled_intensity"] = quantiles(intensity_values)
    report["recomputed_minus_stored_lidar_points"] = quantiles(lidar_count_differences)
    return report


def append_box(records, split, name, box, lidar_points):
    if name not in CLASSES:
        return
    center_z, width, length, height = map(float, (box[2], box[3], box[4], box[5]))
    records[name].append(
        {
            "split": split,
            "bottom_z": center_z - height * 0.5,
            "center_z": center_z,
            "top_z": center_z + height * 0.5,
            "width": width,
            "length": length,
            "height": height,
            "radius": float(np.hypot(box[0], box[1])),
            "x": float(box[0]),
            "y": float(box[1]),
            "lidar_points": float(lidar_points),
        }
    )


def summarize_records(records):
    output = {}
    for name, rows in records.items():
        output[name] = {"count": len(rows)}
        for key in ("bottom_z", "center_z", "top_z", "width", "length", "height", "radius", "lidar_points"):
            output[name][key] = quantiles([row[key] for row in rows])
    return output


def audit_converted(args):
    root = Path(args.converted_root).resolve()
    records = defaultdict(list)
    scene_filters = parse_scene_filters(args.scene)
    report = {
        "root": str(root),
        "selection": {
            "splits": list(args.splits),
            "scene_filters": {k: sorted(v) for k, v in scene_filters.items()},
            "converted_mode": args.converted_mode,
            "sample_every": args.sample_every if args.converted_mode == "stride" else None,
            "sample_offset": args.sample_offset if args.converted_mode == "stride" else None,
            "max_converted_samples_per_split": (
                args.max_converted_samples_per_split
                if args.converted_mode == "stride" and args.max_converted_samples_per_split > 0
                else None
            ),
        },
        "splits": {},
        "point_samples": {},
    }

    for split in args.splits:
        path = root / ("uavdataset_infos_{}.pkl".format(split))
        if not path.is_file():
            print("[WARN] missing {}, skip {}".format(path, split))
            continue
        with path.open("rb") as handle:
            data = pickle.load(handle)
        all_infos = list(data["infos"])
        infos = filter_infos_by_scene(all_infos, split, scene_filters)
        scenes = sorted({str(info.get("scene_name", "unknown")) for info in infos})
        point_infos = select_point_audit_infos(infos, args)

        print(
            "[{}] scope: {} / {} frames, scenes={} | point audit: {} frames ({})".format(
                split, len(infos), len(all_infos), ",".join(scenes),
                len(point_infos), args.converted_mode,
            )
        )

        valid_counts = Counter()
        invalid_counts = Counter()
        scene_counts = defaultdict(Counter)
        empty = 0
        center_outside_range = 0

        # Metadata/GT statistics are cheap: run them on every frame in the
        # selected split/scene scope. The expensive point-cloud pipeline below
        # follows --converted-mode / --sample-every.
        for info in tqdm(
            infos,
            desc="{} GT metadata".format(split),
            unit="frame",
            disable=args.no_progress,
        ):
            boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)
            names = np.asarray(info.get("gt_names", []), dtype=object)
            valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
            lidar_counts = np.asarray(info.get("num_lidar_pts", np.zeros(len(boxes))), dtype=np.int64)
            if not valid.any():
                empty += 1
            for box, raw_name, is_valid, lidar_points in zip(boxes, names, valid, lidar_counts):
                name = str(raw_name)
                (valid_counts if is_valid else invalid_counts)[name] += 1
                if is_valid:
                    scene_counts[str(info.get("scene_name", "unknown"))][name] += 1
                    append_box(records, split, name, box, lidar_points)
                    low = np.asarray(args.point_cloud_range[:3])
                    high = np.asarray(args.point_cloud_range[3:])
                    center_outside_range += int(
                        not ((box[:3] >= low) & (box[:3] < high)).all()
                    )
        report["splits"][split] = {
            "samples_in_pkl": len(all_infos),
            "samples_in_selected_scope": len(infos),
            "selected_scenes": scenes,
            "samples": len(infos),
            "empty_or_no_valid_gt_samples": empty,
            "valid_class_counts": dict(valid_counts),
            "invalid_class_counts": dict(invalid_counts),
            "valid_class_counts_by_scene": {
                scene: dict(counts) for scene, counts in sorted(scene_counts.items())
            },
            "valid_gt_centers_outside_point_cloud_range": center_outside_range,
        }

        metrics = defaultdict(list)
        for info in tqdm(
            point_infos,
            desc="{} point audit".format(split),
            unit="frame",
            disable=args.no_progress,
        ):
            points = load_multisweep(info, root, args.sweeps_num)
            current = np.fromfile(
                resolve_path(root, info["lidar_path"]), dtype=np.float32
            ).reshape(-1, 5)
            camera_info = next(iter(info["cams"].values()))
            _, lidar2image = camera_matrices(camera_info)
            width, height = map(int, camera_info["image_size"])
            depth, _, frustum = project(points, lidar2image, width, height, args.margin_px)
            inside_range = range_mask(points, args.point_cloud_range)
            used = frustum & inside_range
            depth_ok = (depth >= args.depth_min) & (depth < args.depth_max)
            voxels = voxel_stats(
                points[used], args.point_cloud_range, args.voxel_size,
                args.max_points_per_voxel,
                args.max_voxels_train if split == "train" else args.max_voxels_test,
            )
            metrics["points_before"].append(len(points))
            metrics["frustum_keep_ratio"].append(float(frustum.mean()))
            metrics["range_keep_ratio_after_frustum"].append(
                float((frustum & inside_range).sum() / max(frustum.sum(), 1))
            )
            metrics["depth_bound_coverage"].append(
                float((used & depth_ok).sum() / max(used.sum(), 1))
            )
            if used.any():
                used_z = points[used, 2]
                metrics["point_z_min"].append(float(used_z.min()))
                metrics["point_z_p01"].append(float(np.percentile(used_z, 1)))
                metrics["point_z_p50"].append(float(np.percentile(used_z, 50)))
                metrics["point_z_p99"].append(float(np.percentile(used_z, 99)))
                metrics["point_z_max"].append(float(used_z.max()))

            _, _, current_frustum = project(
                current, lidar2image, width, height, args.margin_px
            )
            current_used = current[
                current_frustum & range_mask(current, args.point_cloud_range)
            ]
            boxes = np.asarray(
                info.get("gt_boxes", []), dtype=np.float64
            ).reshape(-1, 7)
            valid = np.asarray(
                info.get("valid_flag", np.ones(len(boxes), dtype=bool)),
                dtype=bool,
            )
            stored_counts = np.asarray(
                info.get("num_lidar_pts", np.zeros(len(boxes))), dtype=np.float64
            )
            retained = np.asarray(
                [points_in_upright_box(current_used, box).sum() for box in boxes[valid]],
                dtype=np.float64,
            )
            if len(retained):
                stored = stored_counts[valid]
                metrics["valid_gt_zero_current_points_after_filters"].append(
                    float((retained == 0).mean())
                )
                metrics["valid_gt_below_15_current_points_after_filters"].append(
                    float((retained < 15).mean())
                )
                metrics["valid_gt_current_point_retention"].append(
                    float(np.median(retained / np.maximum(stored, 1)))
                )
            for key, value in voxels.items():
                metrics["voxel_" + key].append(value)

        report["point_samples"][split] = {
            "sampled": len(point_infos),
            "mode": args.converted_mode,
            "sample_every": args.sample_every if args.converted_mode == "stride" else None,
            "sample_offset": args.sample_offset if args.converted_mode == "stride" else None,
            "selected_scenes": scenes,
            **{key: quantiles(values) for key, values in metrics.items()},
        }

    report["valid_gt_distribution"] = summarize_records(records)
    return report


def self_check():
    items = list(range(10))
    assert choose_evenly(items, 3) == [0, 4, 9]
    demo = [{"scene_name": "A", "frame_index": i} for i in range(7)]
    assert [x["frame_index"] for x in choose_every_n_frames(demo, 3)] == [0, 3, 6]
    points = np.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1]])
    stats = voxel_stats(points, [0, 0, 0, 2, 2, 2], [1, 1, 1], 1, 10)
    assert stats["voxels"] == 2 and np.isclose(stats["point_drop_ratio"], 1 / 3)
    print("self-check passed")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root")
    parser.add_argument("--converted-root")
    parser.add_argument("--output", default="uav_audit_report.json")
    parser.add_argument("--max-raw-frames-per-scene", type=int, default=100)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=["train", "val", "test"],
        help="Converted splits to inspect.",
    )
    parser.add_argument(
        "--scene", action="append", default=[], metavar="SPLIT:SCENE",
        help=(
            "Restrict a split to a scene/subdataset. Repeat as needed, e.g. "
            "--scene train:Town04_Opt --scene val:Town05_Opt. "
            "A selected split without --scene keeps all its scenes."
        ),
    )
    parser.add_argument(
        "--converted-mode", choices=("all", "stride"), default="stride",
        help=(
            "all: run the expensive point audit on every frame in scope; "
            "stride: sample every Nth frame independently per scene."
        ),
    )
    parser.add_argument(
        "--sample-every", type=int, default=10,
        help="With --converted-mode stride, audit every Nth frame per scene.",
    )
    parser.add_argument(
        "--sample-offset", type=int, default=0,
        help="Frame-index offset for stride sampling; must be in [0, N).",
    )
    parser.add_argument(
        "--max-converted-samples-per-split", type=int, default=0,
        help=(
            "Optional final cap after stride sampling (0 = no cap). "
            "Ignored in --converted-mode all."
        ),
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--sweeps-num", type=int, default=9)
    parser.add_argument("--point-cloud-range", type=float, nargs=6, default=[-51.2, -51.2, -12.0, 51.2, 51.2, 14.0])
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[0.1, 0.1, 0.2])
    parser.add_argument("--max-points-per-voxel", type=int, default=10)
    parser.add_argument("--max-voxels-train", type=int, default=90000)
    parser.add_argument("--max-voxels-test", type=int, default=120000)
    parser.add_argument("--depth-min", type=float, default=1.0)
    parser.add_argument("--depth-max", type=float, default=70.0)
    parser.add_argument("--margin-px", type=float, default=2.0)
    parser.add_argument("--intensity-scale", type=float, default=255.0)
    parser.add_argument("--self-check", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.self_check:
        self_check()
        return
    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be >= 1")
    if args.sample_offset < 0 or args.sample_offset >= args.sample_every:
        raise SystemExit("--sample-offset must satisfy 0 <= offset < sample_every")
    if not args.raw_root and not args.converted_root:
        raise SystemExit("Pass --raw-root and/or --converted-root")
    report = {"config": vars(args)}
    if args.raw_root:
        report["raw"] = audit_raw(args)
    if args.converted_root:
        report["converted"] = audit_converted(args)
    output = Path(args.output).resolve()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
