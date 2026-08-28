"""Convert CARLA UAV recordings into BEVFusion training information files.

The recorder stores points and boxes in CARLA/UE coordinates (left handed,
x forward, y right, z up).  This converter creates a right-handed, ground
referenced frame for every recording frame:

    x forward, y left, z up, origin below the LiDAR at route-ground height.

Moving the reference origin from the UAV to the ground keeps vehicles around
z=0 and lets the original BEVFusion vertical voxel range remain compact.
"""

import argparse
import json
import math
import os
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw


SUPPORTED_CLASSES = ("car", "van", "truck", "bus")
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_SCENE_ITEMS = (
    "calibration.json",
    "metadata.json",
    "rgb",
    "lidar",
    "pose",
    "labels",
)

S3 = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
S4 = np.eye(4, dtype=np.float64)
S4[:3, :3] = S3


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def homogeneous(rotation=None, translation=None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def matrix_to_quaternion(rotation: np.ndarray) -> List[float]:
    """Return a normalized quaternion in nuScenes/mmcv [w, x, y, z] order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def indexed_files(directory: Path, suffix: str) -> Dict[int, Path]:
    files = {}
    if not directory.is_dir():
        return files
    for path in sorted(directory.glob("*" + suffix)):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        files[index] = path
    return files


def discover_scenes(root_path: Path) -> List[Path]:
    if not root_path.is_dir():
        raise FileNotFoundError("Raw dataset root does not exist: {}".format(root_path))
    scenes = []
    for candidate in sorted(path for path in root_path.iterdir() if path.is_dir()):
        if all((candidate / item).exists() for item in REQUIRED_SCENE_ITEMS):
            scenes.append(candidate)
    if not scenes:
        raise RuntimeError(
            "No scene directories found below {}. A scene must contain: {}".format(
                root_path, ", ".join(REQUIRED_SCENE_ITEMS)
            )
        )
    return scenes


def inspect_scene_files(scene_dir: Path, allow_partial: bool) -> dict:
    maps = {
        "rgb": indexed_files(scene_dir / "rgb", ".png"),
        "lidar": indexed_files(scene_dir / "lidar", ".bin"),
        "pose": indexed_files(scene_dir / "pose", ".json"),
        "labels": indexed_files(scene_dir / "labels", ".json"),
    }
    sets = [set(value) for value in maps.values()]
    common = sorted(set.intersection(*sets)) if sets else []
    union = sorted(set.union(*sets)) if sets else []
    if not common:
        raise RuntimeError("Scene has no complete frames: {}".format(scene_dir))

    mismatches = {
        name: sorted(set(union) - set(values))
        for name, values in maps.items()
        if set(values) != set(union)
    }
    metadata = load_json(scene_dir / "metadata.json")
    expected = int(metadata.get("actual_num_frames", len(common)))
    incomplete = bool(mismatches) or len(common) != expected
    if incomplete and not allow_partial:
        summary = {name: len(values) for name, values in maps.items()}
        raise RuntimeError(
            "Incomplete scene {}: expected {} frames, found {} common frames; "
            "per-directory counts={}. Use --allow-partial-scenes only for "
            "debug samples, not for training data.".format(
                scene_dir.name, expected, len(common), summary
            )
        )
    if len(common) > 1:
        gaps = [right - left for left, right in zip(common[:-1], common[1:])]
        if any(gap != 1 for gap in gaps) and not allow_partial:
            raise RuntimeError("Non-contiguous frame indices in {}".format(scene_dir))
    return {
        "files": maps,
        "indices": common,
        "metadata": metadata,
        "calibration": load_json(scene_dir / "calibration.json"),
        "incomplete": incomplete,
        "mismatches": {name: len(value) for name, value in mismatches.items()},
    }


def load_split_config(split_file: Path, scene_names: Sequence[str]) -> dict:
    """Load and validate an explicit scene-level train/val/test assignment.

    YAML format::

        version: 1
        splits:
          train: [Town01_Opt, Town02_Opt]
          val: [Town05_Opt]
          test: [Town07_Opt]

    Every discovered scene must appear exactly once. Keeping assignment at scene
    granularity prevents map leakage between train/val/test while leaving the
    generated BEVFusion info schema unchanged.
    """
    split_file = Path(split_file).expanduser().resolve()
    if not split_file.is_file():
        raise FileNotFoundError("Split YAML does not exist: {}".format(split_file))

    with split_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Split YAML must contain a mapping at the top level")
    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Split YAML must contain a 'splits' mapping")

    missing_keys = [name for name in SPLIT_NAMES if name not in splits]
    extra_keys = [name for name in splits if name not in SPLIT_NAMES]
    if missing_keys or extra_keys:
        raise ValueError(
            "Split YAML must define exactly train/val/test; missing={}, extra={}".format(
                missing_keys, extra_keys
            )
        )

    normalized = {}
    scene_to_split = {}
    duplicates = {}
    for split_name in SPLIT_NAMES:
        values = splits[split_name]
        if not isinstance(values, list):
            raise ValueError("splits.{} must be a YAML list".format(split_name))
        if not values:
            raise ValueError("splits.{} must contain at least one scene".format(split_name))

        normalized[split_name] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "splits.{} contains an invalid scene name: {!r}".format(
                        split_name, value
                    )
                )
            scene_name = value.strip()
            if Path(scene_name).name != scene_name:
                raise ValueError(
                    "Scene names in split YAML must be directory basenames, got: {}".format(
                        scene_name
                    )
                )
            normalized[split_name].append(scene_name)
            if scene_name in scene_to_split:
                duplicates.setdefault(scene_name, [scene_to_split[scene_name]]).append(
                    split_name
                )
            else:
                scene_to_split[scene_name] = split_name

    if duplicates:
        raise ValueError(
            "Scenes assigned to more than one split: {}".format(duplicates)
        )

    discovered = set(scene_names)
    configured = set(scene_to_split)
    unknown = sorted(configured - discovered)
    unassigned = sorted(discovered - configured)
    if unknown or unassigned:
        raise ValueError(
            "Split YAML does not exactly match discovered scenes; unknown={}, "
            "unassigned={}".format(unknown, unassigned)
        )

    return {
        "path": str(split_file),
        "version": data.get("version", 1),
        "splits": normalized,
        "scene_to_split": scene_to_split,
    }


def reference_height_m(pose: dict, altitude_m: float) -> float:
    uav_z = float(pose["uav"]["location"]["z"])
    lidar_z = float(pose["lidar"]["location"]["z"])
    route_ground_z = uav_z - float(altitude_m)
    return lidar_z - route_ground_z


def world_reference_transform(pose: dict, height_m: float) -> np.ndarray:
    world_lhs_from_lidar_lhs = np.asarray(pose["T_world_lidar"], dtype=np.float64)
    world_rhs_from_lidar_rhs = S4 @ world_lhs_from_lidar_lhs @ S4
    if abs(float(world_rhs_from_lidar_rhs[2, 2])) < 0.999:
        raise RuntimeError(
            "Ground-referenced conversion requires a level LiDAR (roll/pitch near zero)"
        )
    lidar_rhs_from_reference_rhs = homogeneous(translation=[0.0, 0.0, -height_m])
    return world_rhs_from_lidar_rhs @ lidar_rhs_from_reference_rhs


def camera_to_reference_transform(
    pose: dict, calibration: dict, world_rhs_from_reference: np.ndarray
) -> np.ndarray:
    world_lhs_from_camera_ue = np.asarray(pose["T_world_camera"], dtype=np.float64)
    cv_from_camera_ue = np.asarray(
        calibration["T_camera_cv_from_camera_ue"], dtype=np.float64
    )
    world_rhs_from_camera_cv = (
        S4 @ world_lhs_from_camera_ue @ np.linalg.inv(cv_from_camera_ue)
    )
    return np.linalg.inv(world_rhs_from_reference) @ world_rhs_from_camera_cv


def convert_points(
    source_path: Path, output_path: Path, height_m: float, intensity_scale: float
) -> dict:
    flat = np.fromfile(str(source_path), dtype=np.float32)
    if flat.size % 4 != 0:
        raise RuntimeError(
            "Point file is not Nx4 float32: {} ({} floats)".format(source_path, flat.size)
        )
    raw = flat.reshape(-1, 4)
    if raw.shape[0] == 0:
        raise RuntimeError("Point file is empty: {}".format(source_path))
    converted = np.empty((raw.shape[0], 5), dtype=np.float32)
    converted[:, 0] = raw[:, 0]
    converted[:, 1] = -raw[:, 1]
    converted[:, 2] = raw[:, 2] + float(height_m)
    converted[:, 3] = raw[:, 3] * float(intensity_scale)
    converted[:, 4] = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.tofile(str(output_path))
    return {
        "num_points": int(converted.shape[0]),
        "finite": bool(np.isfinite(converted).all()),
        "xyz_min": converted[:, :3].min(axis=0).astype(float).tolist(),
        "xyz_max": converted[:, :3].max(axis=0).astype(float).tolist(),
        "intensity_min": float(converted[:, 3].min()),
        "intensity_max": float(converted[:, 3].max()),
    }


def portable_path(path: Path, dataset_root: Path) -> str:
    try:
        value = os.path.relpath(str(path.resolve()), str(dataset_root.resolve()))
        return Path(value).as_posix()
    except ValueError:
        return str(path.resolve())


def prepare_image_path(
    source_path: Path,
    output_root: Path,
    scene_name: str,
    frame_index: int,
    image_mode: str,
    overwrite: bool,
) -> str:
    if image_mode == "reference":
        return portable_path(source_path, output_root)

    target = output_root / "images" / scene_name / "{:06d}.png".format(frame_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if overwrite:
            target.unlink()
        else:
            return portable_path(target, output_root)
    if image_mode == "copy":
        shutil.copy2(str(source_path), str(target))
    elif image_mode == "hardlink":
        os.link(str(source_path), str(target))
    elif image_mode == "symlink":
        relative_source = os.path.relpath(str(source_path.resolve()), str(target.parent))
        target.symlink_to(relative_source)
    else:
        raise ValueError("Unknown image mode: {}".format(image_mode))
    return portable_path(target, output_root)


def object_track_key(obj: dict) -> Optional[Tuple[str, int]]:
    if obj.get("source") == "dynamic_actor" and obj.get("actor_id") is not None:
        return ("dynamic_actor", int(obj["actor_id"]))
    return None


def collect_tracks(
    indices: Sequence[int], label_files: Dict[int, Path], poses: Dict[int, dict]
) -> Dict[Tuple[str, int], List[Tuple[float, np.ndarray]]]:
    tracks = defaultdict(list)
    for frame_index in indices:
        label = load_json(label_files[frame_index])
        timestamp = float(poses[frame_index]["carla_timestamp"])
        for obj in label.get("objects", []):
            key = object_track_key(obj)
            world_bbox = obj.get("bbox3d", {}).get("world")
            if key is None or not world_bbox:
                continue
            center = np.asarray(world_bbox["center_xyz_m"], dtype=np.float64)
            tracks[key].append((timestamp, center))
    for observations in tracks.values():
        observations.sort(key=lambda item: item[0])
    return dict(tracks)


def estimate_world_velocity(
    observations: Sequence[Tuple[float, np.ndarray]],
    timestamp: float,
    max_gap_s: float,
) -> np.ndarray:
    if len(observations) < 2:
        return np.zeros(3, dtype=np.float64)
    times = [item[0] for item in observations]
    current = min(range(len(times)), key=lambda index: abs(times[index] - timestamp))
    previous = current - 1 if current > 0 else None
    following = current + 1 if current + 1 < len(times) else None

    if previous is not None and following is not None:
        t0, p0 = observations[previous]
        t1, p1 = observations[following]
        if timestamp - t0 <= max_gap_s and t1 - timestamp <= max_gap_s and t1 > t0:
            return (p1 - p0) / (t1 - t0)
    for other in (following, previous):
        if other is None:
            continue
        t1, p1 = observations[other]
        t0, p0 = observations[current]
        if abs(t1 - t0) <= max_gap_s and t1 != t0:
            return (p1 - p0) / (t1 - t0)
    return np.zeros(3, dtype=np.float64)


def convert_annotations(
    label: dict,
    timestamp: float,
    height_m: float,
    world_rhs_from_reference: np.ndarray,
    tracks: Dict[Tuple[str, int], List[Tuple[float, np.ndarray]]],
    max_velocity_gap_s: float,
) -> Tuple[dict, Counter]:
    gt_boxes = []
    gt_names = []
    gt_velocity = []
    num_lidar_pts = []
    class_counts = Counter()
    reference_from_world_rhs = np.linalg.inv(world_rhs_from_reference)

    for obj in label.get("objects", []):
        class_name = str(obj.get("class", ""))
        lidar_bbox = obj.get("bbox3d", {}).get("lidar")
        size_xyz = obj.get("bbox3d", {}).get("size_xyz_m")
        if class_name not in SUPPORTED_CLASSES or not lidar_bbox or size_xyz is None:
            continue

        center_lhs = np.asarray(lidar_bbox["center_xyz_m"], dtype=np.float64)
        center_reference = S3 @ center_lhs + np.asarray([0.0, 0.0, height_m])
        orientation_lhs = np.asarray(
            lidar_bbox["orientation_matrix"], dtype=np.float64
        )
        forward_reference = S3 @ orientation_lhs[:, 0]
        heading = math.atan2(float(forward_reference[1]), float(forward_reference[0]))
        bevfusion_yaw = normalize_angle(-heading - math.pi / 2.0)

        length, width, height = [float(value) for value in size_xyz]
        gt_boxes.append(
            [
                float(center_reference[0]),
                float(center_reference[1]),
                float(center_reference[2]),
                width,
                length,
                height,
                bevfusion_yaw,
            ]
        )
        gt_names.append(class_name)
        points_in_box = int(obj.get("num_lidar_points", 0))
        num_lidar_pts.append(points_in_box)
        class_counts[class_name] += 1

        velocity_lhs = np.zeros(3, dtype=np.float64)
        track_key = object_track_key(obj)
        if track_key is not None and track_key in tracks:
            velocity_lhs = estimate_world_velocity(
                tracks[track_key], timestamp, max_velocity_gap_s
            )
        velocity_world_rhs = S3 @ velocity_lhs
        velocity_reference = reference_from_world_rhs[:3, :3] @ velocity_world_rhs
        gt_velocity.append(velocity_reference[:2].astype(float).tolist())

    boxes_array = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
    names_array = np.asarray(gt_names, dtype=object)
    velocities_array = np.asarray(gt_velocity, dtype=np.float32).reshape(-1, 2)
    lidar_counts = np.asarray(num_lidar_pts, dtype=np.int64)
    annotations = {
        "gt_boxes": boxes_array,
        "gt_names": names_array,
        "gt_velocity": velocities_array,
        "num_lidar_pts": lidar_counts,
        "num_radar_pts": np.zeros(len(boxes_array), dtype=np.int64),
        "valid_flag": lidar_counts > 0,
    }
    return annotations, class_counts


def build_sweep(
    scene_name: str,
    frame_index: int,
    point_path: Path,
    timestamp_us: int,
    world_rhs_from_previous_reference: np.ndarray,
    world_rhs_from_current_reference: np.ndarray,
    output_root: Path,
) -> dict:
    current_reference_from_previous_reference = (
        np.linalg.inv(world_rhs_from_current_reference)
        @ world_rhs_from_previous_reference
    )
    return {
        "data_path": portable_path(point_path, output_root),
        "type": "lidar",
        "sample_data_token": "{}/{:06d}".format(scene_name, frame_index),
        "sensor2ego_translation": [0.0, 0.0, 0.0],
        "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego2global_translation": world_rhs_from_previous_reference[:3, 3].tolist(),
        "ego2global_rotation": matrix_to_quaternion(
            world_rhs_from_previous_reference[:3, :3]
        ),
        "timestamp": int(timestamp_us),
        "sensor2lidar_rotation": current_reference_from_previous_reference[
            :3, :3
        ].astype(np.float32),
        "sensor2lidar_translation": current_reference_from_previous_reference[
            :3, 3
        ].astype(np.float32),
    }


def projection_audit(info: dict, dataset_root: Path) -> dict:
    point_path = resolve_portable_path(info["lidar_path"], dataset_root)
    points = np.fromfile(str(point_path), dtype=np.float32).reshape(-1, 5)
    camera = next(iter(info["cams"].values()))
    camera_to_reference = homogeneous(
        camera["sensor2lidar_rotation"], camera["sensor2lidar_translation"]
    )
    reference_to_camera = np.linalg.inv(camera_to_reference)
    intrinsic = np.asarray(camera["camera_intrinsics"], dtype=np.float64)
    camera_points = (
        reference_to_camera
        @ np.column_stack((points[:, :3], np.ones(len(points), dtype=np.float32))).T
    ).T[:, :3]
    in_front = camera_points[:, 2] > 0.05
    if not np.any(in_front):
        return {
            "total_points": int(len(points)),
            "points_in_front": 0,
            "points_in_image": 0,
            "in_image_ratio": 0.0,
            "camera_depth_min_m": None,
            "camera_depth_max_m": None,
        }
    projected = (intrinsic @ camera_points[in_front].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    width, height = camera["image_size"]
    in_image = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    return {
        "total_points": int(len(points)),
        "points_in_front": int(in_front.sum()),
        "points_in_image": int(in_image.sum()),
        "in_image_ratio": float(in_image.sum() / max(len(points), 1)),
        "camera_depth_min_m": float(camera_points[in_front, 2].min()),
        "camera_depth_max_m": float(camera_points[in_front, 2].max()),
    }


def resolve_portable_path(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (dataset_root / path).resolve()


def bev_box_corners(box: Sequence[float]) -> np.ndarray:
    x, y, _, width, length, _, yaw = [float(value) for value in box[:7]]
    corners = np.asarray(
        [
            [-width / 2.0, -length / 2.0],
            [width / 2.0, -length / 2.0],
            [width / 2.0, length / 2.0],
            [-width / 2.0, length / 2.0],
        ],
        dtype=np.float64,
    )
    rotation_for_row_vectors = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return corners @ rotation_for_row_vectors + np.asarray([x, y])


def save_bev_visualization(info: dict, dataset_root: Path, output_path: Path) -> None:
    canvas_size = 900
    range_m = 55.0
    scale = canvas_size / (2.0 * range_m)
    image = Image.new("RGB", (canvas_size, canvas_size), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    def pixel(x_value, y_value):
        return (
            int(round(canvas_size / 2.0 + y_value * scale)),
            int(round(canvas_size / 2.0 - x_value * scale)),
        )

    points_path = resolve_portable_path(info["lidar_path"], dataset_root)
    points = np.fromfile(str(points_path), dtype=np.float32).reshape(-1, 5)
    mask = (
        (np.abs(points[:, 0]) <= range_m)
        & (np.abs(points[:, 1]) <= range_m)
        & (points[:, 2] >= -5.0)
        & (points[:, 2] <= 3.0)
    )
    visible_points = points[mask]
    stride = max(1, len(visible_points) // 22000)
    for x_value, y_value, z_value in visible_points[::stride, :3]:
        shade = int(np.clip(150.0 - 20.0 * z_value, 35.0, 210.0))
        px, py = pixel(float(x_value), float(y_value))
        draw.point((px, py), fill=(shade, shade, shade))

    colors = {
        "car": (34, 139, 230),
        "van": (160, 75, 210),
        "truck": (230, 120, 25),
        "bus": (210, 45, 55),
    }
    for box, class_name in zip(info["gt_boxes"], info["gt_names"]):
        polygon = [pixel(float(x), float(y)) for x, y in bev_box_corners(box)]
        polygon.append(polygon[0])
        draw.line(polygon, fill=colors.get(str(class_name), (0, 160, 0)), width=3)
    draw.line([pixel(-range_m, 0.0), pixel(range_m, 0.0)], fill=(70, 180, 70), width=1)
    draw.line([pixel(0.0, -range_m), pixel(0.0, range_m)], fill=(70, 180, 70), width=1)
    draw.text((12, 12), "{}  {}".format(info["scene_name"], info["frame_index"]), fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))


def write_pickle(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_dataset(args: argparse.Namespace) -> dict:
    raw_root = Path(args.root_path).expanduser().resolve()
    output_root = Path(args.out_dir).expanduser().resolve()
    scenes = discover_scenes(raw_root)
    split_config = load_split_config(
        Path(args.split_file), [scene_dir.name for scene_dir in scenes]
    )
    scene_to_split = split_config["scene_to_split"]
    output_root.mkdir(parents=True, exist_ok=True)

    info_paths = {
        split_name: output_root / "uavdataset_infos_{}.pkl".format(split_name)
        for split_name in SPLIT_NAMES
    }
    existing_infos = [str(path) for path in info_paths.values() if path.exists()]
    if existing_infos and not args.overwrite:
        raise FileExistsError(
            "Output info files already exist: {}. Pass --overwrite to regenerate.".format(
                existing_infos
            )
        )

    infos_by_split = {split_name: [] for split_name in SPLIT_NAMES}
    split_manifest = {
        "strategy": "explicit_scene_assignment",
        "split_file": split_config["path"],
        "assignments": split_config["splits"],
        "keyframe_stride": int(args.keyframe_stride),
        "max_sweeps": int(args.max_sweeps),
        "boundary_policy": "each scene belongs to exactly one split; sweeps never cross scene boundaries",
        "scenes": {},
    }
    report = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "scene_count": len(scenes),
        "split_file": split_config["path"],
        "split_assignments": split_config["splits"],
        "classes": list(SUPPORTED_CLASSES),
        "coordinate_system": "right-handed ground reference: x forward, y left, z up",
        "intensity_scale": float(args.intensity_scale),
        "scenes": {},
        "splits": {},
        "warnings": [],
    }

    for scene_dir in scenes:
        scene_name = scene_dir.name
        inspected = inspect_scene_files(scene_dir, args.allow_partial_scenes)
        indices = inspected["indices"]
        files = inspected["files"]
        metadata = inspected["metadata"]
        calibration = inspected["calibration"]
        altitude_m = float(metadata["uav_altitude_above_road_m"])
        poses = {index: load_json(files["pose"][index]) for index in indices}
        heights = {
            index: reference_height_m(poses[index], altitude_m) for index in indices
        }
        world_references = {
            index: world_reference_transform(poses[index], heights[index])
            for index in indices
        }

        points_dir = output_root / "points" / scene_name
        point_reports = []
        for frame_index in indices:
            output_point_path = points_dir / "{:06d}.bin".format(frame_index)
            point_reports.append(
                convert_points(
                    files["lidar"][frame_index],
                    output_point_path,
                    heights[frame_index],
                    args.intensity_scale,
                )
            )
        if not all(item["finite"] for item in point_reports):
            raise RuntimeError("Non-finite converted point found in {}".format(scene_name))

        tracks = collect_tracks(indices, files["labels"], poses)
        split_name = scene_to_split[scene_name]
        candidates = list(indices[:: args.keyframe_stride])
        kept_keyframes = []
        dropped_keyframes = []
        scene_class_counts = Counter()

        # Candidate keyframes remain every Nth raw frame, but the entire scene
        # belongs to one split. Sweeps can use earlier raw frames from this same
        # scene only, so they never leak across maps/splits.
        position_by_index = {
            frame_index: pos for pos, frame_index in enumerate(indices)
        }
        for frame_index in candidates:
            position = position_by_index[frame_index]
            if args.max_sweeps > 0:
                previous_indices = indices[
                    max(0, position - args.max_sweeps) : position
                ]
            else:
                previous_indices = []
            if len(previous_indices) < args.max_sweeps:
                dropped_keyframes.append(frame_index)
                continue

            pose = poses[frame_index]
            timestamp_us = int(round(float(pose["carla_timestamp"]) * 1e6))
            world_reference = world_references[frame_index]
            camera_to_reference = camera_to_reference_transform(
                pose, calibration, world_reference
            )
            point_path = points_dir / "{:06d}.bin".format(frame_index)
            image_path = prepare_image_path(
                files["rgb"][frame_index],
                output_root,
                scene_name,
                frame_index,
                args.image_mode,
                args.overwrite,
            )
            label = load_json(files["labels"][frame_index])
            annotations, class_counts = convert_annotations(
                label,
                float(pose["carla_timestamp"]),
                heights[frame_index],
                world_reference,
                tracks,
                args.max_velocity_gap_s,
            )
            scene_class_counts.update(class_counts)

            camera_info = {
                "data_path": image_path,
                "type": "CAM_DOWN",
                "sample_data_token": "{}/{:06d}/CAM_DOWN".format(
                    scene_name, frame_index
                ),
                "sensor2ego_translation": camera_to_reference[:3, 3].tolist(),
                "sensor2ego_rotation": matrix_to_quaternion(
                    camera_to_reference[:3, :3]
                ),
                "ego2global_translation": world_reference[:3, 3].tolist(),
                "ego2global_rotation": matrix_to_quaternion(
                    world_reference[:3, :3]
                ),
                "timestamp": timestamp_us,
                "sensor2lidar_rotation": camera_to_reference[:3, :3].astype(
                    np.float32
                ),
                "sensor2lidar_translation": camera_to_reference[:3, 3].astype(
                    np.float32
                ),
                "camera_intrinsics": np.asarray(
                    calibration["K"], dtype=np.float32
                ),
                "image_size": [
                    int(calibration["camera_resolution"][0]),
                    int(calibration["camera_resolution"][1]),
                ],
            }
            sweeps = [
                build_sweep(
                    scene_name,
                    previous_index,
                    points_dir / "{:06d}.bin".format(previous_index),
                    int(
                        round(
                            float(poses[previous_index]["carla_timestamp"]) * 1e6
                        )
                    ),
                    world_references[previous_index],
                    world_reference,
                    output_root,
                )
                for previous_index in reversed(previous_indices)
            ]
            info = {
                "lidar_path": portable_path(point_path, output_root),
                "token": "{}/{:06d}".format(scene_name, frame_index),
                "scene_name": scene_name,
                "frame_index": int(frame_index),
                "split": split_name,
                "sweeps": sweeps,
                "cams": {"CAM_DOWN": camera_info},
                "lidar2ego_translation": [0.0, 0.0, 0.0],
                "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "ego2global_translation": world_reference[:3, 3].tolist(),
                "ego2global_rotation": matrix_to_quaternion(
                    world_reference[:3, :3]
                ),
                "timestamp": timestamp_us,
                "location": str(metadata.get("route_map", "unknown")),
                "route_name": str(metadata.get("route_name", "unknown")),
                "source_label_path": portable_path(
                    files["labels"][frame_index], output_root
                ),
            }
            info.update(annotations)
            infos_by_split[split_name].append(info)
            kept_keyframes.append(frame_index)

        # Keep the old per-split diagnostic shape for each scene. Only the
        # assigned split contains frames. split_manifest.json is diagnostic and
        # is not consumed by UAVDataset.
        manifest_scene = {
            name: {
                "candidate_keyframes": [],
                "kept_keyframes": [],
                "dropped_for_sweep_boundary": [],
            }
            for name in SPLIT_NAMES
        }
        manifest_scene[split_name] = {
            "candidate_keyframes": candidates,
            "kept_keyframes": kept_keyframes,
            "dropped_for_sweep_boundary": dropped_keyframes,
            "raw_segment_start": int(indices[0]),
            "raw_segment_end_exclusive": int(indices[-1]) + 1,
        }
        manifest_scene["assigned_split"] = split_name
        split_manifest["scenes"][scene_name] = manifest_scene
        first_info = next(
            (
                info
                for split_name in SPLIT_NAMES
                for info in infos_by_split[split_name]
                if info["scene_name"] == scene_name
            ),
            None,
        )
        projection = projection_audit(first_info, output_root) if first_info else None
        report["scenes"][scene_name] = {
            "raw_frames": len(indices),
            "metadata_expected_frames": int(
                metadata.get("actual_num_frames", len(indices))
            ),
            "partial_scene": bool(inspected["incomplete"]),
            "assigned_split": split_name,
            "keyframes_kept": int(len(kept_keyframes)),
            "keyframes_dropped_for_boundaries": int(len(dropped_keyframes)),
            "class_counts": dict(scene_class_counts),
            "point_count_min": int(min(item["num_points"] for item in point_reports)),
            "point_count_max": int(max(item["num_points"] for item in point_reports)),
            "height_shift_min_m": float(min(heights.values())),
            "height_shift_max_m": float(max(heights.values())),
            "projection_audit": projection,
        }

    metadata = {
        "version": "uavdataset-v1.0",
        "classes": list(SUPPORTED_CLASSES),
        "coordinate_system": "right-handed ground reference: x forward, y left, z up",
        "box_storage_origin": "gravity_center",
        "point_format": "float32 x,y,z,intensity,time_lag",
        "keyframe_stride": int(args.keyframe_stride),
        "max_sweeps": int(args.max_sweeps),
        "intensity_scale": float(args.intensity_scale),
    }
    for split_name in SPLIT_NAMES:
        infos_by_split[split_name].sort(
            key=lambda info: (info["scene_name"], info["frame_index"])
        )
        write_pickle(
            {"infos": infos_by_split[split_name], "metadata": metadata},
            info_paths[split_name],
        )
        counts = Counter()
        empty_frames = 0
        for info in infos_by_split[split_name]:
            counts.update(str(value) for value in info["gt_names"])
            if len(info["gt_names"]) == 0:
                empty_frames += 1
        report["splits"][split_name] = {
            "samples": len(infos_by_split[split_name]),
            "empty_samples": empty_frames,
            "class_counts": dict(counts),
            "info_path": str(info_paths[split_name]),
        }

    dump_json(split_manifest, output_root / "split_manifest.json")
    dump_json(report, output_root / "conversion_report.json")

    if args.visualize_samples > 0:
        candidates = [
            info
            for split_name in SPLIT_NAMES
            for info in infos_by_split[split_name]
        ][: args.visualize_samples]
        for info in candidates:
            output_path = (
                output_root
                / "validation"
                / "{}_{}_{:06d}_bev.png".format(
                    info["split"], info["scene_name"], info["frame_index"]
                )
            )
            save_bev_visualization(info, output_root, output_path)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CARLA UAV recordings for BEVFusion"
    )
    parser.add_argument(
        "--root-path", required=True, help="Raw dataset root containing scene directories"
    )
    parser.add_argument(
        "--out-dir", default="data/uavdataset", help="Converted dataset output root"
    )
    parser.add_argument(
        "--split-file",
        required=True,
        help="YAML file assigning every scene directory to train/val/test",
    )
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--max-sweeps", type=int, default=9)
    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=255.0,
        help="Scale CARLA [0,1]-style intensity toward nuScenes pretrained range",
    )
    parser.add_argument(
        "--image-mode",
        choices=("reference", "copy", "hardlink", "symlink"),
        default="reference",
        help="How converted infos refer to RGB files",
    )
    parser.add_argument("--max-velocity-gap-s", type=float, default=0.5)
    parser.add_argument("--visualize-samples", type=int, default=3)
    parser.add_argument("--allow-partial-scenes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.keyframe_stride < 1:
        parser.error("--keyframe-stride must be >= 1")
    if args.max_sweeps < 0:
        parser.error("--max-sweeps must be >= 0")
    if args.intensity_scale <= 0.0:
        parser.error("--intensity-scale must be > 0")
    report = convert_dataset(args)
    print(json.dumps(report["splits"], indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
