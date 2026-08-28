#!/usr/bin/env python3
"""Audit UAVDataset GT vertical coverage and suggest a Z range.

This script reads converted uavdataset_infos_*.pkl files directly and does not
need MMDetection3D imports.

The converter stores GT z as the geometric/gravity center, so:
    bottom_z = center_z - height / 2
    top_z    = center_z + height / 2
"""

import argparse
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_infos(path):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "infos" not in data:
        raise ValueError("Expected {'infos', 'metadata'} in {}".format(path))
    return data["infos"], data.get("metadata", {})


def snap_down(value, step):
    return math.floor(value / step) * step


def snap_up(value, step):
    return math.ceil(value / step) * step


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def collect_records(infos, split, include_invalid=False):
    records = []
    for info in infos:
        boxes = np.asarray(
            info.get("gt_boxes", []), dtype=np.float64
        ).reshape(-1, 7)
        names = np.asarray(info.get("gt_names", []), dtype=object)
        valid = np.asarray(
            info.get("valid_flag", np.ones(len(boxes), dtype=bool)),
            dtype=bool,
        )
        if len(names) != len(boxes) or len(valid) != len(boxes):
            raise ValueError(
                "GT array length mismatch in {}".format(info.get("token"))
            )

        if not include_invalid:
            boxes = boxes[valid]
            names = names[valid]

        for box, name in zip(boxes, names):
            center_z = float(box[2])
            height = float(box[5])
            records.append(
                {
                    "split": split,
                    "scene": str(info.get("scene_name", "unknown")),
                    "location": str(info.get("location", "unknown")),
                    "class": str(name),
                    "center": center_z,
                    "bottom": center_z - 0.5 * height,
                    "top": center_z + 0.5 * height,
                    "height": height,
                }
            )
    return records


def summarize(records, z_min, z_max):
    if not records:
        return None

    bottom = np.asarray([r["bottom"] for r in records], dtype=np.float64)
    center = np.asarray([r["center"] for r in records], dtype=np.float64)
    top = np.asarray([r["top"] for r in records], dtype=np.float64)
    n = len(records)

    return {
        "n": n,
        "bottom_min": float(bottom.min()),
        "bottom_p001": percentile(bottom, 0.1),
        "bottom_p01": percentile(bottom, 1.0),
        "bottom_p50": percentile(bottom, 50.0),
        "center_min": float(center.min()),
        "center_p50": percentile(center, 50.0),
        "center_max": float(center.max()),
        "top_p50": percentile(top, 50.0),
        "top_p99": percentile(top, 99.0),
        "top_p999": percentile(top, 99.9),
        "top_max": float(top.max()),
        "center_out": int(((center < z_min) | (center > z_max)).sum()),
        "partly_out": int(((bottom < z_min) | (top > z_max)).sum()),
        "fully_out": int(((top < z_min) | (bottom > z_max)).sum()),
    }


def print_summary(title, summary):
    if summary is None:
        print("{}: no GT".format(title))
        return

    n = summary["n"]
    print("\n== {} ==".format(title))
    print("GT count: {}".format(n))
    print(
        "bottom z: min={:.3f}, p0.1={:.3f}, p1={:.3f}, median={:.3f}".format(
            summary["bottom_min"],
            summary["bottom_p001"],
            summary["bottom_p01"],
            summary["bottom_p50"],
        )
    )
    print(
        "center z: min={:.3f}, median={:.3f}, max={:.3f}".format(
            summary["center_min"],
            summary["center_p50"],
            summary["center_max"],
        )
    )
    print(
        "top z: median={:.3f}, p99={:.3f}, p99.9={:.3f}, max={:.3f}".format(
            summary["top_p50"],
            summary["top_p99"],
            summary["top_p999"],
            summary["top_max"],
        )
    )

    for key, label in (
        ("center_out", "center outside configured Z"),
        ("partly_out", "box partly outside configured Z"),
        ("fully_out", "box fully outside configured Z"),
    ):
        value = summary[key]
        print(
            "{}: {} ({:.4f}%)".format(
                label, value, 100.0 * value / max(n, 1)
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/uavdataset")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"]
    )
    parser.add_argument("--z-min", type=float, default=-5.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--voxel-z", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=99.9,
        help="Central GT coverage percentage used for robust suggestion.",
    )
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    if not 50.0 < args.robust_percentile < 100.0:
        raise ValueError("--robust-percentile must be in (50, 100)")

    root = Path(args.dataset_root)
    all_records = []

    for split in args.splits:
        path = root / "uavdataset_infos_{}.pkl".format(split)
        if not path.is_file():
            print("WARNING: missing {}, skipping".format(path))
            continue

        infos, _ = load_infos(path)
        records = collect_records(
            infos, split, include_invalid=args.include_invalid
        )
        all_records.extend(records)
        print_summary(
            "{} (configured Z [{:.3f}, {:.3f}])".format(
                split, args.z_min, args.z_max
            ),
            summarize(records, args.z_min, args.z_max),
        )

    if not all_records:
        raise SystemExit("No GT records found")

    grouped = defaultdict(list)
    for record in all_records:
        grouped[(record["split"], record["class"])].append(record)

    for key in sorted(grouped):
        print_summary(
            "{} / {}".format(*key),
            summarize(grouped[key], args.z_min, args.z_max),
        )

    # Use train+val to design preprocessing; test is audit-only.
    design_records = [
        r for r in all_records if r["split"] in ("train", "val")
    ]
    if not design_records:
        design_records = all_records

    bottoms = np.asarray(
        [r["bottom"] for r in design_records], dtype=np.float64
    )
    tops = np.asarray(
        [r["top"] for r in design_records], dtype=np.float64
    )

    tail = (100.0 - args.robust_percentile) / 2.0
    robust_low = snap_down(
        percentile(bottoms, tail) - args.margin, args.voxel_z
    )
    robust_high = snap_up(
        percentile(tops, 100.0 - tail) + args.margin, args.voxel_z
    )
    all_low = snap_down(
        float(bottoms.min()) - args.margin, args.voxel_z
    )
    all_high = snap_up(
        float(tops.max()) + args.margin, args.voxel_z
    )

    def shape_for(low, high):
        bins = int(round((high - low) / args.voxel_z))
        if not np.isclose(
            low + bins * args.voxel_z, high, atol=1e-6
        ):
            raise ValueError(
                "Suggested range is not divisible by voxel-z"
            )
        # This BEVFusion fork uses 41 for [-5, 3] at voxel-z 0.2:
        # 8 / 0.2 = 40 bins, sparse/grid z shape = 41.
        return bins + 1

    print("\n== Suggested Z ranges (derived from train+val) ==")
    print(
        "Robust {:.3f}% coverage + {:.2f} m margin: "
        "[{:.3f}, {:.3f}]".format(
            args.robust_percentile,
            args.margin,
            robust_low,
            robust_high,
        )
    )
    print(
        "  Sparse/grid Z shape: {}".format(
            shape_for(robust_low, robust_high)
        )
    )
    print(
        "  DepthLSSTransform zbound, ONE vertical bin: "
        "[{:.3f}, {:.3f}, {:.3f}]".format(
            robust_low,
            robust_high,
            robust_high - robust_low,
        )
    )

    print(
        "Cover ALL train+val GT + {:.2f} m margin: "
        "[{:.3f}, {:.3f}]".format(
            args.margin, all_low, all_high
        )
    )
    print(
        "  Sparse/grid Z shape: {}".format(
            shape_for(all_low, all_high)
        )
    )
    print(
        "  DepthLSSTransform zbound, ONE vertical bin: "
        "[{:.3f}, {:.3f}, {:.3f}]".format(
            all_low, all_high, all_high - all_low
        )
    )

    print("\nDecision guide:")
    print(
        "  fully_out <= 0.1%: current Z is usually acceptable; "
        "fix filtering/eval consistency first."
    )
    print(
        "  fully_out 0.1%-1%: inspect those scenes; robust 99.9% "
        "range is a good candidate."
    )
    print(
        "  fully_out > 1%: widen Z; use robust range, or all-GT "
        "range if multi-level roads are in-scope."
    )
    print(
        "  Keep point_cloud_range, voxelizer, sparse/grid Z shape, "
        "camera zbound, and evaluation policy consistent."
    )


if __name__ == "__main__":
    main()
