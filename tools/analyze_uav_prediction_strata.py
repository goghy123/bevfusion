#!/usr/bin/env python3
"""Stratified diagnostic analysis for UAVDataset predictions.

This script mirrors the native evaluator's IoU/matching logic, then breaks GT
recall down by:
  - GT center-z band
  - XY distance
  - single-level vs multi-level frames
  - frame-index windows

It also separates misses into:
  1) matched at IoU threshold only when low-score proposals are allowed
  2) no same-class proposal reaches the IoU threshold at all

Run inside the BEVFusion environment.
"""

# python tools/analyze_uav_prediction_strata.py \
#   --ann data/uavdataset/uavdataset_infos_val_s9.pkl \
#   --pred runs/uavdataset-bevfusion-s9/val_results.pkl \
#   --iou 0.5 \
#   --score 0.1 \
#   --lower-z -5 \
#   --upper-z 5 \
#   --multi-level-span 5 \
#   --frame-bin 500 \
#   --output-json runs/uavdataset-bevfusion-s9/val_strata_analysis.json \
#   2>&1 | tee runs/uavdataset-bevfusion-s9/val_strata_analysis.txt


import argparse
import json
import math
import pickle
from collections import defaultdict

import numpy as np


CLASSES = ("car", "van", "truck", "bus")


def load_any(path):
    try:
        import mmcv
        return mmcv.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def box_bev_corners(box):
    x, y, _, width, length, _, yaw = [float(v) for v in box[:7]]
    corners = np.asarray(
        [
            [-width / 2.0, -length / 2.0],
            [ width / 2.0, -length / 2.0],
            [ width / 2.0,  length / 2.0],
            [-width / 2.0,  length / 2.0],
        ],
        dtype=np.float64,
    )
    rot = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)],
         [math.sin(yaw),  math.cos(yaw)]],
        dtype=np.float64,
    )
    return corners @ rot + np.asarray([x, y], dtype=np.float64)


def cross_2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def line_intersection(p0, p1, q0, q1):
    dp = p1 - p0
    dq = q1 - q0
    denom = cross_2d(dp, dq)
    if abs(denom) < 1e-12:
        return (p0 + p1) * 0.5
    factor = cross_2d(q0 - p0, dq) / denom
    return p0 + factor * dp


def convex_polygon_clip(subject, clipper):
    output = [p for p in subject]
    for edge_idx in range(len(clipper)):
        clip_start = clipper[edge_idx]
        clip_end = clipper[(edge_idx + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break

        def inside(point):
            return cross_2d(clip_end - clip_start, point - clip_start) >= -1e-9

        previous = input_points[-1]
        previous_inside = inside(previous)
        for current in input_points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(
                        line_intersection(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    line_intersection(previous, current, clip_start, clip_end)
                )
            previous = current
            previous_inside = current_inside

    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(
        0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def pair_iou(first_box, second_box, mode):
    first_poly = box_bev_corners(first_box)
    second_poly = box_bev_corners(second_box)
    inter_poly = convex_polygon_clip(first_poly, second_poly)
    inter_bev = polygon_area(inter_poly)
    if inter_bev <= 0.0:
        return 0.0

    first_area = float(first_box[3] * first_box[4])
    second_area = float(second_box[3] * second_box[4])

    if mode == "bev":
        union = first_area + second_area - inter_bev
        return inter_bev / max(union, 1e-12)

    first_bottom = float(first_box[2])
    second_bottom = float(second_box[2])
    first_top = first_bottom + float(first_box[5])
    second_top = second_bottom + float(second_box[5])

    inter_h = max(
        0.0, min(first_top, second_top) - max(first_bottom, second_bottom)
    )
    inter_vol = inter_bev * inter_h
    first_vol = first_area * float(first_box[5])
    second_vol = second_area * float(second_box[5])
    union_vol = first_vol + second_vol - inter_vol
    return inter_vol / max(union_vol, 1e-12)


def extract_predictions(results, class_index):
    predictions = []
    per_sample = defaultdict(list)

    for sample_idx, result in enumerate(results):
        boxes = as_numpy(result["boxes_3d"].tensor).astype(np.float64)
        scores = as_numpy(result["scores_3d"]).astype(np.float64)
        labels = as_numpy(result["labels_3d"]).astype(np.int64)

        mask = labels == class_index
        for box, score in zip(boxes[mask], scores[mask]):
            item = (float(score), sample_idx, box[:7].copy())
            predictions.append(item)
            per_sample[sample_idx].append((float(score), box[:7].copy()))

    return predictions, per_sample


def build_gt(infos, class_name):
    gt_by_sample = {}
    meta_by_sample = {}

    for sample_idx, info in enumerate(infos):
        valid = np.asarray(info.get("valid_flag", []), dtype=bool)
        names = np.asarray(info.get("gt_names", []), dtype=object)
        raw_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64).reshape(-1, 7)

        if len(valid) == 0 and len(raw_boxes):
            valid = np.ones(len(raw_boxes), dtype=bool)

        all_valid_boxes = raw_boxes[valid]
        if len(all_valid_boxes):
            frame_z_min = float(np.min(all_valid_boxes[:, 2]))
            frame_z_max = float(np.max(all_valid_boxes[:, 2]))
            frame_z_span = frame_z_max - frame_z_min
        else:
            frame_z_min = frame_z_max = frame_z_span = 0.0

        mask = valid & (names == class_name)
        centers = raw_boxes[mask].copy()

        eval_boxes = centers.copy()
        if len(eval_boxes):
            # Mirror UAVDataset._ground_truth_for_evaluation():
            # converter stores geometric center; evaluator converts z to bottom center.
            eval_boxes[:, 2] -= eval_boxes[:, 5] * 0.5

        gt_by_sample[sample_idx] = eval_boxes.reshape(-1, 7)

        frame_index = info.get("frame_index", sample_idx)
        try:
            frame_index = int(frame_index)
        except Exception:
            frame_index = sample_idx

        meta = []
        for local_idx, center_box in enumerate(centers):
            x, y, z = [float(v) for v in center_box[:3]]
            meta.append(
                dict(
                    sample_index=sample_idx,
                    local_gt_index=local_idx,
                    frame_index=frame_index,
                    scene=str(info.get("scene_name", info.get("location", "unknown"))),
                    x=x,
                    y=y,
                    z=z,
                    distance=float(math.hypot(x, y)),
                    frame_z_min=frame_z_min,
                    frame_z_max=frame_z_max,
                    frame_z_span=frame_z_span,
                )
            )
        meta_by_sample[sample_idx] = meta

    return gt_by_sample, meta_by_sample


def greedy_match(predictions, gt_by_sample, iou_threshold, mode, score_threshold=None):
    ordered = sorted(predictions, key=lambda x: x[0], reverse=True)
    if score_threshold is not None:
        ordered = [x for x in ordered if x[0] >= score_threshold]

    matched = {
        sample_idx: np.zeros(len(boxes), dtype=bool)
        for sample_idx, boxes in gt_by_sample.items()
    }
    matched_score = {
        sample_idx: np.full(len(boxes), np.nan, dtype=np.float64)
        for sample_idx, boxes in gt_by_sample.items()
    }

    tp = 0
    fp = 0

    for score, sample_idx, pred_box in ordered:
        gt_boxes = gt_by_sample.get(sample_idx)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp += 1
            continue

        ious = np.asarray(
            [pair_iou(pred_box, gt_box, mode) for gt_box in gt_boxes],
            dtype=np.float64,
        )
        ious[matched[sample_idx]] = -1.0

        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= iou_threshold:
            matched[sample_idx][best_idx] = True
            matched_score[sample_idx][best_idx] = float(score)
            tp += 1
        else:
            fp += 1

    return matched, matched_score, tp, fp


def best_proposal_stats(per_sample_predictions, gt_by_sample, mode, iou_threshold):
    out = {}

    for sample_idx, gt_boxes in gt_by_sample.items():
        preds = per_sample_predictions.get(sample_idx, [])
        stats = []

        for gt_box in gt_boxes:
            best_iou = 0.0
            score_at_best_iou = 0.0
            max_score_above_iou = 0.0

            for score, pred_box in preds:
                iou = pair_iou(pred_box, gt_box, mode)
                if iou > best_iou:
                    best_iou = iou
                    score_at_best_iou = float(score)
                if iou >= iou_threshold:
                    max_score_above_iou = max(max_score_above_iou, float(score))

            stats.append(
                dict(
                    best_iou=float(best_iou),
                    score_at_best_iou=float(score_at_best_iou),
                    max_score_above_iou=float(max_score_above_iou),
                )
            )
        out[sample_idx] = stats

    return out


def z_band(z, lower_z, upper_z):
    if z < lower_z:
        return "lower"
    if z > upper_z:
        return "upper"
    return "normal"


def distance_band(distance):
    if distance < 20.0:
        return "0-20m"
    if distance < 35.0:
        return "20-35m"
    if distance < 51.2:
        return "35-51.2m"
    return ">=51.2m"


def summarize_rows(rows):
    n = len(rows)
    if n == 0:
        return dict(gt=0)

    matched_bev = sum(r["bev_matched_score"] for r in rows)
    matched_3d = sum(r["3d_matched_score"] for r in rows)
    any_bev = sum(r["bev_matched_any"] for r in rows)
    any_3d = sum(r["3d_matched_any"] for r in rows)

    return dict(
        gt=n,
        bev_recall_score=matched_bev / n,
        d3_recall_score=matched_3d / n,
        bev_recall_any=any_bev / n,
        d3_recall_any=any_3d / n,
        bev_low_confidence_only=(any_bev - matched_bev) / n,
        bev_no_iou_match=(n - any_bev) / n,
        d3_low_confidence_only=(any_3d - matched_3d) / n,
        d3_no_iou_match=(n - any_3d) / n,
    )


def group_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize_rows(items) for name, items in sorted(groups.items())}


def print_table(title, summary, order=None):
    print("\n" + title)
    print(
        f"{'group':>18} {'GT':>6} "
        f"{'BEV@score':>10} {'BEV@any':>9} {'low-score':>10} {'no-IoU':>9} "
        f"{'3D@score':>10} {'3D@any':>9}"
    )

    keys = list(summary.keys())
    if order:
        keys = [k for k in order if k in summary] + [k for k in keys if k not in order]

    for name in keys:
        s = summary[name]
        if s.get("gt", 0) == 0:
            continue
        print(
            f"{name:>18} {s['gt']:6d} "
            f"{s['bev_recall_score']:10.4f} {s['bev_recall_any']:9.4f} "
            f"{s['bev_low_confidence_only']:10.4f} {s['bev_no_iou_match']:9.4f} "
            f"{s['d3_recall_score']:10.4f} {s['d3_recall_any']:9.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="val/test info PKL")
    parser.add_argument("--pred", required=True, help="test.py --out result PKL")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.1)
    parser.add_argument("--lower-z", type=float, default=-5.0)
    parser.add_argument("--upper-z", type=float, default=5.0)
    parser.add_argument("--multi-level-span", type=float, default=5.0)
    parser.add_argument("--frame-bin", type=int, default=500)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    ann = load_any(args.ann)
    results = load_any(args.pred)

    if not isinstance(ann, dict) or "infos" not in ann:
        raise ValueError("ann file must contain {'infos': ...}")

    infos = list(ann["infos"])
    if len(infos) != len(results):
        raise ValueError(
            f"sample count mismatch: ann={len(infos)}, pred={len(results)}"
        )

    report = {
        "config": vars(args),
        "num_samples": len(infos),
        "classes": {},
    }

    all_rows = []

    for class_idx, class_name in enumerate(CLASSES):
        gt_by_sample, meta_by_sample = build_gt(infos, class_name)
        predictions, per_sample_predictions = extract_predictions(results, class_idx)

        matched_bev_any, _, tp_bev_any, fp_bev_any = greedy_match(
            predictions, gt_by_sample, args.iou, "bev", None
        )
        matched_bev_score, bev_score_map, tp_bev_score, fp_bev_score = greedy_match(
            predictions, gt_by_sample, args.iou, "bev", args.score
        )

        matched_3d_any, _, tp_3d_any, fp_3d_any = greedy_match(
            predictions, gt_by_sample, args.iou, "3d", None
        )
        matched_3d_score, d3_score_map, tp_3d_score, fp_3d_score = greedy_match(
            predictions, gt_by_sample, args.iou, "3d", args.score
        )

        best_bev = best_proposal_stats(
            per_sample_predictions, gt_by_sample, "bev", args.iou
        )
        best_3d = best_proposal_stats(
            per_sample_predictions, gt_by_sample, "3d", args.iou
        )

        rows = []
        for sample_idx in range(len(infos)):
            metas = meta_by_sample[sample_idx]
            for local_idx, meta in enumerate(metas):
                row = dict(meta)
                row["class"] = class_name
                row["z_band"] = z_band(row["z"], args.lower_z, args.upper_z)
                row["distance_band"] = distance_band(row["distance"])
                row["multi_level"] = (
                    "multi_level"
                    if row["frame_z_span"] >= args.multi_level_span
                    else "single_level"
                )
                frame_start = (row["frame_index"] // args.frame_bin) * args.frame_bin
                row["frame_bin"] = f"{frame_start}-{frame_start + args.frame_bin - 1}"

                row["bev_matched_any"] = bool(matched_bev_any[sample_idx][local_idx])
                row["bev_matched_score"] = bool(matched_bev_score[sample_idx][local_idx])
                row["bev_match_score"] = (
                    None
                    if np.isnan(bev_score_map[sample_idx][local_idx])
                    else float(bev_score_map[sample_idx][local_idx])
                )

                row["3d_matched_any"] = bool(matched_3d_any[sample_idx][local_idx])
                row["3d_matched_score"] = bool(matched_3d_score[sample_idx][local_idx])
                row["3d_match_score"] = (
                    None
                    if np.isnan(d3_score_map[sample_idx][local_idx])
                    else float(d3_score_map[sample_idx][local_idx])
                )

                row["bev_best_iou"] = best_bev[sample_idx][local_idx]["best_iou"]
                row["bev_best_iou_score"] = best_bev[sample_idx][local_idx]["score_at_best_iou"]
                row["bev_max_score_iou_ok"] = best_bev[sample_idx][local_idx]["max_score_above_iou"]

                row["3d_best_iou"] = best_3d[sample_idx][local_idx]["best_iou"]
                row["3d_best_iou_score"] = best_3d[sample_idx][local_idx]["score_at_best_iou"]
                row["3d_max_score_iou_ok"] = best_3d[sample_idx][local_idx]["max_score_above_iou"]

                rows.append(row)

        num_gt = len(rows)
        pred_score_count = sum(
            1 for score, _, _ in predictions if score >= args.score
        )

        overall = summarize_rows(rows)
        overall.update(
            dict(
                predictions_total=len(predictions),
                predictions_at_score=pred_score_count,
                bev_tp_at_score=tp_bev_score,
                bev_fp_at_score=fp_bev_score,
                bev_precision_at_score=tp_bev_score / max(tp_bev_score + fp_bev_score, 1),
                d3_tp_at_score=tp_3d_score,
                d3_fp_at_score=fp_3d_score,
                d3_precision_at_score=tp_3d_score / max(tp_3d_score + fp_3d_score, 1),
            )
        )

        by_z = group_summary(rows, "z_band")
        by_distance = group_summary(rows, "distance_band")
        by_level = group_summary(rows, "multi_level")
        by_frame = group_summary(rows, "frame_bin")

        misses = [
            r for r in rows
            if not r["bev_matched_score"]
        ]
        misses = sorted(
            misses,
            key=lambda r: (
                r["bev_matched_any"],        # no-IoU first
                -abs(r["z"]),
                -r["distance"],
            )
        )[:100]

        report["classes"][class_name] = {
            "overall": overall,
            "by_z": by_z,
            "by_distance": by_distance,
            "by_level": by_level,
            "by_frame": by_frame,
            "miss_examples_first100": misses,
        }

        all_rows.extend(rows)

        print("\n" + "=" * 88)
        print(f"CLASS: {class_name}")
        print(
            f"GT={num_gt}  pred@{args.score:.2f}={pred_score_count}  "
            f"BEV TP/FP={tp_bev_score}/{fp_bev_score}  "
            f"BEV P/R={overall['bev_precision_at_score']:.4f}/"
            f"{overall['bev_recall_score']:.4f}  "
            f"3D P/R={overall['d3_precision_at_score']:.4f}/"
            f"{overall['d3_recall_score']:.4f}"
        )

        print_table(
            "By GT center Z",
            by_z,
            order=["lower", "normal", "upper"],
        )
        print_table(
            "By XY distance",
            by_distance,
            order=["0-20m", "20-35m", "35-51.2m", ">=51.2m"],
        )
        print_table(
            "By vertical scene structure",
            by_level,
            order=["single_level", "multi_level"],
        )

    report["all_classes"] = {
        "overall": summarize_rows(all_rows),
        "by_z": group_summary(all_rows, "z_band"),
        "by_distance": group_summary(all_rows, "distance_band"),
        "by_level": group_summary(all_rows, "multi_level"),
        "by_frame": group_summary(all_rows, "frame_bin"),
    }

    print("\n" + "=" * 88)
    print("ALL CLASSES")
    print_table(
        "By GT center Z",
        report["all_classes"]["by_z"],
        order=["lower", "normal", "upper"],
    )
    print_table(
        "By XY distance",
        report["all_classes"]["by_distance"],
        order=["0-20m", "20-35m", "35-51.2m", ">=51.2m"],
    )
    print_table(
        "By vertical scene structure",
        report["all_classes"]["by_level"],
        order=["single_level", "multi_level"],
    )

    print_table(
        "By frame-index window",
        report["all_classes"]["by_frame"],
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote: {args.output_json}")


if __name__ == "__main__":
    main()
