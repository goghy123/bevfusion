"""Offline UAV Z accuracy and stratified detection diagnostics."""

import math
import re
from collections import defaultdict

import numpy as np

from mmdet3d.datasets.uav_dataset import (
    pair_iou,
    z_error_summary,
    z_threshold_ap,
)


GROUPS = {
    "by_z": "z_band",
    "by_distance": "distance_band",
    "by_level": "multi_level",
    "by_frame": "frame_bin",
    "by_town": "town",
    "by_scene": "scene",
    "by_weather": "weather",
    "by_motion": "motion_state",
    "by_disturbance": "disturbance",
    "by_scene_frame": "scene_frame",
}


def build_gt(dataset, class_index, args, native=False):
    gt, metadata = {}, {}
    class_name = dataset.CLASSES[class_index]

    for index, info in enumerate(dataset.data_infos):
        raw = np.asarray(
            info["gt_boxes"], dtype=np.float64
        ).reshape(-1, 7)
        names = np.asarray(info["gt_names"], dtype=object)

        if native:
            valid = dataset._base_gt_mask(info)
        else:
            valid = np.asarray(
                info.get("valid_flag", []), dtype=bool
            )
            if not len(valid):
                valid = np.ones(len(raw), dtype=bool)

        if len(valid) != len(raw) or len(names) != len(raw):
            raise ValueError(
                "Malformed GT in sample {}".format(index)
            )

        valid_boxes = raw[valid]
        z_min = (
            float(valid_boxes[:, 2].min())
            if len(valid_boxes) else 0.0
        )
        z_max = (
            float(valid_boxes[:, 2].max())
            if len(valid_boxes) else 0.0
        )

        centers = raw[valid & (names == class_name)].copy()
        boxes = centers.copy()
        boxes[:, 2] -= boxes[:, 5] * 0.5
        gt[index] = boxes

        try:
            frame = int(info.get("frame_index", index))
        except (TypeError, ValueError):
            frame = index

        scene = str(
            info.get("scene_name", info.get("location", "unknown"))
        )
        location = str(
            info.get("town", info.get("location", "unknown"))
        )
        town_match = re.search(r"Town\d+[A-Za-z0-9_]*", location)
        town = town_match.group(0) if town_match else location

        start = frame // args.frame_bin * args.frame_bin
        window = "{}-{}".format(
            start, start + args.frame_bin - 1
        )

        rows = []
        for local_index, box in enumerate(centers):
            x, y, z = [float(v) for v in box[:3]]
            distance = float(math.hypot(x, y))

            band = (
                "0-20m" if distance < 20
                else "20-35m" if distance < 35
                else "35-51.2m" if distance < 51.2
                else ">=51.2m"
            )

            rows.append(
                dict(
                    sample_index=index,
                    local_gt_index=local_index,
                    frame_index=frame,
                    scene=scene,
                    town=town,
                    weather=info.get("weather", "unknown"),
                    motion_state=info.get("motion_state", "unknown"),
                    disturbance=("active" if info.get("attitude_disturbance", {}).get("event_index") is not None else "inactive"),
                    **{"class": str(class_name)},
                    x=x,
                    y=y,
                    z=z,
                    distance=distance,
                    frame_z_min=z_min,
                    frame_z_max=z_max,
                    frame_z_span=z_max - z_min,
                    z_band=(
                        "lower" if z < args.lower_z
                        else "upper" if z > args.upper_z
                        else "normal"
                    ),
                    distance_band=band,
                    multi_level=(
                        "multi_level"
                        if z_max - z_min >= args.multi_level_span
                        else "single_level"
                    ),
                    frame_bin=window,
                    scene_frame=scene + "/" + window,
                )
            )

        metadata[index] = rows

    if native:
        # 保留原生评估器的 GT 和筛选口径。
        gt = dataset._ground_truth_for_evaluation(class_index)

    return gt, metadata


def match(
    predictions,
    gt,
    threshold,
    mode,
    score=None,
    overlaps=None,
):
    overlaps = {} if overlaps is None else overlaps

    hits = {
        i: np.zeros(len(boxes), dtype=bool)
        for i, boxes in gt.items()
    }
    scores = {
        i: np.full(len(boxes), np.nan)
        for i, boxes in gt.items()
    }
    errors = {
        i: np.full(len(boxes), np.nan)
        for i, boxes in gt.items()
    }

    tp = fp = 0

    for number, (confidence, sample, box) in enumerate(predictions):
        if score is not None and confidence < score:
            continue

        boxes = gt.get(sample)
        if boxes is None or not len(boxes):
            fp += 1
            continue

        key = (number, mode)
        if key not in overlaps:
            overlaps[key] = np.asarray(
                [pair_iou(box, g, mode) for g in boxes]
            )

        ious = overlaps[key].copy()
        ious[hits[sample]] = -1
        best = int(np.argmax(ious))

        if ious[best] >= threshold:
            hits[sample][best] = True
            scores[sample][best] = confidence
            errors[sample][best] = float(
                box[2] + box[5] * 0.5
                - boxes[best, 2] - boxes[best, 5] * 0.5
            )
            tp += 1
        else:
            fp += 1

    return dict(
        hit=hits,
        score=scores,
        error=errors,
        tp=tp,
        fp=fp,
    )


def summarize(rows):
    n = len(rows)
    result = dict(gt=n)

    for prefix, name in (("bev", "bev"), ("3d", "d3")):
        at_score = sum(
            row[prefix + "_matched_score"] for row in rows
        )
        at_any = sum(
            row[prefix + "_matched_any"] for row in rows
        )

        result[name + "_recall_score"] = at_score / max(n, 1)
        result[name + "_recall_any"] = at_any / max(n, 1)
        result[name + "_low_confidence_only"] = (
            at_any - at_score
        ) / max(n, 1)
        result[name + "_unmatched_any"] = (
            n - at_any
        ) / max(n, 1)

    return result


def summarize_z(rows):
    summary = z_error_summary(
        [
            row["z_error_m"]
            for row in rows
            if row["z_error_m"] is not None
        ]
    )
    return dict(
        summary,
        gt=len(rows),
        bev_recall_score=summary["matches"] / max(len(rows), 1),
    )


def grouped(rows, summary_function):
    result = {"overall": summary_function(rows)}

    for output_key, row_key in GROUPS.items():
        buckets = defaultdict(list)

        for row in rows:
            buckets[str(row[row_key])].append(row)

        result[output_key] = {
            name: summary_function(items)
            for name, items in sorted(buckets.items())
        }

    return result


def evaluate(dataset, results, args):
    strata, z_report = {}, {}
    all_rows, all_z_rows = [], []
    ap_values = {str(t): [] for t in args.z_thresholds}

    for class_index, class_name in enumerate(dataset.CLASSES):
        predictions = sorted(
            dataset._predictions_for_evaluation(
                results, class_index
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        # 分层召回保留旧分层脚本的 GT 筛选方式。
        gt, metadata = build_gt(dataset, class_index, args)

        overlaps, matched = {}, {}
        for mode in ("bev", "3d"):
            for label, threshold in (
                ("any", None),
                ("score", args.score_threshold),
            ):
                matched[mode, label] = match(
                    predictions,
                    gt,
                    args.iou_threshold,
                    mode,
                    threshold,
                    overlaps,
                )

        per_sample = defaultdict(list)
        for number, (score, sample, _) in enumerate(predictions):
            per_sample[sample].append((number, score))

        rows = []
        for sample, metas in metadata.items():
            for local_index, meta in enumerate(metas):
                row = dict(meta)

                for mode in ("bev", "3d"):
                    for label in ("any", "score"):
                        row[mode + "_matched_" + label] = bool(
                            matched[mode, label]["hit"][sample][local_index]
                        )

                    confidence = (
                        matched[mode, "score"]["score"][sample][local_index]
                    )
                    row[mode + "_match_score"] = (
                        None if np.isnan(confidence)
                        else float(confidence)
                    )

                    proposals = [
                        (
                            float(overlaps[number, mode][local_index]),
                            score,
                        )
                        for number, score in per_sample[sample]
                    ]

                    best_iou, best_score = max(
                        proposals,
                        key=lambda p: p[0],
                        default=(0.0, 0.0),
                    )

                    row[mode + "_best_iou"] = best_iou
                    row[mode + "_best_iou_score"] = (
                        best_score if best_iou > 0 else 0.0
                    )
                    row[mode + "_max_score_iou_ok"] = max(
                        (
                            score
                            for iou, score in proposals
                            if iou >= args.iou_threshold
                        ),
                        default=0.0,
                    )

                rows.append(row)

        class_report = grouped(rows, summarize)
        class_report["overall"].update(
            predictions_total=len(predictions),
            predictions_at_score=sum(
                p[0] >= args.score_threshold for p in predictions
            ),
        )

        for mode, prefix in (("bev", "bev"), ("3d", "d3")):
            matched_at_score = matched[mode, "score"]
            tp = matched_at_score["tp"]
            fp = matched_at_score["fp"]

            class_report["overall"].update(
                {
                    prefix + "_tp_at_score": tp,
                    prefix + "_fp_at_score": fp,
                    prefix + "_precision_at_score": (
                        tp / max(tp + fp, 1)
                    ),
                }
            )

        class_report["miss_examples_first100"] = sorted(
            [
                row
                for row in rows
                if not row["bev_matched_score"]
            ],
            key=lambda row: (
                row["bev_matched_any"],
                -abs(row["z"]),
                -row["distance"],
            ),
        )[:100]

        strata[str(class_name)] = class_report
        all_rows.extend(rows)

        # Z 统计保留原 evaluate_uav_z 的原生 GT 筛选方式。
        native_gt, native_meta = build_gt(
            dataset, class_index, args, native=True
        )
        z_matches = match(
            predictions,
            native_gt,
            args.iou_threshold,
            "bev",
            args.score_threshold,
        )

        z_rows = []
        for sample, metas in native_meta.items():
            for local_index, meta in enumerate(metas):
                error = z_matches["error"][sample][local_index]
                z_rows.append(
                    dict(
                        meta,
                        z_error_m=(
                            None if np.isnan(error)
                            else float(error)
                        ),
                    )
                )

        class_z = grouped(z_rows, summarize_z)
        class_z["z_ap"] = {}

        for threshold in args.z_thresholds:
            value = z_threshold_ap(
                predictions,
                native_gt,
                args.iou_threshold,
                threshold,
            )
            class_z["z_ap"][str(threshold)] = float(value)

            if len(z_rows):
                ap_values[str(threshold)].append(value)

        z_report[str(class_name)] = class_z
        all_z_rows.extend(z_rows)

    strata_all = grouped(all_rows, summarize)

    for key in (
        "predictions_total",
        "predictions_at_score",
        "bev_tp_at_score",
        "bev_fp_at_score",
        "d3_tp_at_score",
        "d3_fp_at_score",
    ):
        strata_all["overall"][key] = sum(
            value["overall"][key]
            for value in strata.values()
        )

    for prefix in ("bev", "d3"):
        overall = strata_all["overall"]
        tp = overall[prefix + "_tp_at_score"]
        fp = overall[prefix + "_fp_at_score"]
        overall[prefix + "_precision_at_score"] = (
            tp / max(tp + fp, 1)
        )

    return dict(
        num_samples=len(dataset),
        classes=list(dataset.CLASSES),
        definitions=dict(
            strata_gt=(
                "Legacy valid_flag filtering; no XYZ ROI filtering"
            ),
            z_gt=(
                "Native UAVDataset._base_gt_mask(), including "
                "configured center XYZ ROI"
            ),
            z_error=(
                "predicted geometric center Z minus GT "
                "geometric center Z, meters"
            ),
            z_matching=(
                "Same class; BEV IoU and score thresholds; "
                "one-to-one score-ordered matching"
            ),
            z_ap=(
                "All saved predictions; BEV IoU plus absolute center "
                "Z error threshold; classes without GT excluded from mAP"
            ),
            any_predictions=(
                "All SAVED model outputs, after model-side "
                "filtering and postprocessing"
            ),
            unmatched_any=(
                "Unmatched even with all saved predictions; "
                "may include GT competition"
            ),
            frame_bins=(
                "by_frame retains legacy merged frame numbers; "
                "by_scene_frame separates scenes"
            ),
            ratios=(
                "All AP, precision and recall values are in [0, 1]"
            ),
        ),
        strata=dict(
            classes=strata,
            all_classes=strata_all,
        ),
        z=dict(
            classes=z_report,
            all_classes=grouped(all_z_rows, summarize_z),
            z_map={
                key: float(np.mean(values)) if values else 0.0
                for key, values in ap_values.items()
            },
        ),
    )


def render_report(report):
    def number(value):
        return "n/a" if value is None else "%.4f" % value

    lines = [
        "UAV evaluation",
        "Samples: {}".format(report["num_samples"]),
    ]

    inputs = report.get("inputs", {})
    for key in (
        "config", "checkpoint", "prediction", "split", "source"
    ):
        if key in inputs:
            lines.append("{}: {}".format(key, inputs[key]))

    arguments = inputs.get("arguments", {})
    for key in (
        "iou_threshold",
        "score_threshold",
        "z_thresholds",
        "lower_z",
        "upper_z",
        "multi_level_span",
        "frame_bin",
    ):
        if key in arguments:
            lines.append("{}: {}".format(key, arguments[key]))

    for key, value in report["definitions"].items():
        lines.append("{}: {}".format(key, value))

    for section in ("strata", "z"):
        subjects = dict(report[section]["classes"])
        subjects["ALL"] = report[section]["all_classes"]

        for name, data in subjects.items():
            lines.append(
                "\n{} / {}".format(section.upper(), name)
            )
            lines.append("Overall: " + str(data["overall"]))

            if "z_ap" in data:
                lines.append("Z-AP: " + str(data["z_ap"]))

            for group in GROUPS:
                lines.append(group)

                if section == "strata":
                    lines.append(
                        "group | GT | BEV@score | BEV@any | "
                        "low-score | unmatched@any | "
                        "3D@score | 3D@any"
                    )
                    keys = (
                        "bev_recall_score",
                        "bev_recall_any",
                        "bev_low_confidence_only",
                        "bev_unmatched_any",
                        "d3_recall_score",
                        "d3_recall_any",
                    )
                else:
                    lines.append(
                        "group | GT | matched | MAE(m) | "
                        "RMSE(m) | bias(m) | P50(m) | P90(m)"
                    )
                    keys = (
                        "matches",
                        "mae_m",
                        "rmse_m",
                        "bias_m",
                        "p50_abs_m",
                        "p90_abs_m",
                    )

                for label, values in data[group].items():
                    lines.append(
                        "{} | {} | {}".format(
                            label,
                            values["gt"],
                            " | ".join(
                                number(values[k]) for k in keys
                            ),
                        )
                    )

    lines.append(
        "\nZ-only mAP: " + str(report["z"]["z_map"])
    )
    return "\n".join(lines) + "\n"
