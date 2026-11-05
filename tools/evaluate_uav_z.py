#!/usr/bin/env python3
"""Evaluate UAV detector height accuracy on one configured dataset split."""

import argparse
from types import SimpleNamespace

import numpy as np
import torch
from torchpack import distributed as dist

from mmdet3d.datasets.uav_dataset import (
    matched_z_errors,
    z_error_summary,
    z_threshold_ap,
)
from test_uav_predictions import _build_test_components


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Training config")
    parser.add_argument("checkpoint", help="Checkpoint to evaluate")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument(
        "--z-thresholds",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0),
        metavar="METERS",
        help="Allowed absolute center-Z error for Z-only AP (default: 0.5 1.0 2.0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.iou_threshold <= 1.0):
        raise ValueError("--iou-threshold must be in (0, 1]")
    if args.score_threshold < 0.0:
        raise ValueError("--score-threshold must be >= 0")
    if not args.z_thresholds or any(value < 0.0 for value in args.z_thresholds):
        raise ValueError("--z-thresholds must contain non-negative values")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run model inference")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    build_args = SimpleNamespace(
        config=args.config,
        checkpoint=args.checkpoint,
        cfg_options=None,
        seed=args.seed,
        deterministic=args.deterministic,
        fuse_conv_bn=args.fuse_conv_bn,
    )
    _, dataset, data_loader, model, class_names = _build_test_components(
        build_args, split=args.split
    )
    if dataset.__class__.__name__ != "UAVDataset":
        raise TypeError("Expected UAVDataset, got {}".format(dataset.__class__.__name__))

    results = []
    for data in data_loader:
        with torch.no_grad():
            batch_results = model(return_loss=False, rescale=True, **data)
        if len(batch_results) != 1:
            raise RuntimeError("Expected one prediction result per sample")
        results.append(batch_results[0])
    if len(results) != len(dataset):
        raise RuntimeError("Prediction count does not match selected split")

    all_errors = []
    class_aps = {threshold: [] for threshold in args.z_thresholds}
    print("Split: {} ({} samples)".format(args.split, len(dataset)))
    print("BEV IoU threshold: {:.2f}; Z-error score threshold: {:.2f}".format(
        args.iou_threshold, args.score_threshold
    ))
    for class_index, class_name in enumerate(class_names):
        gt_by_sample = dataset._ground_truth_for_evaluation(class_index)
        predictions = dataset._predictions_for_evaluation(results, class_index)
        errors = matched_z_errors(
            predictions, gt_by_sample, args.iou_threshold, args.score_threshold
        )
        all_errors.extend(errors)
        summary = z_error_summary(errors)
        aps = {}
        num_gt = sum(len(boxes) for boxes in gt_by_sample.values())
        for threshold in args.z_thresholds:
            ap = z_threshold_ap(
                predictions, gt_by_sample, args.iou_threshold, threshold
            )
            aps[threshold] = ap
            if num_gt:
                class_aps[threshold].append(ap)
        print(
            "{}: GT={} matched@score={}; Z MAE/RMSE/bias={}/{}/{} m; {}".format(
                class_name,
                num_gt,
                summary["matches"],
                *("n/a" if summary[key] is None else "{:.3f}".format(summary[key])
                  for key in ("mae_m", "rmse_m", "bias_m")),
                " ".join("Z-AP@{:.2f}m={:.4f}".format(t, aps[t]) for t in args.z_thresholds),
            )
        )

    overall = z_error_summary(all_errors)
    print(
        "ALL: matched@score={}; Z MAE/RMSE/bias={}/{}/{} m; P50/P90 |error|={}/{} m".format(
            overall["matches"],
            *("n/a" if overall[key] is None else "{:.3f}".format(overall[key])
              for key in ("mae_m", "rmse_m", "bias_m", "p50_abs_m", "p90_abs_m")),
        )
    )
    print("Z-only mAP: {}".format(
        ", ".join(
            "@{:.2f}m={:.4f}".format(t, float(np.mean(class_aps[t])) if class_aps[t] else 0.0)
            for t in args.z_thresholds
        )
    ))


if __name__ == "__main__":
    main()
