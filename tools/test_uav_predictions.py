#!/usr/bin/env python3
"""Run UAVDataset test inference, export predictions, timings, and dataset metrics.

The script is intentionally single-GPU.  It follows the same config / checkpoint
loading path as tools/test.py, but does not create visualizations or pickle the
raw MMDetection3D result objects.  Instead it writes portable JSON predictions
that can be consumed by scripts/play_uav_results.py.
"""

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a UAVDataset checkpoint and export prediction JSON files."
    )
    parser.add_argument("config", help="Training/test config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory that receives predictions/, timing.csv, metrics.json, summary.json and manifest.json",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold passed to UAVDataset.evaluate(). Default: 0.5",
    )
    parser.add_argument(
        "--eval-score-threshold",
        type=float,
        default=0.1,
        help="Score threshold used for evaluator precision/recall. Predictions themselves are not filtered. Default: 0.1",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Fuse convolution and batch-normalization layers before testing.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config values, e.g. data.test.ann_file=...",
    )
    return parser.parse_args()


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _safe_float(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Non-finite floating-point value encountered while exporting")
    return value


def _class_names(dataset, checkpoint):
    checkpoint_classes = checkpoint.get("meta", {}).get("CLASSES")
    if checkpoint_classes:
        return tuple(str(v) for v in checkpoint_classes)
    return tuple(str(v) for v in dataset.CLASSES)


def _sample_identity(info, sample_index):
    token = str(info.get("token", sample_index))
    scene_name = str(info.get("scene_name", "unknown"))
    frame_index = info.get("frame_index")
    if frame_index is None:
        tail = token.rsplit("/", 1)[-1]
        try:
            frame_index = int(tail)
        except ValueError:
            frame_index = int(sample_index)
    return token, scene_name, int(frame_index)


def _export_result(result, info, sample_index, class_names, inference_ms, output_dir):
    boxes_obj = result["boxes_3d"]
    boxes = _to_numpy(boxes_obj.tensor).astype(np.float64, copy=False)
    if len(boxes):
        corners = _to_numpy(boxes_obj.corners).astype(np.float64, copy=False)
    else:
        corners = np.empty((0, 8, 3), dtype=np.float64)
    scores = _to_numpy(result["scores_3d"]).astype(np.float64, copy=False)
    labels = _to_numpy(result["labels_3d"]).astype(np.int64, copy=False)

    if not (len(boxes) == len(corners) == len(scores) == len(labels)):
        raise RuntimeError("Prediction tensor lengths do not match")

    token, scene_name, frame_index = _sample_identity(info, sample_index)
    detections = []
    for det_index, (box, box_corners, score, label) in enumerate(
        zip(boxes, corners, scores, labels)
    ):
        label = int(label)
        class_name = class_names[label] if 0 <= label < len(class_names) else "class_{}".format(label)
        detections.append(
            {
                "index": int(det_index),
                "class": class_name,
                "label": label,
                "score": _safe_float(score),
                "box_3d_reference": [_safe_float(v) for v in box[:7]],
                "corners_3d_reference": [
                    [_safe_float(v) for v in corner[:3]] for corner in box_corners
                ],
            }
        )

    relative_path = Path("predictions") / scene_name / "{:06d}.json".format(frame_index)
    payload = {
        "format_version": 1,
        "coordinate_system": "BEVFusion UAV reference: right-handed, x forward, y left, z up, ground-referenced",
        "token": token,
        "scene_name": scene_name,
        "frame_index": int(frame_index),
        "sample_index": int(sample_index),
        "timestamp_us": int(info.get("timestamp", 0)),
        "inference_ms": _safe_float(inference_ms),
        "num_predictions": int(len(detections)),
        "detections": detections,
    }
    _json_dump(payload, output_dir / relative_path)
    return str(relative_path.as_posix()), token, scene_name, frame_index, len(detections)


def _build_test_components(args):
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if samples_per_gpu != 1:
        raise ValueError(
            "This exporter requires samples_per_gpu=1 so every timing value maps to exactly one test frame. "
            "Override data.test.samples_per_gpu=1."
        )

    set_random_seed(args.seed, deterministic=args.deterministic)
    dataset = build_dataset(cfg.data.test)
    if dataset.__class__.__name__ != "UAVDataset":
        print("WARNING: dataset class is {}, not UAVDataset.".format(dataset.__class__.__name__))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    class_names = _class_names(dataset, checkpoint)
    model.CLASSES = class_names
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return cfg, dataset, data_loader, model, class_names


def _timing_summary(values_ms):
    values_ms = [float(v) for v in values_ms]
    total_ms = float(sum(values_ms))
    count = len(values_ms)
    if not values_ms:
        return {
            "samples": 0,
            "total_inference_ms": 0.0,
            "average_inference_ms": 0.0,
            "median_inference_ms": 0.0,
            "min_inference_ms": 0.0,
            "max_inference_ms": 0.0,
            "inference_fps": 0.0,
        }
    return {
        "samples": int(count),
        "total_inference_ms": total_ms,
        "average_inference_ms": float(statistics.fmean(values_ms)),
        "median_inference_ms": float(statistics.median(values_ms)),
        "min_inference_ms": float(min(values_ms)),
        "max_inference_ms": float(max(values_ms)),
        "inference_fps": float(1000.0 * count / total_ms) if total_ms > 0 else 0.0,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this timing script")
    if not (0.0 < args.iou_threshold <= 1.0):
        raise ValueError("--iou-threshold must be in (0, 1]")
    if args.eval_score_threshold < 0.0:
        raise ValueError("--eval-score-threshold must be >= 0")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    out_dir = Path(args.out_dir).expanduser().resolve()
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    cfg, dataset, data_loader, model, class_names = _build_test_components(args)
    print("Test samples : {}".format(len(dataset)))
    print("Classes      : {}".format(", ".join(class_names)))
    print("Output       : {}".format(out_dir))

    results = []
    timings = []
    manifest_samples = []
    progress = mmcv.ProgressBar(len(dataset))
    wall_start = time.perf_counter()

    for sample_index, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            batch_result = model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000.0

        if len(batch_result) != 1:
            raise RuntimeError("Expected exactly one prediction result per test iteration")
        result = batch_result[0]
        results.append(result)

        info = dataset.data_infos[sample_index]
        relative_path, token, scene_name, frame_index, num_predictions = _export_result(
            result=result,
            info=info,
            sample_index=sample_index,
            class_names=class_names,
            inference_ms=inference_ms,
            output_dir=out_dir,
        )
        timings.append(
            {
                "sample_index": int(sample_index),
                "token": token,
                "scene_name": scene_name,
                "frame_index": int(frame_index),
                "inference_ms": float(inference_ms),
                "num_predictions": int(num_predictions),
            }
        )
        manifest_samples.append(
            {
                "sample_index": int(sample_index),
                "token": token,
                "scene_name": scene_name,
                "frame_index": int(frame_index),
                "prediction_file": relative_path,
            }
        )
        progress.update()

    wall_seconds = time.perf_counter() - wall_start
    print()

    eval_start = time.perf_counter()
    metrics = dataset.evaluate(
        results,
        metric="bbox",
        iou_threshold=float(args.iou_threshold),
        score_threshold=float(args.eval_score_threshold),
    )
    eval_seconds = time.perf_counter() - eval_start
    metrics = {str(k): _safe_float(v) for k, v in metrics.items()}

    timing_values = [row["inference_ms"] for row in timings]
    timing_summary = _timing_summary(timing_values)
    summary = {
        "format_version": 1,
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "output_dir": str(out_dir),
        "classes": list(class_names),
        "timing_definition": (
            "CUDA-synchronized duration of model(return_loss=False, rescale=True, **data); "
            "DataLoader disk I/O is outside the timed interval, while MMDataParallel CPU-to-GPU scatter, "
            "forward, decoding and model-side post-processing are inside it."
        ),
        "timing": timing_summary,
        "whole_test_wall_time_s": float(wall_seconds),
        "evaluation_wall_time_s": float(eval_seconds),
        "evaluation": {
            "iou_threshold": float(args.iou_threshold),
            "score_threshold": float(args.eval_score_threshold),
            "metrics": metrics,
        },
    }

    with (out_dir / "timing.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "token",
                "scene_name",
                "frame_index",
                "inference_ms",
                "num_predictions",
            ],
        )
        writer.writeheader()
        writer.writerows(timings)

    _json_dump(metrics, out_dir / "metrics.json")
    _json_dump(summary, out_dir / "summary.json")
    _json_dump(
        {
            "format_version": 1,
            "classes": list(class_names),
            "num_samples": int(len(manifest_samples)),
            "samples": manifest_samples,
        },
        out_dir / "manifest.json",
    )

    print("Inference timing")
    print("  samples : {}".format(timing_summary["samples"]))
    print("  total   : {:.3f} s".format(timing_summary["total_inference_ms"] / 1000.0))
    print("  average : {:.3f} ms".format(timing_summary["average_inference_ms"]))
    print("  median  : {:.3f} ms".format(timing_summary["median_inference_ms"]))
    print(
        "  min/max : {:.3f} / {:.3f} ms".format(
            timing_summary["min_inference_ms"], timing_summary["max_inference_ms"]
        )
    )
    print("  FPS     : {:.3f}".format(timing_summary["inference_fps"]))
    print("  wall    : {:.3f} s (test loop, excluding evaluation)".format(wall_seconds))
    print("Results written to {}".format(out_dir))


if __name__ == "__main__":
    main()
