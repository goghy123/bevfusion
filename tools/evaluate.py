"""Evaluate cached or freshly generated UAV predictions on train/val/test."""

import argparse
import math
from pathlib import Path

import mmcv
from mmcv import DictAction

from uav_eval_io import (
    cache_identity,
    dump_json,
    file_stamp,
    find_cache,
    infer,
    load_data,
    run_directory,
    save_predictions,
    validate_results,
)
from uav_evaluation import evaluate, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint", nargs="?")

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--force-infer", action="store_true")
    parser.add_argument(
        "--pred",
        help=(
            "Optional explicit PKL, without automatic "
            "provenance checks"
        ),
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output-dir")

    parser.add_argument(
        "--iou-threshold", "--iou",
        type=float, default=0.5,
    )
    parser.add_argument(
        "--score-threshold", "--score",
        type=float, default=0.1,
    )
    parser.add_argument(
        "--z-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument("--lower-z", type=float, default=-5.0)
    parser.add_argument("--upper-z", type=float, default=5.0)
    parser.add_argument(
        "--multi-level-span", type=float, default=5.0
    )
    parser.add_argument("--frame-bin", type=int, default=500)

    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    args = parser.parse_args()

    if not args.pred and not args.checkpoint:
        parser.error(
            "Supply a checkpoint for automatic caching, "
            "or --pred for offline analysis"
        )
    if args.pred and args.force_infer:
        parser.error(
            "--pred and --force-infer are mutually exclusive"
        )

    values = [
        args.iou_threshold,
        args.score_threshold,
        args.lower_z,
        args.upper_z,
        args.multi_level_span,
    ] + list(args.z_thresholds)

    if not all(math.isfinite(value) for value in values):
        parser.error("Thresholds must be finite")

    if not 0 < args.iou_threshold <= 1 or args.score_threshold < 0:
        parser.error(
            "IoU must be in (0,1] and score must be non-negative"
        )

    if (
        args.lower_z >= args.upper_z
        or args.multi_level_span < 0
        or args.frame_bin <= 0
    ):
        parser.error("Invalid group boundaries")

    if any(value < 0 for value in args.z_thresholds):
        parser.error("Z error thresholds must be non-negative")

    cfg, data_cfg, dataset, batch_size = load_data(args)

    if dataset.__class__.__name__ != "UAVDataset":
        raise TypeError("This evaluator requires UAVDataset")

    if args.pred:
        path = Path(args.pred)
        results = mmcv.load(str(path))
        source, metadata = "explicit_unverified", None

    else:
        metadata = cache_identity(
            args, cfg, data_cfg, dataset, batch_size
        )
        path, results = find_cache(args, metadata)
        source = "cache"

        if results is None:
            print("Running inference; prediction cache:", path)
            results = infer(args, cfg, dataset, batch_size)
            validate_results(results, dataset)

            if (
                file_stamp(args.checkpoint)
                != metadata["identity"]["checkpoint"]
            ):
                raise RuntimeError(
                    "Checkpoint changed during inference; retry"
                )

            save_predictions(results, path, metadata)
            source = "inference"

    validate_results(results, dataset)
    print("Prediction source:", source, path)

    report = evaluate(dataset, results, args)
    report["inputs"] = dict(
        config=str(Path(args.config).absolute()),
        checkpoint=(
            str(Path(args.checkpoint).absolute())
            if args.checkpoint else None
        ),
        prediction=str(path.absolute()),
        source=source,
        split=args.split,
        cache_metadata=metadata,
        arguments=vars(args),
    )

    if args.checkpoint:
        root = run_directory(args)
    else:
        root = (
            Path(args.run_dir)
            if args.run_dir else path.parent
        )

    model_name = (
        Path(args.checkpoint).stem
        if args.checkpoint else "offline"
    )
    output = (
        Path(args.output_dir)
        if args.output_dir else
        root / "evaluation" / model_name / args.split
    )

    dump_json(report, output / "evaluation.json")

    text = render_report(report)
    (output / "report.txt").write_text(
        text, encoding="utf-8"
    )

    print(text)
    print("Reports saved:", output)


if __name__ == "__main__":
    main()