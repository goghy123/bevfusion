"""Single-GPU test, prediction PKL cache and optional UAV JSON export."""

import argparse

from mmcv import DictAction

from uav_eval_io import (
    cache_identity,
    default_cache,
    export_json,
    file_stamp,
    infer,
    load_data,
    save_predictions,
    validate_results,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--out",
        help="Prediction .pkl; default: automatic run cache",
    )
    parser.add_argument(
        "--json-out-dir",
        help="Optional per-frame JSON export",
    )
    parser.add_argument("--eval", nargs="+")
    parser.add_argument("--format-only", action="store_true")
    parser.add_argument(
        "--eval-options", nargs="+", action=DictAction
    )
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction
    )
    parser.add_argument(
        "--run-dir",
        help="Default: checkpoint parent directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    args = parser.parse_args()
    args.split = "test"

    if args.eval and args.format_only:
        parser.error(
            "--eval and --format-only are mutually exclusive"
        )
    if args.out and not args.out.endswith((".pkl", ".pickle")):
        parser.error("--out must end in .pkl or .pickle")

    cfg, data_cfg, dataset, batch_size = load_data(args)

    metadata = cache_identity(
        args, cfg, data_cfg, dataset, batch_size
    )
    results = infer(args, cfg, dataset, batch_size)
    validate_results(results, dataset)

    if file_stamp(args.checkpoint) != metadata["identity"]["checkpoint"]:
        raise RuntimeError(
            "Checkpoint changed during inference; retry"
        )

    path = args.out or default_cache(args)
    save_predictions(results, path, metadata)
    print("Predictions saved:", path)

    if args.json_out_dir:
        export_json(results, dataset, args.json_out_dir)
        print("JSON exported:", args.json_out_dir)

    options = args.eval_options or {}

    if args.format_only:
        dataset.format_results(results, **options)

    if args.eval:
        evaluation = dict(cfg.get("evaluation", {}))

        for key in (
            "interval",
            "tmpdir",
            "start",
            "gpu_collect",
            "save_best",
            "rule",
            "by_epoch",
            "broadcast_bn_buffer",
            "dynamic_intervals",
        ):
            evaluation.pop(key, None)

        evaluation.update(options)
        evaluation["metric"] = args.eval
        print(dataset.evaluate(results, **evaluation))


if __name__ == "__main__":
    main()