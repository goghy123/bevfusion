"""Shared single-GPU inference, prediction cache and JSON export."""

import copy
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import mmcv
import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def dump_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stamp(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return dict(
        path=str(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def load_data(args):
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet.datasets import replace_ImageToTensor
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import recursive_eval

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    data_cfg = copy.deepcopy(cfg.data[args.split])

    while data_cfg.get("type") in (
        "RepeatDataset",
        "CBGSDataset",
        "ClassBalancedDataset",
    ):
        data_cfg = data_cfg["dataset"]

    if not isinstance(data_cfg.get("ann_file"), str):
        raise ValueError("Expected a single dataset with one ann_file")

    if args.split == "train":
        reference = cfg.data.get("val", cfg.data.test)
        data_cfg["pipeline"] = copy.deepcopy(reference["pipeline"])

    data_cfg["test_mode"] = True
    data_cfg["filter_empty_gt"] = False

    batch_size = int(data_cfg.pop("samples_per_gpu", 1))
    if batch_size > 1:
        data_cfg["pipeline"] = replace_ImageToTensor(data_cfg["pipeline"])

    dataset = build_dataset(data_cfg)
    return cfg, data_cfg, dataset, batch_size


def cache_identity(args, cfg, data_cfg, dataset, batch_size):
    model_cfg = copy.deepcopy(dict(cfg.model))
    model_cfg.update(pretrained=None, train_cfg=None)

    tokens = [
        str(info.get("token", index))
        for index, info in enumerate(dataset.data_infos)
    ]
    order = hashlib.sha256(
        json.dumps(tokens, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    identity = dict(
        version=1,
        checkpoint=file_stamp(args.checkpoint),
        annotations=file_stamp(dataset.ann_file),
        dataset=data_cfg,
        model=model_cfg,
        test_cfg=cfg.get("test_cfg"),
        fp16=cfg.get("fp16"),
        cudnn_benchmark=bool(cfg.get("cudnn_benchmark", False)),
        split=args.split,
        classes=list(dataset.CLASSES),
        samples=len(dataset),
        order=order,
        seed=args.seed,
        deterministic=args.deterministic,
        fuse_conv_bn=args.fuse_conv_bn,
        batch_size=batch_size,
    )

    encoded = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return dict(
        key=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        identity=identity,
    )


def run_directory(args):
    if args.run_dir:
        return Path(args.run_dir).expanduser().absolute()
    return Path(args.checkpoint).expanduser().absolute().parent


def default_cache(args):
    return (
        run_directory(args)
        / "prediction_cache"
        / Path(args.checkpoint).stem
        / (args.split + ".pkl")
    )


def save_predictions(results, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        path.stem + "." + uuid.uuid4().hex + ".tmp.pkl"
    )

    try:
        mmcv.dump(results, str(tmp))
        metadata = dict(metadata, pkl_sha256=digest_file(tmp))
        os.replace(str(tmp), str(path))
        dump_json(metadata, str(path) + ".meta.json")
    finally:
        if tmp.exists():
            tmp.unlink()


def find_cache(args, metadata):
    target = default_cache(args)
    root = run_directory(args)
    candidates = [Path(str(target) + ".meta.json")]

    if root.exists():
        candidates += sorted(root.rglob("*.pkl.meta.json"))
        candidates += sorted(root.rglob("*.pickle.meta.json"))

    for sidecar in dict.fromkeys(candidates):
        path = Path(str(sidecar)[:-len(".meta.json")])
        try:
            saved = json.loads(sidecar.read_text(encoding="utf-8"))

            if saved.get("key") != metadata["key"] or not path.is_file():
                continue

            if args.force_infer:
                return path, None

            if saved.get("pkl_sha256") != digest_file(path):
                continue

            return path, mmcv.load(str(path))

        except (OSError, ValueError, EOFError):
            continue

    return target, None


def infer(args, cfg, dataset, batch_size):
    import torch
    from mmcv.cnn import fuse_conv_bn
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataloader
    from mmdet3d.models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required when predictions must be generated"
        )

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = bool(
        cfg.get("cudnn_benchmark", False)
    )
    set_random_seed(args.seed, deterministic=args.deterministic)

    loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
        dist=False,
        shuffle=False,
    )

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.update(pretrained=None, train_cfg=None)
    model = build_model(model_cfg, test_cfg=cfg.get("test_cfg"))

    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(
        model, args.checkpoint, map_location="cpu"
    )

    classes = checkpoint.get("meta", {}).get("CLASSES")
    if classes is not None and tuple(classes) != tuple(dataset.CLASSES):
        raise ValueError("Checkpoint and dataset class orders differ")

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    return single_gpu_test(model, loader)


def validate_results(results, dataset):
    if not isinstance(results, (list, tuple)) or len(results) != len(dataset):
        raise ValueError(
            "Prediction count does not match the selected split"
        )

    for result in results:
        boxes = as_numpy(result["boxes_3d"].tensor)
        scores = as_numpy(result["scores_3d"])
        labels = as_numpy(result["labels_3d"])

        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("Prediction tensor lengths differ")

        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("Non-finite predictions")

        if np.any(labels < 0) or np.any(labels >= len(dataset.CLASSES)):
            raise ValueError(
                "Prediction labels exceed the dataset class range"
            )


def export_json(results, dataset, output_dir):
    output_dir = Path(output_dir)
    samples, used = [], set()

    for index, (result, info) in enumerate(
        zip(results, dataset.data_infos)
    ):
        token = str(info.get("token", index))
        scene = str(info.get("scene_name", "unknown"))

        try:
            frame = int(
                info.get("frame_index", token.rsplit("/", 1)[-1])
            )
        except (ValueError, TypeError):
            frame = index

        safe_scene = re.sub(r"[^\w.-]", "_", scene)
        if safe_scene in ("", ".", ".."):
            safe_scene = "unknown"

        relative = (
            Path("predictions")
            / safe_scene
            / ("%06d.json" % frame)
        )
        if relative in used:
            relative = relative.with_name(
                "%06d_%d.json" % (frame, index)
            )
        used.add(relative)

        boxes_obj = result["boxes_3d"]
        boxes = as_numpy(boxes_obj.tensor)
        corners = as_numpy(boxes_obj.corners) if len(boxes) else []

        detections = []
        for number, (box, corner, score, label) in enumerate(
            zip(
                boxes,
                corners,
                as_numpy(result["scores_3d"]),
                as_numpy(result["labels_3d"]),
            )
        ):
            label = int(label)
            detections.append(
                dict(
                    index=number,
                    label=label,
                    **{"class": dataset.CLASSES[label]},
                    score=float(score),
                    box_3d_reference=box[:7].tolist(),
                    corners_3d_reference=corner.tolist(),
                )
            )

        dump_json(
            dict(
                format_version=1,
                coordinate_system=(
                    "BEVFusion UAV reference: right-handed, "
                    "x forward, y left, z up, ground-referenced"
                ),
                box_origin="bottom_center",
                token=token,
                scene_name=scene,
                frame_index=frame,
                sample_index=index,
                timestamp_us=int(info.get("timestamp", 0)),
                num_predictions=len(detections),
                detections=detections,
            ),
            output_dir / relative,
        )

        samples.append(
            dict(
                sample_index=index,
                token=token,
                scene_name=scene,
                frame_index=frame,
                prediction_file=relative.as_posix(),
            )
        )

    dump_json(
        dict(
            format_version=1,
            classes=list(dataset.CLASSES),
            num_samples=len(samples),
            samples=samples,
        ),
        output_dir / "manifest.json",
    )