#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derive fixed-N UAV sweep info PKLs from an existing converted UAV dataset.

This script DOES NOT reconvert LiDAR or RGB data. It only rewrites the
uavdataset_infos_{train,val,test}.pkl metadata files.

For an existing S9 dataset produced by uavdataset_converter.py:
    info["sweeps"] is ordered nearest -> farthest.
Therefore S3/S6 can be derived exactly for the SAME existing keyframes via:
    info["sweeps"][:3]
    info["sweeps"][:6]

Keeping the same keyframes makes sweep-count ablations fairer than rerunning
the raw converter with a smaller --max-sweeps, because the converter may keep
additional scene-boundary keyframes when fewer historical frames are required.
"""

import argparse
import pickle
from pathlib import Path


SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(
        description="Derive UAV info PKLs with fixed nearest-N historical sweeps."
    )
    p.add_argument(
        "--dataset-root",
        default="data/uavdataset",
        help="Converted UAV dataset root containing the source info PKLs.",
    )
    p.add_argument(
        "--sweeps",
        nargs="+",
        type=int,
        default=[3, 6, 9],
        help="Historical sweep counts to generate.",
    )
    p.add_argument(
        "--source-pattern",
        default="uavdataset_infos_{split}.pkl",
        help="Source info filename pattern. Must contain {split}.",
    )
    p.add_argument(
        "--output-pattern",
        default="uavdataset_infos_{split}_s{sweeps}.pkl",
        help="Output filename pattern. Must contain {split} and {sweeps}.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing derived PKLs.",
    )
    return p.parse_args()


def validate_sweep_order(info, split, sample_index):
    """Validate nearest -> farthest order using timestamps."""
    sweeps = info.get("sweeps", [])
    if not sweeps:
        return

    current_ts = info.get("timestamp")
    if current_ts is None:
        return

    lags = []
    for sweep in sweeps:
        sweep_ts = sweep.get("timestamp")
        if sweep_ts is None:
            return
        lag = current_ts - sweep_ts
        if lag <= 0:
            raise RuntimeError(
                f"{split}[{sample_index}] has a non-historical sweep timestamp."
            )
        lags.append(lag)

    if any(lags[i] > lags[i + 1] for i in range(len(lags) - 1)):
        raise RuntimeError(
            f"{split}[{sample_index}] sweeps are not ordered nearest -> farthest."
        )


def derive_one(source_data, n, split):
    infos = source_data.get("infos")
    if not isinstance(infos, list):
        raise RuntimeError(f"{split}: source PKL does not contain list key 'infos'.")

    derived_infos = []
    for i, info in enumerate(infos):
        validate_sweep_order(info, split, i)

        sweeps = info.get("sweeps")
        if sweeps is None:
            raise RuntimeError(f"{split}[{i}] has no 'sweeps' field.")
        if len(sweeps) < n:
            token = info.get("token", f"index={i}")
            raise RuntimeError(
                f"{split}: sample {token} has only {len(sweeps)} sweeps; "
                f"cannot derive S{n}."
            )

        # Shallow-copy the info so GT/calibration arrays are preserved exactly;
        # only the sweep list is replaced.
        new_info = info.copy()
        new_info["sweeps"] = list(sweeps[:n])
        derived_infos.append(new_info)

    derived = source_data.copy()
    derived["infos"] = derived_infos

    metadata = dict(source_data.get("metadata", {}))
    metadata["max_sweeps"] = n
    metadata["sweep_selection"] = "nearest_N"
    metadata["derived_from_existing_infos"] = True
    derived["metadata"] = metadata

    return derived


def main():
    args = parse_args()

    if "{split}" not in args.source_pattern:
        raise SystemExit("--source-pattern must contain {split}.")
    if "{split}" not in args.output_pattern or "{sweeps}" not in args.output_pattern:
        raise SystemExit("--output-pattern must contain {split} and {sweeps}.")

    requested = sorted(set(args.sweeps))
    if not requested or any(n < 0 for n in requested):
        raise SystemExit("--sweeps must contain non-negative integers.")

    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    generated = []

    for split in SPLITS:
        src = root / args.source_pattern.format(split=split)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source info: {src}")

        with src.open("rb") as f:
            source_data = pickle.load(f)

        source_infos = source_data.get("infos", [])
        source_meta = source_data.get("metadata", {})
        source_max = source_meta.get("max_sweeps")

        print(
            f"[{split}] source={src.name}, samples={len(source_infos)}, "
            f"metadata.max_sweeps={source_max}"
        )

        for n in requested:
            dst = root / args.output_pattern.format(split=split, sweeps=n)

            if dst.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output exists: {dst}. Pass --overwrite to replace it."
                )

            derived = derive_one(source_data, n, split)

            with dst.open("wb") as f:
                pickle.dump(derived, f, protocol=pickle.HIGHEST_PROTOCOL)

            generated.append(dst)
            print(
                f"  S{n}: {dst.name} "
                f"(samples={len(derived['infos'])}, sweeps/sample={n})"
            )

    print("\nGenerated:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
