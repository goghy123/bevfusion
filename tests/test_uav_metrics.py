import ast
import math
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


try:
    from mmdet3d.datasets.uav_dataset import (
        interpolated_ap,
        match_predictions,
        pair_iou,
    )
except ModuleNotFoundError as error:
    # The converter's lightweight CI may not have mmcv/mmdet installed.  Load
    # the exact pure-numpy metric functions from the dataset module so their
    # numerical tests still run; a full BEVFusion environment uses the normal
    # import path above.
    if error.name not in {"mmcv", "mmdet", "torch"}:
        raise
    source_path = (
        Path(__file__).resolve().parents[1]
        / "mmdet3d"
        / "datasets"
        / "uav_dataset.py"
    )
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    metric_names = {
        "box_bev_corners",
        "cross_2d",
        "line_intersection",
        "convex_polygon_clip",
        "polygon_area",
        "pair_iou",
        "interpolated_ap",
        "match_predictions",
    }
    metric_tree = ast.Module(
        body=[
            node
            for node in source_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in metric_names
        ],
        type_ignores=[],
    )
    namespace = {
        "math": math,
        "np": np,
        "Dict": Dict,
        "List": List,
        "Sequence": Sequence,
        "Tuple": Tuple,
    }
    exec(compile(metric_tree, str(source_path), "exec"), namespace)
    interpolated_ap = namespace["interpolated_ap"]
    match_predictions = namespace["match_predictions"]
    pair_iou = namespace["pair_iou"]


class UAVMetricTest(unittest.TestCase):
    def test_oriented_iou(self):
        box = np.asarray([0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0])
        same = box.copy()
        disjoint = box.copy()
        disjoint[0] = 10.0
        rotated = box.copy()
        rotated[6] = math.pi / 2.0
        self.assertAlmostEqual(pair_iou(box, same, "bev"), 1.0)
        self.assertAlmostEqual(pair_iou(box, same, "3d"), 1.0)
        self.assertEqual(pair_iou(box, disjoint, "bev"), 0.0)
        self.assertAlmostEqual(pair_iou(box, rotated, "bev"), 1.0 / 3.0)

    def test_matching_and_ap(self):
        gt_box = np.asarray([0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0])
        false_box = gt_box.copy()
        false_box[0] = 10.0
        predictions = [(0.9, 0, gt_box), (0.8, 0, false_box)]
        true_positive, false_positive, num_gt = match_predictions(
            predictions, {0: np.asarray([gt_box])}, 0.5, "3d"
        )
        self.assertEqual(num_gt, 1)
        np.testing.assert_array_equal(true_positive, [1.0, 0.0])
        np.testing.assert_array_equal(false_positive, [0.0, 1.0])
        recall = np.cumsum(true_positive) / num_gt
        precision = np.cumsum(true_positive) / (
            np.cumsum(true_positive) + np.cumsum(false_positive)
        )
        self.assertAlmostEqual(interpolated_ap(recall, precision), 1.0)


if __name__ == "__main__":
    unittest.main()
