"""Dataset integration and native evaluation for the CARLA UAV dataset."""

import copy
import math
import os.path as osp
from typing import Dict, List, Sequence, Tuple

import mmcv
import numpy as np

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def make_transform(rotation, translation) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    if np.asarray(rotation).shape == (3, 3):
        transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    else:
        transform[:3, :3] = quaternion_to_matrix(rotation)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def box_bev_corners(box: Sequence[float]) -> np.ndarray:
    x, y, _, width, length, _, yaw = [float(value) for value in box[:7]]
    corners = np.asarray(
        [
            [-width / 2.0, -length / 2.0],
            [width / 2.0, -length / 2.0],
            [width / 2.0, length / 2.0],
            [-width / 2.0, length / 2.0],
        ],
        dtype=np.float64,
    )
    rotation_for_row_vectors = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return corners @ rotation_for_row_vectors + np.asarray([x, y], dtype=np.float64)


def cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def line_intersection(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> np.ndarray:
    direction_p = p1 - p0
    direction_q = q1 - q0
    denominator = cross_2d(direction_p, direction_q)
    if abs(denominator) < 1e-12:
        return (p0 + p1) * 0.5
    factor = cross_2d(q0 - p0, direction_q) / denominator
    return p0 + factor * direction_p


def convex_polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = [point for point in subject]
    for edge_index in range(len(clipper)):
        clip_start = clipper[edge_index]
        clip_end = clipper[(edge_index + 1) % len(clipper)]
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


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_values, np.roll(y_values, -1))
            - np.dot(y_values, np.roll(x_values, -1))
        )
    )


def pair_iou(first_box: np.ndarray, second_box: np.ndarray, mode: str) -> float:
    first_polygon = box_bev_corners(first_box)
    second_polygon = box_bev_corners(second_box)
    intersection_polygon = convex_polygon_clip(first_polygon, second_polygon)
    intersection_bev = polygon_area(intersection_polygon)
    if intersection_bev <= 0.0:
        return 0.0
    first_area = float(first_box[3] * first_box[4])
    second_area = float(second_box[3] * second_box[4])
    if mode == "bev":
        union = first_area + second_area - intersection_bev
        return intersection_bev / max(union, 1e-12)

    first_bottom = float(first_box[2])
    second_bottom = float(second_box[2])
    first_top = first_bottom + float(first_box[5])
    second_top = second_bottom + float(second_box[5])
    intersection_height = max(
        0.0, min(first_top, second_top) - max(first_bottom, second_bottom)
    )
    intersection_volume = intersection_bev * intersection_height
    first_volume = first_area * float(first_box[5])
    second_volume = second_area * float(second_box[5])
    union_volume = first_volume + second_volume - intersection_volume
    return intersection_volume / max(union_volume, 1e-12)


def interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    padded_recall = np.concatenate(([0.0], recalls, [1.0]))
    padded_precision = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(padded_precision) - 2, -1, -1):
        padded_precision[index] = max(
            padded_precision[index], padded_precision[index + 1]
        )
    changes = np.where(padded_recall[1:] != padded_recall[:-1])[0]
    return float(
        np.sum(
            (padded_recall[changes + 1] - padded_recall[changes])
            * padded_precision[changes + 1]
        )
    )


def match_predictions(
    predictions: List[Tuple[float, int, np.ndarray]],
    gt_by_sample: Dict[int, np.ndarray],
    iou_threshold: float,
    mode: str,
    score_threshold: float = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ordered = sorted(predictions, key=lambda item: item[0], reverse=True)
    if score_threshold is not None:
        ordered = [item for item in ordered if item[0] >= score_threshold]
    matched = {
        sample_index: np.zeros(len(boxes), dtype=bool)
        for sample_index, boxes in gt_by_sample.items()
    }
    true_positive = np.zeros(len(ordered), dtype=np.float64)
    false_positive = np.zeros(len(ordered), dtype=np.float64)
    for prediction_index, (_, sample_index, prediction_box) in enumerate(ordered):
        gt_boxes = gt_by_sample.get(sample_index)
        if gt_boxes is None or len(gt_boxes) == 0:
            false_positive[prediction_index] = 1.0
            continue
        ious = np.asarray(
            [pair_iou(prediction_box, gt_box, mode) for gt_box in gt_boxes],
            dtype=np.float64,
        )
        ious[matched[sample_index]] = -1.0
        best_index = int(np.argmax(ious))
        if ious[best_index] >= iou_threshold:
            matched[sample_index][best_index] = True
            true_positive[prediction_index] = 1.0
        else:
            false_positive[prediction_index] = 1.0
    num_gt = int(sum(len(boxes) for boxes in gt_by_sample.values()))
    return true_positive, false_positive, num_gt


@DATASETS.register_module()
class UAVDataset(Custom3DDataset):
    """Single-camera UAV Camera+LiDAR dataset produced by the UAV converter."""

    CLASSES = ("car", "van", "truck", "bus")

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        use_valid_flag=True,
        point_cloud_range=None,
    ):
        self.load_interval = int(load_interval)
        self.use_valid_flag = bool(use_valid_flag)
        self.with_velocity = bool(with_velocity)
        self.point_cloud_range = (
            None
            if point_cloud_range is None
            else np.asarray(point_cloud_range, dtype=np.float32)
        )
        if self.point_cloud_range is not None and self.point_cloud_range.shape != (6,):
            raise ValueError("point_cloud_range must contain 6 values")
        self.map_classes = map_classes
        if modality is None:
            modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        if not isinstance(data, dict) or "infos" not in data:
            raise ValueError("UAV info file must contain {'infos', 'metadata'}")
        self.metadata = data.get("metadata", {})
        infos = list(data["infos"])[:: self.load_interval]
        return infos

    def _resolve_path(self, value):
        if osp.isabs(value):
            return value
        return osp.abspath(osp.join(self.dataset_root, value))

    def _base_gt_mask(self, info):
        boxes = np.asarray(info["gt_boxes"], dtype=np.float32).reshape(-1, 7)
        if self.use_valid_flag:
            mask = np.asarray(info["valid_flag"], dtype=bool).copy()
        else:
            mask = np.asarray(info["num_lidar_pts"]) > 0

        # Use the same physical ROI for dataset loading and native evaluation.
        # ROI membership is defined by GT CENTER XYZ. A box may extend partly
        # outside the range and still remain a valid target when its center is
        # inside. Strict bounds match BaseInstance3DBoxes.in_range_3d().
        if self.point_cloud_range is not None and len(boxes):
            low = self.point_cloud_range[:3]
            high = self.point_cloud_range[3:]
            center_inside = (
                (boxes[:, 0] > low[0])
                & (boxes[:, 1] > low[1])
                & (boxes[:, 2] > low[2])
                & (boxes[:, 0] < high[0])
                & (boxes[:, 1] < high[1])
                & (boxes[:, 2] < high[2])
            )
            mask &= center_inside
        return mask

    def get_cat_ids(self, index):
        info = self.data_infos[index]
        mask = self._base_gt_mask(info)
        names = np.asarray(info["gt_names"], dtype=object)[mask]
        return sorted(
            {self.cat2id[str(name)] for name in names if str(name) in self.cat2id}
        )

    def get_data_info(self, index: int) -> Dict:
        info = self.data_infos[index]
        sweeps = copy.deepcopy(info["sweeps"])
        for sweep in sweeps:
            sweep["data_path"] = self._resolve_path(sweep["data_path"])
        data = dict(
            token=info["token"],
            sample_idx=info["token"],
            lidar_path=self._resolve_path(info["lidar_path"]),
            sweeps=sweeps,
            timestamp=info["timestamp"],
            location=info.get("location", "unknown"),
        )

        data["ego2global"] = make_transform(
            info["ego2global_rotation"], info["ego2global_translation"]
        )
        data["lidar2ego"] = make_transform(
            info["lidar2ego_rotation"], info["lidar2ego_translation"]
        )

        if self.modality.get("use_camera", False):
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []
            for camera_info in info["cams"].values():
                data["image_paths"].append(
                    self._resolve_path(camera_info["data_path"])
                )
                camera2lidar = make_transform(
                    camera_info["sensor2lidar_rotation"],
                    camera_info["sensor2lidar_translation"],
                )
                lidar2camera = np.linalg.inv(camera2lidar).astype(np.float32)
                camera_intrinsics = np.eye(4, dtype=np.float32)
                camera_intrinsics[:3, :3] = np.asarray(
                    camera_info["camera_intrinsics"], dtype=np.float32
                )
                data["camera2lidar"].append(camera2lidar)
                data["lidar2camera"].append(lidar2camera)
                data["camera_intrinsics"].append(camera_intrinsics)
                data["lidar2image"].append(camera_intrinsics @ lidar2camera)
                data["camera2ego"].append(
                    make_transform(
                        camera_info["sensor2ego_rotation"],
                        camera_info["sensor2ego_translation"],
                    )
                )

        # Validation/test pipelines still load GT so the native evaluator can run.
        data["ann_info"] = self.get_ann_info(index)
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = self._base_gt_mask(info)
        gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32)[mask]
        gt_names = np.asarray(info["gt_names"], dtype=object)[mask]
        gt_labels = np.asarray(
            [self.cat2id.get(str(name), -1) for name in gt_names], dtype=np.int64
        )
        if self.with_velocity:
            velocity = np.asarray(info["gt_velocity"], dtype=np.float32)[mask]
            velocity[~np.isfinite(velocity)] = 0.0
            gt_boxes = np.concatenate((gt_boxes, velocity), axis=-1)

        # Converter stores geometric centers. Convert them to the bottom-center
        # representation expected by this BEVFusion fork.
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_boxes,
            box_dim=gt_boxes.shape[-1],
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)
        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
            gt_names=gt_names,
        )

    def _ground_truth_for_evaluation(self, class_index: int) -> Dict[int, np.ndarray]:
        gt_by_sample = {}
        class_name = self.CLASSES[class_index]
        for sample_index, info in enumerate(self.data_infos):
            names = np.asarray(info["gt_names"], dtype=object)
            mask = self._base_gt_mask(info) & (names == class_name)
            boxes = np.asarray(info["gt_boxes"], dtype=np.float64)[mask].copy()
            if len(boxes):
                boxes[:, 2] -= boxes[:, 5] * 0.5
            gt_by_sample[sample_index] = boxes.reshape(-1, 7)
        return gt_by_sample

    def _predictions_for_evaluation(
        self, results: Sequence[dict], class_index: int
    ) -> List[Tuple[float, int, np.ndarray]]:
        predictions = []
        for sample_index, result in enumerate(results):
            boxes = as_numpy(result["boxes_3d"].tensor).astype(np.float64)
            scores = as_numpy(result["scores_3d"]).astype(np.float64)
            labels = as_numpy(result["labels_3d"]).astype(np.int64)
            for box, score in zip(boxes[labels == class_index], scores[labels == class_index]):
                predictions.append((float(score), sample_index, box[:7]))
        return predictions

    def evaluate(
        self,
        results,
        metric="bbox",
        iou_threshold=0.5,
        score_threshold=0.1,
        **kwargs
    ):
        if len(results) != len(self):
            raise AssertionError(
                "Prediction count {} does not match dataset size {}".format(
                    len(results), len(self)
                )
            )
        if not results:
            return {}

        metrics = {}
        means = {"bev": [], "3d": []}
        classes_with_gt = 0
        for class_index, class_name in enumerate(self.CLASSES):
            ground_truth = self._ground_truth_for_evaluation(class_index)
            predictions = self._predictions_for_evaluation(results, class_index)
            num_gt = int(sum(len(boxes) for boxes in ground_truth.values()))
            if num_gt > 0:
                classes_with_gt += 1
            for mode, display_name in (("bev", "BEV"), ("3d", "3D")):
                true_positive, false_positive, _ = match_predictions(
                    predictions, ground_truth, float(iou_threshold), mode
                )
                cumulative_tp = np.cumsum(true_positive)
                cumulative_fp = np.cumsum(false_positive)
                recalls = cumulative_tp / max(num_gt, 1)
                precisions = cumulative_tp / np.maximum(
                    cumulative_tp + cumulative_fp, 1e-12
                )
                ap = interpolated_ap(recalls, precisions) if num_gt > 0 else 0.0
                metrics[
                    "uav/{}_AP@{:.2f}/{}".format(
                        display_name, float(iou_threshold), class_name
                    )
                ] = ap
                if num_gt > 0:
                    means[mode].append(ap)

                threshold_tp, threshold_fp, _ = match_predictions(
                    predictions,
                    ground_truth,
                    float(iou_threshold),
                    mode,
                    score_threshold=float(score_threshold),
                )
                tp_count = float(threshold_tp.sum())
                fp_count = float(threshold_fp.sum())
                precision = tp_count / max(tp_count + fp_count, 1.0)
                recall = tp_count / max(num_gt, 1)
                metrics[
                    "uav/{}_precision@score{:.2f}/{}".format(
                        display_name, float(score_threshold), class_name
                    )
                ] = precision
                metrics[
                    "uav/{}_recall@score{:.2f}/{}".format(
                        display_name, float(score_threshold), class_name
                    )
                ] = recall

        metrics["uav/BEV_mAP@{:.2f}".format(float(iou_threshold))] = float(
            np.mean(means["bev"]) if means["bev"] else 0.0
        )
        metrics["uav/3D_mAP@{:.2f}".format(float(iou_threshold))] = float(
            np.mean(means["3d"]) if means["3d"] else 0.0
        )
        metrics["uav/classes_with_gt"] = float(classes_with_gt)

        print("\nUAV detection evaluation (IoU={:.2f})".format(iou_threshold))
        for class_name in self.CLASSES:
            print(
                "  {:>5s}: BEV AP={:.4f}, 3D AP={:.4f}".format(
                    class_name,
                    metrics[
                        "uav/BEV_AP@{:.2f}/{}".format(iou_threshold, class_name)
                    ],
                    metrics[
                        "uav/3D_AP@{:.2f}/{}".format(iou_threshold, class_name)
                    ],
                )
            )
        print(
            "  mean : BEV mAP={:.4f}, 3D mAP={:.4f}".format(
                metrics["uav/BEV_mAP@{:.2f}".format(iou_threshold)],
                metrics["uav/3D_mAP@{:.2f}".format(iou_threshold)],
            )
        )
        return metrics


__all__ = ["UAVDataset"]
