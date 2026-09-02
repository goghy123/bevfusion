"""Image augmentation variants for a nadir-facing UAV camera."""

import numpy as np

from mmdet.datasets.builder import PIPELINES

from .transforms_3d import ImageAug3D



@PIPELINES.register_module()
class LiDARCameraFrustumFilter:
    """Keep LiDAR points visible in the raw camera frustum.

    The frustum is derived from ``lidar2image`` and the actual loaded image
    size. No camera FOV, resolution, altitude, or mounting angle is hard-coded.
    With multiple cameras, ``mode='union'`` keeps a point if it is visible in
    at least one camera.

    This transform should run after multi-sweep loading and before image
    augmentation. Dataset GT visibility is defined from the raw sensors, while
    random image crop/resize/flip/rotation is only training augmentation.

    Args:
        min_depth (float): Minimum positive camera depth in metres.
        max_depth (float | None): Optional maximum camera depth.
        margin_px (float): Shrink each image border by this many pixels.
        min_points (int): Minimum number of retained points.
        min_keep_ratio (float): Minimum retained/original point ratio.
        mode (str): ``union`` or ``intersection`` for multiple cameras.
        on_threshold_failure (str): ``raise`` or ``keep``.
    """

    def __init__(
        self,
        min_depth=0.05,
        max_depth=None,
        margin_px=2.0,
        min_points=1000,
        min_keep_ratio=0.005,
        mode="union",
        on_threshold_failure="raise",
    ):
        self.min_depth = float(min_depth)
        self.max_depth = None if max_depth is None else float(max_depth)
        self.margin_px = float(margin_px)
        self.min_points = int(min_points)
        self.min_keep_ratio = float(min_keep_ratio)
        self.mode = str(mode)
        self.on_threshold_failure = str(on_threshold_failure)

        if self.min_depth <= 0:
            raise ValueError("min_depth must be > 0")
        if self.max_depth is not None and self.max_depth <= self.min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        if self.margin_px < 0:
            raise ValueError("margin_px must be >= 0")
        if self.min_points < 0:
            raise ValueError("min_points must be >= 0")
        if not 0.0 <= self.min_keep_ratio <= 1.0:
            raise ValueError("min_keep_ratio must be in [0, 1]")
        if self.mode not in ("union", "intersection"):
            raise ValueError("mode must be 'union' or 'intersection'")
        if self.on_threshold_failure not in ("raise", "keep"):
            raise ValueError("on_threshold_failure must be 'raise' or 'keep'")

    @staticmethod
    def _image_size(image):
        # PIL.Image.size is (width, height). Keep numpy support for reuse.
        if hasattr(image, "size") and not isinstance(image, np.ndarray):
            width, height = image.size
            return int(width), int(height)
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            return int(width), int(height)
        raise TypeError(
            "Unsupported image type for frustum filtering: {}".format(type(image))
        )

    def __call__(self, data):
        if "points" not in data:
            raise KeyError("LiDARCameraFrustumFilter requires data['points']")
        if "lidar2image" not in data:
            raise KeyError("LiDARCameraFrustumFilter requires data['lidar2image']")
        if "img" not in data:
            raise KeyError("LiDARCameraFrustumFilter requires loaded data['img']")

        points = data["points"]
        num_points = len(points)
        if num_points == 0:
            raise RuntimeError(
                "LiDARCameraFrustumFilter received an empty point cloud"
            )

        images = list(data["img"])
        matrices = list(data["lidar2image"])
        if len(images) != len(matrices) or len(images) == 0:
            raise RuntimeError(
                "Camera/image calibration mismatch: {} images vs {} "
                "lidar2image matrices".format(len(images), len(matrices))
            )

        xyz = (
            points.tensor[:, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        homogeneous = np.ones((num_points, 4), dtype=np.float64)
        homogeneous[:, :3] = xyz

        if self.mode == "union":
            keep = np.zeros(num_points, dtype=bool)
        else:
            keep = np.ones(num_points, dtype=bool)

        per_camera = []
        for camera_index, (image, matrix) in enumerate(zip(images, matrices)):
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(
                    "lidar2image[{}] must be 4x4, got {}".format(
                        camera_index, matrix.shape
                    )
                )

            width, height = self._image_size(image)
            if width <= 2 * self.margin_px or height <= 2 * self.margin_px:
                raise ValueError(
                    "margin_px={} leaves no valid image area for {}x{} image".format(
                        self.margin_px, width, height
                    )
                )

            projected = homogeneous @ matrix.T
            depth = projected[:, 2]
            valid_depth = np.isfinite(depth) & (depth > self.min_depth)
            if self.max_depth is not None:
                valid_depth &= depth <= self.max_depth

            u = np.full(num_points, np.nan, dtype=np.float64)
            v = np.full(num_points, np.nan, dtype=np.float64)
            u[valid_depth] = projected[valid_depth, 0] / depth[valid_depth]
            v[valid_depth] = projected[valid_depth, 1] / depth[valid_depth]

            visible = (
                valid_depth
                & np.isfinite(u)
                & np.isfinite(v)
                & (u >= self.margin_px)
                & (u < width - self.margin_px)
                & (v >= self.margin_px)
                & (v < height - self.margin_px)
            )

            if self.mode == "union":
                keep |= visible
            else:
                keep &= visible
            per_camera.append(int(visible.sum()))

        kept_points = int(keep.sum())
        keep_ratio = float(kept_points / max(num_points, 1))
        data["frustum_filter_stats"] = {
            "before": int(num_points),
            "after": kept_points,
            "keep_ratio": keep_ratio,
            "per_camera_visible": per_camera,
            "mode": self.mode,
        }

        threshold_failed = (
            kept_points < self.min_points or keep_ratio < self.min_keep_ratio
        )
        if threshold_failed:
            message = (
                "LiDARCameraFrustumFilter threshold failure: before={}, "
                "after={}, keep_ratio={:.6f}, per_camera={}, min_points={}, "
                "min_keep_ratio={:.6f}. Check camera intrinsics/extrinsics, "
                "image resolution/FOV, and LiDAR-to-camera calibration."
            ).format(
                num_points,
                kept_points,
                keep_ratio,
                per_camera,
                self.min_points,
                self.min_keep_ratio,
            )
            if self.on_threshold_failure == "raise":
                raise RuntimeError(message)
            data["frustum_filter_stats"]["threshold_failure"] = message
            return data

        data["points"] = points[keep]
        return data

    def __repr__(self):
        return (
            "{}(min_depth={}, max_depth={}, margin_px={}, min_points={}, "
            "min_keep_ratio={}, mode={!r}, on_threshold_failure={!r})"
        ).format(
            self.__class__.__name__,
            self.min_depth,
            self.max_depth,
            self.margin_px,
            self.min_points,
            self.min_keep_ratio,
            self.mode,
            self.on_threshold_failure,
        )


@PIPELINES.register_module()
class UAVObjectRangeFilter3D:
    """Filter UAV GT boxes by center XYZ using ``point_cloud_range``.

    The generic ObjectRangeFilter in this BEVFusion fork uses
    ``in_range_bev`` for LiDAR boxes, so it checks only GT center X/Y.
    UAV scenes can contain meaningful multi-level Z structure, therefore
    training GT ROI is defined here by center X/Y/Z using the same physical
    range as PointsRangeFilter.

    A GT box is kept when its CENTER satisfies strict bounds:
        xmin < x < xmax
        ymin < y < ymax
        zmin < z < zmax

    The whole 3D box is NOT required to fit inside the range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.asarray(point_cloud_range, dtype=np.float32)
        if self.pcd_range.shape != (6,):
            raise ValueError("point_cloud_range must contain 6 values")

    def __call__(self, data):
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        mask = gt_bboxes_3d.in_range_3d(self.pcd_range)
        data["gt_bboxes_3d"] = gt_bboxes_3d[mask]
        data["gt_labels_3d"] = gt_labels_3d[
            mask.detach().cpu().numpy().astype(np.bool_)
        ]

        data["gt_bboxes_3d"].limit_yaw(offset=0.5, period=2 * np.pi)
        return data

    def __repr__(self):
        return "{}(point_cloud_range={})".format(
            self.__class__.__name__, self.pcd_range.tolist()
        )


@PIPELINES.register_module()
class UAVImageAug3D(ImageAug3D):
    """ImageAug3D with a centered vertical crop for top-down imagery.

    The original transform intentionally keeps the bottom of a road-facing
    image.  A nadir camera has no privileged image bottom, so vertical crops
    are centered with a small optional training jitter.
    """

    def __init__(self, *args, **kwargs):
        self.vertical_jitter = float(kwargs.pop("vertical_jitter", 0.25))
        if not 0.0 <= self.vertical_jitter <= 0.5:
            raise ValueError("vertical_jitter must be in [0, 0.5]")
        super().__init__(*args, **kwargs)

    def sample_augmentation(self, results):
        width, height = results["ori_shape"]
        final_height, final_width = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resized_width = int(width * resize)
            resized_height = int(height * resize)
            if resized_width < final_width or resized_height < final_height:
                raise ValueError(
                    "resize_lim produces an image smaller than final_dim: "
                    "{}x{} < {}x{}".format(
                        resized_width,
                        resized_height,
                        final_width,
                        final_height,
                    )
                )
            horizontal_margin = resized_width - final_width
            vertical_margin = resized_height - final_height
            crop_x = int(np.random.uniform(0, horizontal_margin + 1))
            jitter = np.random.uniform(
                -self.vertical_jitter, self.vertical_jitter
            ) * vertical_margin
            crop_y = int(np.clip(vertical_margin * 0.5 + jitter, 0, vertical_margin))
            flip = bool(self.rand_flip and np.random.choice([0, 1]))
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = float(np.mean(self.resize_lim))
            resized_width = int(width * resize)
            resized_height = int(height * resize)
            if resized_width < final_width or resized_height < final_height:
                raise ValueError(
                    "Test resize is smaller than final_dim: {}x{} < {}x{}".format(
                        resized_width,
                        resized_height,
                        final_width,
                        final_height,
                    )
                )
            crop_x = int((resized_width - final_width) / 2)
            crop_y = int((resized_height - final_height) / 2)
            flip = False
            rotate = 0.0
        crop = (
            crop_x,
            crop_y,
            crop_x + final_width,
            crop_y + final_height,
        )
        return (
            resize,
            (resized_width, resized_height),
            crop,
            flip,
            rotate,
        )


__all__ = [
    "LiDARCameraFrustumFilter",
    "UAVObjectRangeFilter3D",
    "UAVImageAug3D",
]
