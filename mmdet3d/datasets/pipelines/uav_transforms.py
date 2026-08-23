"""Image augmentation variants for a nadir-facing UAV camera."""

import numpy as np

from mmdet.datasets.builder import PIPELINES

from .transforms_3d import ImageAug3D


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


__all__ = ["UAVImageAug3D"]
