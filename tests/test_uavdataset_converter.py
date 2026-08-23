import math
import unittest

import numpy as np

from tools.data_converter.uavdataset_converter import (
    allocate_split_counts,
    estimate_world_velocity,
    matrix_to_quaternion,
    scene_split_segments,
)


def quaternion_to_matrix(quaternion):
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class UAVConverterMathTest(unittest.TestCase):
    def test_split_counts(self):
        self.assertEqual(allocate_split_counts(100, (0.7, 0.15, 0.15)), [70, 15, 15])
        segments = scene_split_segments(list(range(500)), 5, (0.7, 0.15, 0.15))
        self.assertEqual(
            sum(len(segments[name]["candidate_keyframes"]) for name in ("train", "val", "test")),
            100,
        )

    def test_velocity(self):
        observations = [
            (0.0, np.asarray([0.0, 0.0, 0.0])),
            (0.1, np.asarray([1.0, 2.0, 0.0])),
            (0.2, np.asarray([2.0, 4.0, 0.0])),
        ]
        velocity = estimate_world_velocity(observations, 0.1, 0.5)
        np.testing.assert_allclose(velocity, [10.0, 20.0, 0.0])

    def test_quaternion_round_trip(self):
        for angle in (0.0, 0.4, -2.2, 3.1):
            rotation = np.asarray(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            reconstructed = quaternion_to_matrix(matrix_to_quaternion(rotation))
            np.testing.assert_allclose(rotation, reconstructed, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
