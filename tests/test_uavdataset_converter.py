import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.data_converter.uavdataset_converter import (
    build_sweep,
    camera_to_reference_transform,
    convert_annotations,
    convert_points,
    estimate_world_velocity,
    load_split_config,
    matrix_to_quaternion,
    reference_height_m,
    world_reference_transform,
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
    def test_scene_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.yaml"
            path.write_text(json.dumps({"splits": {
                "train": ["A"], "val": ["B"], "test": ["C"]}}))
            plan = load_split_config(path, ["A", "B", "C"])
            self.assertEqual(plan["scene_to_split"]["B"], "val")
            with self.assertRaises(ValueError):
                load_split_config(path, ["A", "B", "C", "D"])

    @staticmethod
    def pose(altitude, x=0.0, road=3.0):
        world = np.eye(4)
        world[:3, 3] = [x, 0, road + altitude + 0.2]
        return {
            "uav": {"location": {"z": road + altitude}},
            "lidar": {"location": {"z": road + altitude + 0.2}},
            "road_z_m": road, "altitude_above_road_m": altitude,
            "T_world_lidar": world.tolist(), "T_world_camera": world.tolist(),
        }

    def test_invalid_height(self):
        for key, value in (("road_z_m", float("nan")),
                           ("altitude_above_road_m", 50.0)):
            pose = self.pose(20.0)
            pose[key] = value
            with self.assertRaises(ValueError):
                reference_height_m(pose)
        with self.assertRaises(KeyError):
            reference_height_m({"uav": {"location": {"z": 50}},
                                "lidar": {"location": {"z": 50}}})

    def test_variable_height_points_boxes_projection_and_sweeps(self):
        # One stationary world target must stay aligned through climb, road change and hover.
        cv_from_ue = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                               [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)
        previous = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (altitude, x, road) in enumerate(
                ((50., 0., 3.), (20., 1., 4.), (20., 1., 4.), (35., 2., 3.))
            ):
                pose = self.pose(altitude, x, road)
                height = reference_height_m(pose)
                world = world_reference_transform(pose, height)
                raw = np.array([[4-x, 2, 4.5-road-altitude-0.2, 0.5]], dtype=np.float32)
                source, dest = root / "raw.bin", root / "converted.bin"
                raw.tofile(source)
                convert_points(source, dest, height, 255)
                point = np.fromfile(dest, dtype=np.float32).reshape(-1, 5)[0]
                np.testing.assert_allclose(point[:3], [4-x, -2, 4.5-road], atol=1e-5)
                label = {"objects": [{"class": "car", "num_lidar_points": 20,
                    "bbox3d": {"size_xyz_m": [4, 2, 1.5], "lidar": {
                        "center_xyz_m": raw[0, :3].tolist(),
                        "orientation_matrix": np.eye(3).tolist()}}}]}
                ann, _ = convert_annotations(label, index * 0.1, height, world, {}, 0.5)
                np.testing.assert_allclose(ann["gt_boxes"][0, :3], point[:3], atol=1e-5)
                camera = camera_to_reference_transform(pose, {"T_camera_cv_from_camera_ue": cv_from_ue}, world)
                projected = np.linalg.inv(camera) @ np.r_[point[:3], 1]
                expected = cv_from_ue @ np.r_[raw[0, :3], 1]
                np.testing.assert_allclose(projected, expected, atol=1e-5)
                if previous is not None:
                    prev_world, prev_point = previous
                    sweep = build_sweep("A", index-1, dest, (index-1)*100000,
                                        prev_world, world, root)
                    aligned = prev_point @ sweep["sensor2lidar_rotation"].T + sweep["sensor2lidar_translation"]
                    np.testing.assert_allclose(aligned, point[:3], atol=1e-5)
                previous = world, point[:3].copy()

    def test_velocity(self):
        observations = [
            (0.0, np.asarray([0.0, 0.0, 0.0])),
            (0.1, np.asarray([1.0, 2.0, 0.0])),
            (0.2, np.asarray([2.0, 4.0, 0.0])),
        ]
        velocity = estimate_world_velocity(observations, 0.1, 0.5)
        np.testing.assert_allclose(velocity, [10.0, 20.0, 0.0])

    def test_audit_frame_profiles_use_model_roi(self):
        from tools.analyze_uav_dataset import audit_gt

        info = {"scene_name": "A", "frame_index": 10,
                "route_distance_m": 8.0, "altitude_above_road_m": 20.0,
                "gt_boxes": [[0, 0, -6, 2, 4, 1.5, 0],
                             [0, 0, 30, 2, 4, 1.5, 0]],
                "gt_names": ["bus", "car"], "valid_flag": [True, True],
                "num_lidar_pts": [20, 30]}
        report, _ = audit_gt({"val": [info]}, [-51.2, -51.2, -12, 51.2, 51.2, 14],
                             ["car", "bus"], 1, 10, 10, [5])
        profile = report["splits"]["val"]["frame_profiles_by_scene"]["A"][0]
        self.assertEqual(profile["class_counts_in_roi"], {"bus": 1})
        self.assertEqual(profile["center_z_in_roi"], [-6.0])
        self.assertEqual(profile["frame_index"], 10)

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
