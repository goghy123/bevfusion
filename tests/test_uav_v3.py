import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data_converter.uavdataset_converter import (
    build_argument_parser, convert_dataset, world_reference_transform,
    reference_from_raw_lidar, convert_points, convert_annotations,
    camera_to_reference_transform, build_sweep,
)
from tools.rebalance_uav_splits import SplitPlan, validate_scene_coverage, rebuild, parse_scene_frame


class UAVV3Test(unittest.TestCase):
    @staticmethod
    def fixture(index):
        angle = .035 * np.sin(index / 7.)
        c, s = np.cos(angle), np.sin(angle)
        world = np.eye(4)
        world[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        altitude = 35 + 15 * np.cos(index / 80.)
        world[:3, 3] = [index * .5, 2., 3 + altitude]
        camera = world.copy()
        camera[:3, :3] = world[:3, :3] @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        pose = dict(uav={"location": {"z": 3+altitude}}, lidar={"location": {"z": 3+altitude}},
                    road_z_m=3., altitude_above_road_m=altitude, T_world_lidar=world.tolist(),
                    T_world_camera=camera.tolist(), base_yaw_deg=0., carla_timestamp=index/10,
                    route_distance_m=index*.5, route_progress=index/220, dataset_time_s=index/10,
                    motion_time_s=index/10, is_hovering=False, hover_index=None,
                    uav_effective_speed_mps=5.)
        center = np.array([index*.5+4, 4, 4.5])
        raw = world[:3, :3].T @ (center - world[:3, 3])
        label = {"objects": [{"class": "car", "num_lidar_points": 20, "bbox3d": {
            "size_xyz_m": [4, 2, 1.5], "lidar": {"center_xyz_m": raw.tolist(),
                "orientation_matrix": world[:3, :3].T.tolist()}}}]}
        calibration = dict(K=[[4, 0, 4], [0, 4, 4], [0, 0, 1]], camera_resolution=[8, 8],
                           T_camera_cv_from_camera_ue=[[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        return pose, raw, label, calibration

    def test_tilted_points_boxes_projection_and_inverse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/"raw.bin", Path(tmp)/"out.bin"
            for index in (0, 25, 80, 160):
                pose, raw, label, calibration = self.fixture(index)
                reference = world_reference_transform(pose, pose["altitude_above_road_m"])
                transform = reference_from_raw_lidar(pose, reference)
                np.array([[*raw, .5]], dtype=np.float32).tofile(source)
                convert_points(source, dest, 0, 255, transform)
                point = np.fromfile(dest, dtype=np.float32).reshape(-1, 5)[0, :3]
                np.testing.assert_allclose(point, [4., -2., 1.5], atol=1e-5)
                ann, _ = convert_annotations(label, index/10, 0, reference, {}, .5, transform)
                np.testing.assert_allclose(ann["gt_boxes"][0, :3], point, atol=1e-5)
                np.testing.assert_allclose((np.linalg.inv(transform) @ [*point, 1])[:3], raw, atol=1e-5)
                camera = camera_to_reference_transform(pose, calibration, reference)
                expected = np.asarray(calibration["T_camera_cv_from_camera_ue"]) @ np.linalg.inv(pose["T_world_camera"]) @ np.asarray(pose["T_world_lidar"]) @ [*raw, 1]
                np.testing.assert_allclose(np.linalg.inv(camera) @ [*point, 1], expected, atol=1e-5)

    def test_nested_conversion_scene_boundary_and_map_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/"raw"/"Town05_Opt_clear"/"run_0001"
            for name in ("rgb", "lidar", "pose", "labels"):
                (run/name).mkdir(parents=True)
            pose, raw, label, calibration = self.fixture(0)
            (run/"calibration.json").write_text(json.dumps(calibration))
            (run/"metadata.json").write_text(json.dumps(dict(actual_num_frames=220, recording_complete=True,
                route_map="Town05_Opt", run_id="run_0001", weather="clear")))
            (run/"scenes.json").write_text(json.dumps({"scenes": [
                dict(scene_id="scene_0000", start_frame=0, end_frame_exclusive=200),
                dict(scene_id="scene_0001", start_frame=200, end_frame_exclusive=220)]}))
            for i in range(220):
                stem = "%06d" % i
                pose, raw, label, _ = self.fixture(i)
                (run/"pose"/(stem+".json")).write_text(json.dumps(pose))
                (run/"labels"/(stem+".json")).write_text(json.dumps(label))
                np.array([[*raw, .5]], dtype=np.float32).tofile(run/"lidar"/(stem+".bin"))
                Image.new("RGB", (8, 8)).save(run/"rgb"/(stem+".png"))
            split = root/"split.yaml"
            split.write_text(json.dumps(dict(group_by="map", splits=dict(train=["Town05_Opt"], val=[], test=[]))))
            args = build_argument_parser().parse_args(["--root-path", str(root/"raw"), "--out-dir", str(root/"out"),
                                                       "--split-file", str(split), "--visualize-samples", "0"])
            report = convert_dataset(args)
            self.assertEqual(report["splits"]["train"]["samples"], 40)
            with (root/"out"/"uavdataset_infos_train.pkl").open("rb") as stream:
                infos = pickle.load(stream)["infos"]
            self.assertEqual(infos[0]["token"], "Town05_Opt_clear/run_0001/000010")
            self.assertEqual([r["frame_index"] for r in infos if r["frame_index"] >= 200], [210, 215])
            for info in infos:
                self.assertEqual(len(info["sweeps"]), 9)
                for sweep in info["sweeps"]:
                    _, frame = parse_scene_frame(sweep["sample_data_token"])
                    self.assertEqual(frame//200, info["frame_index"]//200)

    def test_spatial_split_shared_across_weather_and_route_versions(self):
        plan = SplitPlan(dict(spatial_maps={"Town05_Opt": dict(default_split="train", buffer_m=10,
            regions=[dict(split="val", bounds_xy=[-100, -100, 100, 100])])}))
        infos = []
        for run in ("Town05_Opt_clear/run_0001", "Town05_Opt_rain/run_0002"):
            for frame, x in enumerate((20, 95, 200)):
                infos.append(dict(scene_name=run, frame_index=frame, token=run+"/"+str(frame),
                                  location="Town05_Opt", world_xy_m=[x, 0], sweeps=[]))
        validate_scene_coverage(plan, infos)
        out, safety = rebuild(plan, infos, 0, 0, False)
        self.assertEqual(len(out["val"]), 2)
        self.assertEqual(len(out["train"]), 2)
        self.assertTrue(all(info["world_xy_m"][0] != 95 for rows in out.values() for info in rows))


if __name__ == "__main__":
    unittest.main()
