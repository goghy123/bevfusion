# UAV LiDAR-Camera common-FOV patch

## Frustum filter

`LiDARCameraFrustumFilter` is inserted after `LoadPointsFromMultiSweeps` and
before `LoadAnnotations3D` / image augmentation in both train and test
pipelines.

It computes visibility from:

- the per-sample `lidar2image` matrix;
- the actual loaded image width/height.

Therefore camera FOV, resolution, altitude and mounting pose are not hard-coded.

Default safeguards:

```yaml
frustum_filter:
  min_depth: 0.05
  max_depth: null
  margin_px: 2.0
  min_points: 1000
  min_keep_ratio: 0.005
  mode: union
  on_threshold_failure: raise
```

If a future camera is intentionally extremely narrow, lower `min_points` or
`min_keep_ratio`; do not change the projection formula.

The filter uses the **raw acquisition FOV**, not a random training crop. GT
visibility was defined from raw RGB/LiDAR; random crop/flip/rotate is training
augmentation and should not silently redefine GT.

## GT-Z audit

Run:

```bash
python tools/analyze_uav_gt_z.py \
  --dataset-root data/uavdataset \
  --z-min -5 --z-max 3 \
  --voxel-z 0.2 \
  --margin 0.5
```

The script reports split/class distributions, boxes partly/fully outside the
configured vertical range, and two candidate Z ranges.

When changing Z, update all coupled settings documented in the accompanying
ChatGPT response, especially `point_cloud_range`, LiDAR sparse/grid Z shape,
and `DepthLSSTransform.zbound`.
