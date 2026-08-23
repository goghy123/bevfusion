# CARLA UAV 数据集接入 BEVFusion

本实现用于 `collect_uavdataset.py` v3.3 生成的数据。任务定义为：使用一个俯视 RGB 相机和一个 128 线俯视 LiDAR，检测同时满足录制器 LiDAR 与 RGB 可见性门槛的 `car / van / truck / bus` 三维目标。

## 1. 原始数据要求

传给转换器的根目录下可以包含任意数量、任意名称的 scene。转换器不依赖 `scene_` 前缀，而是识别以下完整结构：

```text
raw_dataset/
├── scene_xxx/
│   ├── calibration.json
│   ├── metadata.json
│   ├── rgb/000000.png
│   ├── lidar/000000.bin
│   ├── pose/000000.json
│   └── labels/000000.json
└── another_scene/
    └── ...
```

正常转换会严格检查：

- 四个逐帧目录的编号完全一致；
- 帧编号连续；
- 文件数等于 `metadata.json.actual_num_frames`；
- 点云严格为 `N×4 float32`；
- LiDAR 保持水平安装；
- 转换后的矩阵、点云均为有限数值。

`--allow-partial-scenes` 只用于首帧样例或调试，不应对正式训练数据使用。

## 2. 坐标和点云转换

原始 CARLA/UE 数据是左手系：`x 前、y 右、z 上`。训练数据使用逐帧地面参考右手系：

```text
x：UAV/LiDAR 前方
y：左方
z：上方
原点：LiDAR 正下方的路线地面高度
```

转换器执行：

- 点云和框的 `y` 取反；
- 使用每帧 UAV、LiDAR 位姿和录制高度，把地面由约 `z=-50 m` 移至 `z≈0`；
- 框尺寸由 CARLA `[length,width,height]` 转成模型的 `[width,length,height]`；
- 由完整方向矩阵计算 BEVFusion yaw；
- 把 4 维点扩展为 `[x,y,z,intensity,time_lag]` 五维点；
- 把 CARLA 强度默认乘以 255，以接近 nuScenes 预训练输入范围；
- 由相同 actor 的相邻世界坐标标注估计速度；静态或无法估计时为 0；
- 为每个关键帧建立只来自同一数据划分区段的历史 sweeps。

三维框在 info 文件中保存几何中心，`UAVDataset` 加载时会正确转成该 BEVFusion 分支要求的底面中心。

## 3. 运行转换

在 BEVFusion 仓库根目录执行：

```bash
python tools/create_uavdataset.py \
  --root-path /absolute/path/to/carla_project/dataset \
  --out-dir data/uavdataset \
  --split-ratios 0.70 0.15 0.15 \
  --keyframe-stride 5 \
  --max-sweeps 9 \
  --intensity-scale 255 \
  --image-mode reference \
  --visualize-samples 6
```

默认 `image-mode=reference` 不复制体积较大的 PNG，而是在 info 文件中保存相对于输出目录的路径。原始数据目录在训练期间必须保留。其他选项：

- `copy`：复制图像，最便于移动，空间占用最大；
- `hardlink`：同一文件系统内不额外占空间；
- `symlink`：创建相对符号链接。

重复生成时显式增加 `--overwrite`。转换器不会递归删除输出目录。

输出结构：

```text
data/uavdataset/
├── points/<scene>/<frame>.bin
├── uavdataset_infos_train.pkl
├── uavdataset_infos_val.pkl
├── uavdataset_infos_test.pkl
├── split_manifest.json
├── conversion_report.json
└── validation/*_bev.png
```

请先检查：

1. `conversion_report.json` 中没有 partial scene；
2. 每个正式 scene 的 `projection_audit.in_image_ratio` 合理；
3. train、val、test 都有样本；
4. 四类计数符合预期；
5. `validation` 中的框与车辆点簇对齐。

## 4. 数据划分定义

每个 scene 分别按时间顺序划成 `70% train / 15% val / 15% test`。原始 10 Hz 帧每 5 帧选一个关键帧，即约 2 Hz；中间帧仍用于 9 个历史 LiDAR sweeps。

划分边界不共享 sweep。每个区段开头会跳过尚未积累满 9 个历史帧的关键帧。具体帧编号完整记录在 `split_manifest.json`，因此划分可复现且不会出现相邻帧跨集合泄漏。

## 5. 准备预训练权重

配置采用四类新检测头，同时图像尺寸、深度 bins 和 BEV 范围不同于 nuScenes。先过滤官方或原 nuScenes BEVFusion 检测 checkpoint：

```bash
python tools/filter_uav_pretrained.py \
  pretrained/bevfusion-det.pth \
  pretrained/bevfusion-uav-init.pth
```

该工具保留兼容的：

- Swin 相机骨干与 FPN；
- LiDAR 稀疏卷积骨干；
- 融合器；
- SECOND 解码骨干与 FPN；
- Depth-LSS 中尺寸兼容的卷积层。

它会删除：

- nuScenes 十类 TransFusion 检测头；
- 必须由 UAV 配置重新生成的相机 frustum、BEV 几何张量；
- 深度 bin 数变化导致形状不兼容的最后预测卷积。

因此不要把原始十类 checkpoint 直接传给本配置。

## 6. 训练

配置文件：

```text
configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml
```

单 GPU 示例：

```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --run-dir runs/uavdataset-bevfusion \
  --load_from pretrained/bevfusion-uav-init.pth
```

同时屏幕显示 + 写入日志:
```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --run-dir runs/uavdataset-bevfusion \
  --load_from pretrained/bevfusion-uav-init.pth 2>&1 | tee train_log.txt
```

当前默认值：

- 输入图像 `320×576`，像素量与 nuScenes 的 `256×704` 接近；
- 单相机，保留接近完整的 16:9 俯视画面；
- 相机深度范围 `1–70 m`；
- 点云范围 `[-51.2,-51.2,-5, 51.2,51.2,3]`；
- voxel size `[0.1,0.1,0.2]`；
- batch size 1；
- 20 epochs；
- 不加载 nuScenes 地图、不做 ObjectPaste、不做 32 线降采样。

如果显存不足，优先减小 `image_size` 和 `max_voxels`，不要直接缩小 z 范围或改变点云维数。

## 7. 测试与评估

```bash
python tools/test.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  runs/uavdataset-bevfusion/latest.pth \
  --eval bbox
```

`UAVDataset` 不调用 nuScenes devkit，也不伪造 NDS。默认报告：

- 每类 BEV AP@0.50；
- 每类 3D AP@0.50；
- 四类中验证集实际存在类别的 BEV/3D mAP；
- score≥0.1 时的逐类 precision 和 recall。

可覆盖阈值：

```bash
python tools/test.py <config> <checkpoint> --eval bbox \
  --eval-options iou_threshold=0.7 score_threshold=0.2
```

## 8. 任务边界

- 当前标注由录制器按 LiDAR 点数和 RGB 可见像素共同硬筛选；转换器不会恢复被录制器删除的目标。
- 该配置学习的是“两种传感器共同可见目标”，与仅按 LiDAR 可见性保留全部三维目标的任务不同。
- 一张俯视相机可直接作为视图数 `N=1` 输入 BEVFusion，不复制成六视图。
- 当前不支持 BEV 地图分割，因为数据中没有 nuScenes map expansion 对应的地图层标注。
