# CARLA UAV 数据集接入 BEVFusion

本实现用于 `collect_uavdataset.py` v3.3 生成的数据。任务定义为：使用一个俯视 RGB 相机和一个 128 线俯视 LiDAR，检测同时满足录制器 LiDAR 与 RGB 可见性门槛的 `car / van / truck / bus` 三维目标。

当前适用数据集包含多个不同 CARLA Town。通过 YAML 文件显式指定完整 scene/Town 的归属。

## 1. 原始数据要求

传给转换器的根目录下可以包含任意数量、任意名称的 scene。随后转换器识别以下完整结构：

```text
raw_dataset/
├── Town01_Opt/
│   ├── calibration.json
│   ├── metadata.json
│   ├── rgb/000000.png
│   ├── lidar/000000.bin
│   ├── pose/000000.json
│   └── labels/000000.json
├── Town02_Opt/
│   └── ...
└── another_scene/
    └── ...
```

正常转换会严格检查：

- 四个逐帧目录的编号完全一致；
- 帧编号连续；
- 文件数等于 `metadata.json.actual_num_frames`；
- 点云严格为 `N×4 float32`；
- LiDAR 保持水平安装；
- 转换后的矩阵、点云均为有限数值；
- YAML 中的每个 scene 名称都真实存在；
- 每个发现的 scene 必须且只能分配到 `train / val / test` 其中一个；
- `train / val / test` 三个列表都必须存在且非空。

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
- 为每个关键帧建立只来自**同一 scene/Town 更早原始帧**的历史 sweeps，绝不跨 scene 或跨 split 使用点云。

三维框在 info 文件中保存几何中心，`UAVDataset` 加载时会正确转成该 BEVFusion 分支要求的底面中心。

## 3. 数据划分 YAML

数据集划分由 `--split-file` 指定的 YAML 文件控制。当前推荐配置位于：

```text
configs/uavdataset/splits/town_split_v1.yaml
```

当前扩充数据集采用：

```yaml
version: 1

splits:
  train:
    - Town01_Opt
    - Town02_Opt
    - Town03_Opt
    - Town04_Opt
    - Town06_Opt

  val:
    - Town05_Opt

  test:
    - Town07_Opt
    - Town10HD_Opt
```

## 4. 运行转换

在 BEVFusion 仓库根目录执行：

```bash
python tools/create_uavdataset.py \
  --root-path data/uavdataset/raw \
  --out-dir data/uavdataset \
  --split-file configs/uavdataset/splits/town_split_v1.yaml \
  --keyframe-stride 5 \
  --max-sweeps 9 \
  --intensity-scale 255 \
  --image-mode reference \
  --visualize-samples 6 \
  --overwrite
```

默认 `image-mode=reference` 不复制体积较大的 PNG，而是在 info 文件中保存相对于输出目录的路径。原始数据目录在训练期间必须保留。其他选项：

- `copy`：复制图像，最便于移动，空间占用最大；
- `hardlink`：同一文件系统内不额外占空间；
- `symlink`：创建相对符号链接。

重复生成时显式增加 `--overwrite`。转换器不会递归删除输出目录。

输出结构如下：

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
4. 四类计数符合预期，尤其确认 val/test 没有类别完全缺失；
5. `validation` 中的框与车辆点簇对齐；
6. `split_manifest.json` 中每个 scene 的 `assigned_split` 与 YAML 一致。

## 5. 关键帧与 sweeps

原始数据为 10 Hz。使用：

```text
--keyframe-stride 5
```

时，每 5 个原始帧选一个候选关键帧，即约 2 Hz。中间原始帧不会作为独立训练样本，但仍用于构建历史 LiDAR sweeps。

使用：

```text
--max-sweeps 9
```

时，每个关键帧最多使用同一 scene 中更早的 9 个原始 LiDAR 帧。sweeps 按时间从近到远写入，并包含相对于当前关键帧的位姿变换和 `time_lag`。

由于当前划分单位是完整 scene/Town：

- sweep 不需要在 scene 内人为的 train/val/test 时间边界重新截断；
- sweep 永远不会跨 Town；
- sweep 永远不会跨 train/val/test；
- 每个 Town 开头历史帧不足的候选关键帧会被跳过；
- 后续关键帧可以连续使用该 Town 自身的历史原始帧。

例如连续帧从 0 开始、`keyframe_stride=5`、`max_sweeps=9` 时，候选关键帧为 `0, 5, 10, 15, ...`。其中前两个候选帧历史不足，`frame 10` 开始可以获得 9 个历史 sweeps。

具体候选关键帧、保留关键帧以及因历史不足被跳过的关键帧都会记录在 `split_manifest.json`。

## 6. 当前扩充数据集的转换结果

使用 `town_split_v1.yaml`、`keyframe_stride=5`、`max_sweeps=9` 转换当前扩充数据集后，得到：

| split | Town | samples | empty samples | car | van | truck | bus |
|---|---|---:|---:|---:|---:|---:|---:|
| train | Town01 / Town02 / Town03 / Town04 / Town06 | 4102 | 873 | 22589 | 4168 | 3404 | 491 |
| val | Town05 | 976 | 104 | 5872 | 951 | 1283 | 117 |
| test | Town07 / Town10HD | 809 | 91 | 4077 | 539 | 773 | 129 |

总计：

```text
samples = 5887
train / val / test = 4102 / 976 / 809
```

四个目标类别在 train、val、test 中均有样本，因此当前划分可用于后续训练和跨地图泛化评估。

`conversion_report.json` 中的 `empty_samples` 表示生成的 info 中没有有效 GT 的关键帧数量。训练配置当前对 train 使用 `filter_empty_gt=True`，因此转换文件中的 train `samples` 数量不一定等于训练时最终参与采样的数据集长度；这与地图划分逻辑本身无关。

## 7. 准备预训练权重

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

## 8. 训练

配置文件：

```text
configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml
```

单 GPU 示例：

```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --run-dir runs/uavdataset-bevfusion \
  --load_from pretrained/bevfusion-uav-init.pth \
  --data.workers_per_gpu 4 \
  2>&1 | tee runs/uavdataset-bevfusion/train_log.txt
```

## 9. 测试与评估

```bash
python tools/test.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  runs/uavdataset-bevfusion/latest.pth \
  --out runs/uavdataset-bevfusion/test_results.pkl \
  --eval bbox \
  2>&1 | tee runs/uavdataset-bevfusion/test_log.txt
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

保存测试json结果，用于可视化：
```BASH
python tools/test_uav_predictions.py
```

结果目录应类似：

```text
test_predictions/
├─ manifest.json
├─ metrics.json
├─ summary.json
├─ timing.csv
└─ predictions/
   ├─ Town07_Opt/
   │  ├─ 000009.json
   │  ├─ 000012.json
   │  └─ ...
   └─ ...
```

## 10. 任务边界

- 当前标注由录制器按 LiDAR 点数和 RGB 可见像素共同硬筛选；转换器不会恢复被录制器删除的目标。
- 该配置学习的是“两种传感器共同可见目标”，与仅按 LiDAR 可见性保留全部三维目标的任务不同。
- 一张俯视相机可直接作为视图数 `N=1` 输入 BEVFusion，不复制成六视图。
- 当前不支持 BEV 地图分割，因为数据中没有 nuScenes map expansion 对应的地图层标注。
