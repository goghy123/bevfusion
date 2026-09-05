# CARLA UAV 数据集接入 BEVFusion

任务定义为：使用一个俯视 RGB 相机和一个 128 线俯视 LiDAR，检测同时满足录制器 LiDAR 与 RGB 可见性门槛的 `car / van / truck / bus` 三维目标。适用数据集包含多个不同 CARLA Town。通过 YAML 文件显式指定完整 scene/Town 的归属。

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

原始数据转换使用：

```text
configs/uavdataset/splits/town_split_v1.yaml
```

该文件只负责保证每个完整 scene/Town 在基础转换时唯一归属于 train / val / test，从而生成完整、无重复的 base info PKL。

基础转换生成：

data/uavdataset/
├── uavdataset_infos_train.pkl
├── uavdataset_infos_val.pkl
├── uavdataset_infos_test.pkl
├── split_manifest.json
└── conversion_report.json

最终训练划分由：

```TXET
configs/uavdataset/splits/uav_balanced_split.yaml
```

控制。

当前设计原则为：

- Town05 包含当前数据集中最明显的低道路 / low-Z 分布，因此使用若干连续 frame range 分配到 Train、Val 和 Test；

- 精确的 Town 和 frame 分配始终以 uav_balanced_split.yaml 为唯一 source of truth，不在文档中重复维护 frame 编号。

原对数据集的 split 中，Town05 的低道路目标主要只出现在 Val，而 Train 中对应的低 Z GT 极少，导致明显的训练/验证高度分布偏移。

## 4. 运行转换

在 BEVFusion 仓库根目录执行基础转换：

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

审计：
每次重新录制数据、修复标注或者重新转换 PKL 后，都应在最终训练前重新执行：

```bash
python tools/analyze_uav_dataset.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --raw-root data/uavdataset/raw \
  --converted-root data/uavdataset \
  --source-pattern 'uavdataset_infos_{split}.pkl' \
  --splits train val test \
  --sweeps 0 3 6 9 \
  --train-augment-repeats 2 \
  --run-name uav_dataset_analysis \
  --output-root runs
```

主要输出：

```text
runs/uav_dataset_analysis/
├── analysis.json
└── recommended_changes.yaml
```

重分：正式写文件前先执行：

```BASH
python tools/rebalance_uav_splits.py \
  --dataset-root data/uavdataset \
  --source-pattern 'uavdataset_infos_{split}.pkl' \
  --plan configs/uavdataset/splits/uav_balanced_split.yaml \
  --output-dir data/uavdataset/balanced \
  --sweeps 0 3 6 9 \
  --guard-keyframes 2 \
  --dry-run \
  --overwrite
```

要求至少确认：

`short_sweep_dropped_total = 0`
`cross_split_sweeps_removed = 0`

通过后，去掉 --dry-run：

```bash
python tools/rebalance_uav_splits.py \
  --dataset-root data/uavdataset \
  --source-pattern 'uavdataset_infos_{split}.pkl' \
  --plan configs/uavdataset/splits/uav_balanced_split.yaml \
  --output-dir data/uavdataset/balanced \
  --sweeps 0 3 6 9 \
  --guard-keyframes 2 \
  --overwrite
```

输出：

```text
data/uavdataset/balanced/
├── uavdataset_infos_train_s0.pkl
├── uavdataset_infos_train_s3.pkl
├── uavdataset_infos_train_s6.pkl
├── uavdataset_infos_train_s9.pkl
├── uavdataset_infos_val_s0.pkl
├── uavdataset_infos_val_s3.pkl
├── uavdataset_infos_val_s6.pkl
├── uavdataset_infos_val_s9.pkl
├── uavdataset_infos_test_s0.pkl
├── uavdataset_infos_test_s3.pkl
├── uavdataset_infos_test_s6.pkl
├── uavdataset_infos_test_s9.pkl
└── split_audit.json
```

## 5. sweep划分

### 数据转换

将数据集按sweep=3、6、9划分，使用如下指令读取已有 PKL，截取最近 N 个 sweep：

```bash
python tools/data_converter/derive_uav_sweep_infos.py \
  --dataset-root data/uavdataset \
  --sweeps 3 6 9
```

如果已经存在，需要覆盖：

```bash
python tools/data_converter/derive_uav_sweep_infos.py \
  --dataset-root data/uavdataset \
  --sweeps 3 6 9 \
  --overwrite
```

### 配置修改

同时需要配合修改config

- sweeps=6

`configs/uavdataset/default.yaml`的修改如下：

```yaml
max_sweeps: 6
```

还有

```yaml
data:
  train:
    ann_file: ${dataset_root + "uavdataset_infos_train_s6.pkl"}

  val:
    ann_file: ${dataset_root + "uavdataset_infos_val_s6.pkl"}

  test:
    ann_file: ${dataset_root + "uavdataset_infos_test_s6.pkl"}
```

以及`configs/uavdataset/det/transfusion/secfpn/camera+lidar/default.yaml`：

```yaml
max_voxels: [150000, 155000]
```

- sweeps=3、9

`configs/uavdataset/default.yaml`的修改同上，将6的部分替换为sweeps的数字；

`configs/uavdataset/det/transfusion/secfpn/camera+lidar/default.yaml`的修改：
对于`sweeps=3`:`max_voxels: [95000, 100000]`
对于`sweeps=9`:`max_voxels: [200000, 205000]`

## 6. 准备预训练权重

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

## 7. 训练

### 单 GPU 训练

```bash
torchpack dist-run -np 1 python tools/train.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --run-dir runs/uavdataset-bevfusion \
  --load_from pretrained/bevfusion-uav-init.pth \
  --data.workers_per_gpu 4 \
  2>&1 | tee runs/uavdataset-bevfusion/train_log.txt
```

### 不同sweep训练

对于`sweeps=6`，先修改对应的config，再执行以下指令：

```bash
mkdir -p runs/uavdataset-bevfusion-s6

torchpack dist-run -np 1 python tools/train.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --run-dir runs/uavdataset-bevfusion-s6 \
  --load_from pretrained/bevfusion-uav-init.pth \
  --data.workers_per_gpu 4
  2>&1 | tee runs/uavdataset-bevfusion-s6/train_log.txt
```

对于`sweeps=3、9`的情况，修改对应的config以及指令中的文件后缀即可。

## 8. 测试与评估

### 评估

```bash
python tools/test.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  runs/uavdataset-bevfusion/latest.pth \
  --out runs/uavdataset-bevfusion/test_results.pkl \
  --eval bbox \
  2>&1 | tee runs/uavdataset-bevfusion/test_log.txt
```

可覆盖阈值：

```bash
python tools/test.py <config> <checkpoint> --eval bbox \
  --eval-options iou_threshold=0.7 score_threshold=0.2
```

### 不同sweep评估

对于`sweeps=6`，先修改对应的config，再执行以下指令：

```bash
python tools/test.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  runs/uavdataset-bevfusion-s6/latest.pth \
  --out runs/uavdataset-bevfusion-s6/test_results.pkl \
  --eval bbox \
  2>&1 | tee runs/uavdataset-bevfusion-s6/test_log.txt
```

对于`sweeps=3、9`的情况，修改对应的config以及指令中的文件后缀即可。

### 保存结果评估

保存测试json结果，用于可视化：
```BASH
python tools/test_uav_predictions.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  runs/uavdataset-bevfusion-s9/latest.pth \
  --out-dir runs/uavdataset-bevfusion-s9/test_predictions
```