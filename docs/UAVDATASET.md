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

## 8. 训练

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
  2>&1 | tee runs/uavdataset-bevfusion/train_log.txt
```

对于`sweeps=3、9`的情况，修改对应的config以及指令中的文件后缀即可。

## 9. 测试与评估

### 评估

```bash
python tools/test.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  runs/uavdataset-bevfusion/latest.pth \
  --out runs/uavdataset-bevfusion/test_results.pkl \
  --eval bbox \
  2>&1 | tee runs/uavdataset-bevfusion/test_log.txt
```

默认报告：

- 每类 BEV AP@0.50；
- 每类 3D AP@0.50；
- 四类中验证集实际存在类别的 BEV/3D mAP；
- score≥0.1 时的逐类 precision 和 recall。

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
  2>&1 | tee runs/uavdataset-bevfusion/test_log.txt
```

对于`sweeps=3、9`的情况，修改对应的config以及指令中的文件后缀即可。

### 保存结果评估

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

# UAV Multi-Sweep Voxel 容量统计

## 目的

`profile_uav_sweeps_voxels.py` 用于统计不同历史 LiDAR sweep 数量下，进入 voxelizer 前实际产生的非空 voxel 数量，从而确定合理的：

* `max_sweeps`
* `model.encoders.lidar.voxelize.max_voxels`

主要用于避免 multi-sweep 点云已经加载，但由于 `max_voxels` 过小而在 voxelization 阶段大量截断。


## 使用方法

在 BEVFusion 项目根目录执行：

```bash
python tools/profile_uav_sweeps_voxels.py \
  configs/uavdataset/det/transfusion/secfpn/camera+lidar/swint_v0p1/convfuser.yaml \
  --converted-root data/uavdataset \
  --sweeps 3 6 9 \
  --max-samples-per-split 0 \
  --train-augment-repeats 2 \
  --output uav_sweep_voxel_profile.json
```

参数说明：

```text
--sweeps 3 6 9
    分别统计3、6、9个历史sweep。
    当前帧不计入该数字，因此3 sweeps表示当前帧+3历史帧。

--max-samples-per-split 0
    统计整个train/val/test数据集，不限制采样数量。

--train-augment-repeats 2
    对训练集重复模拟两次3D数据增强，以覆盖增强造成的voxel数量变化。

--output
    指定JSON统计结果文件。
```

数据集更新并重新生成 `uavdataset_infos_*.pkl` 后，应重新运行本脚本。

## 统计方法

每个 sweep 配置按照：

```text
当前LiDAR
+ 最近N个历史LiDAR
→ 历史帧坐标变换
→ Frustum Filter
→ 训练集3D Augmentation
→ Point Cloud Range Filter
→ Voxelization
→ 统计截断前非空voxel数量
```

主要输出：

```text
p50
p95
p99
p99.5
max
```

并对不同 `max_voxels` 候选值统计：

```text
sample_overflow_rate
mean_candidate_voxel_drop_ratio
p99_candidate_voxel_drop_ratio
```

## 参数确定方法

一般使用：

```text
Train max_voxels：
参考Train的p99～max。

Test max_voxels：
参考Val/Test中较大的p99～max。
```

如果优先控制显存和速度，可接近 `p95/p99`。

如果希望基本不截断，可取接近 `max` 并向上取整。

当前数据(uavdataset_v1.0)统计得到的建议为：

```yaml
# 3 historical sweeps
max_sweeps: 3
max_voxels: [95000, 100000]

# 6 historical sweeps
max_sweeps: 6
max_voxels: [150000, 155000]

# 9 historical sweeps
max_sweeps: 9
max_voxels: [200000, 205000]
```

最终通过 3/6/9 sweep 独立训练，比较：

```text
3D mAP
BEV mAP
GPU显存
训练速度
推理延迟
```

后确定最终部署配置。