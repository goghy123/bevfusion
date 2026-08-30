# 运行
``` BASH
bash tools/dist_test.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  models/bevfusion_converted.pth \
  1
```

demo 展示

python projects/BEVFusion/demo/multi_modality_demo.py \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
demo/data/nuscenes/ \
demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
models/bevfusion_converted_view_transform.pth \
--cam-type all \
--score-thr 0.2 \
--out-dir demo/out



# 修改:
mmdet3d/datasets/nuscenes_dataset.py文件中的v1.0-trainval改成v1.0-mini
用于适配mini数据集。

projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py文件中的checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'改为checkpoint='models/swin_tiny_patch4_window7_224.pth'
用于解决网络问题。

# 评估

```SHELL

torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox

```

## 保存评估结果

如果你想把预测结果也保存下来，推荐这样跑：

```SHELL

mkdir -p outputs/bevfusion_det_val

torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --out outputs/bevfusion_det_val/results.pkl \
  --eval bbox

```

## 使用训练模型评估

```SHELL

mkdir -p outputs/bevfusion_det_val

torchpack dist-run -np 1 python tools/test.py \
  runs/run-52d713f2-c67a560d/configs.yaml \
  runs/run-52d713f2-c67a560d/epoch_8.pth \
  --out outputs/bevfusion_det_val/results.pkl \
  --eval bbox

```

# 训练

```SHELL

torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth

```

保存训练日志,方便以后排查和记录：

```SHELL

mkdir -p outputs/bevfusion_det_train

torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  2>&1 | tee outputs/bevfusion_det_train/train_log.txt

```

## 训练提速

```SHELL
mkdir -p outputs/bevfusion_det_train

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --data.workers_per_gpu 4 \
  --evaluation.interval 6 \
  2>&1 | tee outputs/bevfusion_det_train/train_workers4_eval6_log.txt

```

> 提速注释

这里：

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
防止 CPU 疯狂开多线程，导致系统卡顿、占用过高、反而拖慢 GPU 训练

--data.workers_per_gpu 4
会让 DataLoader 用 4 个 worker 预取和处理数据，通常能提高 GPU 利用率。

--evaluation.interval 6
表示 6 个 epoch (当前总的 epoch 就是6个)结束后再做验证。评估不参与反向传播，不影响训练权重，但能省掉前 5 次验证的时间（2-3H）。

> 不建议马上改 samples_per_gpu=2

> 提速效果查看及改进

先看 200~500 iter 后的日志。如果变成：

data_time: 0.05~0.12
GPU 利用率: 70~90%

就说明有效。

第二档，继续尝试

如果内存还够、系统不卡，可以试：

--data.workers_per_gpu 6

第三档，不一定更快
--data.workers_per_gpu 8

不一定比 4 或 6 更快，因为 worker 太多可能导致 CPU 调度、磁盘读取、内存占用变差。

## 训练效果不理想

在单卡4080super上训练效果不理想，精度 mAP: 0.5798 / NDS: 0.6463 ，远低于官方 68.52 mAP / 71.38 NDS ，可能的原因：没有模拟官方的全局 batch 和 optimizer step 节奏，模型从一个很强的 LiDAR-only 初始化继续训练，结果被过多、高学习率的单样本更新带偏，所以最后反而只到 58/64 左右。

> 解决方案

用单卡 16GB 用梯度累积 8，8 个 micro-batch 累积一次梯度 ≈ 模拟官方全局 batch size 8 ≈ optimizer step 数量接近官方 8 卡训练。

设置保持官方 LR：optimizer.lr = 0.0002 ；改动 warmup ：armup_iters = 4000

建议训练命令如下：

```SHELL
mkdir -p outputs/bevfusion_det_train_accum8

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --optimizer.lr 0.0002 \
  --optimizer_config.type GradientCumulativeOptimizerHook \
  --optimizer_config.cumulative_iters 8 \
  --optimizer_config.grad_clip.max_norm 35 \
  --optimizer_config.grad_clip.norm_type 2 \
  --lr_config.warmup_iters 4000 \
  --data.workers_per_gpu 4 \
  --evaluation.interval 1 \
  --checkpoint_config.interval 1 \
  --checkpoint_config.max_keep_ckpts 6 \
  2>&1 | tee outputs/bevfusion_det_train_accum8/train_accum8_log.txt

```

# 几个注意

## cudasm

在编译以及使用前使用以下指令解决40系显卡cudasm版本不适配问题：

``` BASH 
unset CUDA_VISIBLE_DEVICES

export CUDA_HOME=/usr/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6+PTX"

```
## view_transform 正确加载

在projects/BEVFusion/README.md中，官方提供的权重文件在使用时会有以下报错：

```BASH 
unexpected key in source state_dict: vtransform.*
missing keys in source state_dict: view_transform.*
```

会导致精度下降：

```BASH 
mAP: 0.5174
NDS: 0.4634
```

原因是 checkpoint 里模块名叫**vtransform**，而当前 MMDetection3D 配置/代码里模块名叫：**view_transform**，也就是说权重没有完全对上，view_transform 这一部分可能没正确加载。虽然它仍然跑完了，但评估结果可能不是严格对应原始 BEVFusion 权重。

解决方法：把 checkpoint 里的 vtransform. 批量改成 view_transform.：

``` BASH
cd ~/mmdetection3d

python - <<'PY'
import torch
src = 'models/bevfusion_converted.pth'
dst = 'models/bevfusion_converted_view_transform.pth'

ckpt = torch.load(src, map_location='cpu')

state_dict = ckpt.get('state_dict', ckpt)
new_state_dict = {}

for k, v in state_dict.items():
    if k.startswith('vtransform.'):
        k = k.replace('vtransform.', 'view_transform.', 1)
    new_state_dict[k] = v

if 'state_dict' in ckpt:
    ckpt['state_dict'] = new_state_dict
else:
    ckpt = new_state_dict

torch.save(ckpt, dst)
print(f'Saved to {dst}')
PY

```

然后用新权重再测：

``` BASH
bash tools/dist_test.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  models/bevfusion_converted.pth \
  1
```