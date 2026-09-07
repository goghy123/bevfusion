"""为 UAV BEVFusion 生成兼容的预训练权重。

默认行为与当前 tools/filter_uav_pretrained.py 保持一致：
- 删除 nuScenes 十类检测头；
- 删除需要由 UAV 配置重新生成的相机几何固定量；
- 删除因相机深度 bin 数变化而不兼容的 depthnet 最后一层；
- 默认保留融合器。

新增能力：
- --drop-fuser：删除整个融合器权重，使融合器从零初始化。
- --keep-depthnet-final：保留 depthnet.6；仅当目标配置的 dbound
  与源 checkpoint 完全兼容时使用，例如把 UAV dbound 恢复为
  [1.0, 60.0, 0.5]，并确认输出通道数与源模型一致。
"""

import argparse
from pathlib import Path

import torch


def normalized_key(key):
    return str(key).replace("module.", "", 1)


def removal_reason(key, drop_fuser=False, keep_depthnet_final=False):
    """返回某个源权重需要被删除的原因；None 表示保留。"""
    normalized = normalized_key(key)

    # UAV 从 nuScenes 的 10 类改成 4 类。
    # 当前项目采用“整个目标检测头重新初始化”的保守策略。
    if normalized.startswith("heads.object.") or ".heads.object." in normalized:
        return "10-class detection head"

    # 【新增】D=7 时 ConvFuser 的输入由 336 变为 976，
    # 第一层卷积尺寸不再兼容；如果希望融合器从零训练，也使用此开关。
    if drop_fuser and normalized.startswith("fuser."):
        return "fusion module reinitialized"

    vtransform_prefix = "encoders.camera.vtransform."
    if normalized.startswith(vtransform_prefix):
        suffix = normalized[len(vtransform_prefix):]

        # 这些量由 image_size / xbound / ybound / zbound / dbound 共同决定，
        # UAV 配置发生变化后必须由新模型重新生成，不能直接复用 checkpoint。
        if suffix in ("dx", "bx", "nx", "frustum"):
            return "camera geometry tensor"

        # 当前 UAV 默认 dbound=[1,70,0.5]，源 nuScenes 为 [1,60,0.5]，
        # 深度 bin 数 D 不同，所以 depthnet 最后一层输出尺寸不同。
        # 【新增】如果你把目标配置恢复到与源 checkpoint 相同的 dbound，
        # 可以传入 --keep-depthnet-final 来保留这一层。
        if suffix.startswith("depthnet.6.") and not keep_depthnet_final:
            return "depth-bin prediction layer"

    return None


def is_object_head_key(key):
    # 保留这个辅助函数，避免已有用户脚本失效。
    normalized = normalized_key(key)
    return normalized.startswith("heads.object.") or ".heads.object." in normalized


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a nuScenes BEVFusion checkpoint for UAV training, "
            "with optional fuser reinitialization."
        )
    )
    parser.add_argument("input", help="Original nuScenes BEVFusion checkpoint")
    parser.add_argument("output", help="Filtered checkpoint for UAV fine-tuning")

    # 【新增】D=7 或希望做 D=2 从零融合器训练时使用。
    parser.add_argument(
        "--drop-fuser",
        action="store_true",
        help="Remove all fuser.* tensors so ConvFuser is initialized from scratch.",
    )

    # 【新增】仅当目标 dbound 与源 checkpoint 的深度 bin 配置完全一致时使用。
    parser.add_argument(
        "--keep-depthnet-final",
        action="store_true",
        help=(
            "Keep encoders.camera.vtransform.depthnet.6.*. "
            "Use only when target depth-bin count matches the source checkpoint."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    checkpoint = torch.load(str(input_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    kept = {}
    removed = []
    reasons = {}

    for key, value in state_dict.items():
        reason = removal_reason(
            key,
            drop_fuser=args.drop_fuser,
            keep_depthnet_final=args.keep_depthnet_final,
        )
        if reason is not None:
            removed.append(key)
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            kept[key] = value

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = kept
        checkpoint.pop("optimizer", None)
        checkpoint.pop("optimizer_states", None)

        if isinstance(checkpoint.get("meta"), dict):
            checkpoint["meta"].pop("CLASSES", None)

        output_checkpoint = checkpoint
    else:
        output_checkpoint = kept

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, str(output_path))

    print("Input parameters : {}".format(len(state_dict)))
    print("Kept parameters  : {}".format(len(kept)))
    print("Removed tensors  : {}".format(len(removed)))
    for reason, count in sorted(reasons.items()):
        print("  {:>4d}  {}".format(count, reason))
    print("Saved to         : {}".format(output_path))


if __name__ == "__main__":
    main()
