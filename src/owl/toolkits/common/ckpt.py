import pathlib
from typing import TypedDict, Any, Union

import torch


class CheckpointDict(TypedDict):
    """定义保存到 .pth 文件中的权重字典结构。

    Attributes:
        epoch (int): 训练结束时的轮次索引。
        model_state (dict): 模型的 state_dict。
        optimizer_state (dict): 优化器的 state_dict。
        scheduler_state (Optional[dict]): 学习率调度器的 state_dict。
    """
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None


def load_checkpoint(path: Union[str, pathlib.Path], device: torch.device) -> CheckpointDict:
    """加载 Owl 标准格式的权重文件。

    Args:
        path: 权重文件路径。
        device: 目标硬件设备。

    Returns:
        CheckpointDict: 符合 Owl 契约的权重字典，包含 epoch, model_state 等。
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"权重文件不存在: {p}")

    ckpt = torch.load(p, map_location=device, weights_only=False)
    return ckpt


def save_checkpoint(ckpt: CheckpointDict, path: Union[str, pathlib.Path]):
    """将权重字典保存到磁盘。

    Args:
        ckpt (CheckpointDict): 包含模型状态、优化器状态等的强类型字典。
        path: 保存路径。
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, p)
