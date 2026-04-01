# src/owl/toolkit/common/file_io.py
import pathlib
import json
from typing import Any, Union

import torch
from PIL import Image

from .types import CheckpointDict


def load_image(path: Union[str, pathlib.Path]) -> Image.Image:
    """
    读取单张图片。

    Args:
        path: 图片路径

    Returns:
        PIL.Image.Image: 图像对象

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 图像损坏或无法解码
    """
    p = pathlib.Path(path)

    if not p.exists():
        raise FileNotFoundError(f"图像不存在: {p}")

    try:
        img = Image.open(p)
        # 强制加载到内存，避免 Lazy Loading 导致的 "Too many open files" 问题
        img.load()
        return img
    except Exception as e:
        raise RuntimeError(f"图像读取失败: {p.name} -> {e}")

def load_json(path: Union[str, pathlib.Path]) -> Any:
    """
    读取 JSON 文件。
    Args:
        path: 文件路径
    Returns:
        List or Dict: 解析后的 JSON 数据
    Raises:
        RuntimeError: 如果文件读取或解析失败
    """
    p = pathlib.Path(path)
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"无法读取 JSON 文件: {p} -> {e}")


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