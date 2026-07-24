from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_batch_to_device(
    batch: Any,
    device: torch.device | str,
    *,
    non_blocking: bool = True,
) -> Any:
    """递归地将 batch 中的 Tensor 移动到指定 device。

    该函数只处理数据搬运，不关心 batch 的具体字段名。

    支持的结构包括：
        - torch.Tensor
        - dict / Mapping
        - list
        - tuple

    其他对象会原样返回，例如：
        - str
        - int
        - float
        - bool
        - None

    Args:
        batch: DataLoader 返回的 batch，通常是 dict[str, Any]。
        device: 目标设备，例如 "cuda"、"cuda:0" 或 torch.device("cpu")。
        non_blocking: 是否启用非阻塞拷贝。

    Returns:
        移动 Tensor 后的新 batch。
    """
    target_device = torch.device(device)
    return _move_to_device(
        value=batch,
        device=target_device,
        non_blocking=non_blocking,
    )


def _move_to_device(
    value: Any,
    device: torch.device,
    *,
    non_blocking: bool,
) -> Any:
    """递归移动单个对象中的 Tensor。"""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, Mapping):
        return {
            key: _move_to_device(
                value=item,
                device=device,
                non_blocking=non_blocking,
            )
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _move_to_device(
                value=item,
                device=device,
                non_blocking=non_blocking,
            )
            for item in value
        )

    if isinstance(value, list):
        return [
            _move_to_device(
                value=item,
                device=device,
                non_blocking=non_blocking,
            )
            for item in value
        ]

    return value