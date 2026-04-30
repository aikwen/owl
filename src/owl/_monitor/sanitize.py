from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """将对象转换为 JSON 可序列化格式。

    Args:
        value: 待转换对象。

    Returns:
        JSON 可序列化对象。
    """

    # JSON 原生支持的基础类型，直接返回。
    #
    # 原始值:
    #   None / str / int / float / bool
    #
    # 转换后:
    #   None / str / int / float / bool
    if value is None or isinstance(value, str | int | float | bool):
        return value

    # Enum 通常用于配置项或状态值。
    #
    # 原始值:
    #   MonitorTransport.HTTP
    #
    # 转换后:
    #   "http"
    if isinstance(value, Enum):
        return value.value

    # Path 不能被 json.dumps 直接序列化。
    #
    # 原始值:
    #   Path("/tmp/owl")
    #
    # 转换后:
    #   "/tmp/owl"
    if isinstance(value, Path):
        return str(value)

    # dict 递归转换 key 和 value。
    #
    # 原始值:
    #   {
    #       "loss": torch.tensor(0.1),
    #       Path("ckpt"): {"epoch": 1},
    #   }
    #
    # 转换后:
    #   {
    #       "loss": 0.1,
    #       "ckpt": {"epoch": 1},
    #   }
    #
    # 注意:
    #   JSON object 的 key 必须是字符串，所以这里会把 key 转成 str。
    if isinstance(value, dict):
        return {
            str(to_jsonable(k)): to_jsonable(v)
            for k, v in value.items()
        }

    # list / tuple / set 递归转换为 list。
    #
    # 原始值:
    #   (torch.tensor(0.1), Path("a.txt"))
    #
    # 转换后:
    #   [0.1, "a.txt"]
    #
    # 注意:
    #   set 本身无序，转换后的 list 不保证顺序。
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]

    # torch.Tensor 分两类处理。
    #
    # 原始值:
    #   torch.tensor(0.123)
    #
    # 转换后:
    #   0.123
    #
    # 原始值:
    #   torch.randn(2, 3, 512, 512, device="cuda:0")
    #
    # 转换后:
    #   {
    #       "type": "Tensor",
    #       "shape": [2, 3, 512, 512],
    #       "dtype": "torch.float32",
    #       "device": "cuda:0",
    #   }
    #
    # 注意:
    #   非标量 Tensor 不展开为 list，避免把大张量写进监控流。
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()

            return {
                "type": "Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    except Exception:
        pass

    # numpy 标量和数组分两类处理。
    #
    # 原始值:
    #   np.float32(0.123)
    #
    # 转换后:
    #   0.123
    #
    # 原始值:
    #   np.zeros((2, 3))
    #
    # 转换后:
    #   {
    #       "type": "ndarray",
    #       "shape": [2, 3],
    #       "dtype": "float64",
    #   }
    #
    # 注意:
    #   非标量 ndarray 不展开为 list，避免输出过大的监控数据。
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()

            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    except Exception:
        pass

    # 兜底转换。
    #
    # 原始值:
    #   任意无法识别的对象
    #
    # 转换后:
    #   str(value)
    #
    # 注意:
    #   monitor 不应该因为某个额外指标不可序列化而影响训练。
    return str(value)