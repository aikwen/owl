from __future__ import annotations

import json
import pathlib
from enum import Enum
from typing import Any

from .types import LaunchConfig


def dump_launch_config(config: LaunchConfig) -> pathlib.Path:
    """将规范化后的 launch config 写入 work_dir/config.log。

    Args:
        config: normalize_launch_kwargs() 返回的规范化配置。

    Returns:
        写入的 config.log 路径。
    """
    work_dir = pathlib.Path(config["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir.joinpath("config.log")

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            _to_jsonable(config),
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    return config_path


def _to_jsonable(value: Any) -> Any:
    """将 launch config 中的对象转换为 JSON 可序列化对象。"""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, pathlib.Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(_to_jsonable(k)): _to_jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]

    return str(value)