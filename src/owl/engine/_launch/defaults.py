from __future__ import annotations

import pathlib
from datetime import datetime

from .types import ComponentConfig, ComponentName


def default_work_dir() -> pathlib.Path:
    """生成默认工作目录。

    默认目录格式为::

        ./YYYYMMDD_HHMMSS_mmm

    该目录用于保存日志、checkpoint、可视化结果等运行产物。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return pathlib.Path(f"./{timestamp}")


def default_optimizer() -> tuple[ComponentName, ComponentConfig]:
    """返回 TRAIN 模式默认优化器配置。

    当前默认使用 AdamW。
    """
    return "adamw", {
        "lr": 1e-3,
        "weight_decay": 1e-2,
    }


def default_scheduler() -> tuple[ComponentName, ComponentConfig]:
    """返回 TRAIN 模式默认学习率调度器配置。

    当前默认使用 poly scheduler。
    """
    return "poly", {
        "power": 0.9,
    }


def default_evaluator() -> tuple[ComponentName, ComponentConfig]:
    """返回默认评估器配置。

    TRAIN 和 VALIDATE 模式默认使用该评估器。
    """
    return "default_auc_f1", {
        "threshold": 0.5,
    }


def default_visualizer() -> tuple[ComponentName, ComponentConfig]:
    """返回 VISUALIZE 模式默认可视化器配置。

    当前默认使用 default_mask。
    """
    return "default_mask", {}


def default_visualizer_save_dir(work_dir: pathlib.Path) -> pathlib.Path:
    """返回默认可视化结果保存目录。

    Args:
        work_dir: 当前运行工作目录。

    Returns:
        默认可视化输出目录，即 work_dir / "visual"。
    """
    return work_dir.joinpath("visual")