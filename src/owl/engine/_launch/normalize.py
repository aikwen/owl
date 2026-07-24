from __future__ import annotations

import pathlib
from typing import Any, cast

from .defaults import (
    default_evaluator,
    default_optimizer,
    default_scheduler,
    default_visualizer,
    default_visualizer_save_dir,
    default_work_dir,
)
from .types import (
    LaunchConfig,
    RawLaunchKwargs,
    TrainLaunchConfig,
    ValidateLaunchConfig,
    VisualizeLaunchConfig,
)
from .validate import validate_raw_launch_kwargs
from ..state import ExecMode


def normalize_launch_kwargs(**kwargs: Any) -> LaunchConfig:
    """规范化 OwlApp.launch() 的原始参数。

    该函数会根据运行模式返回当前任务真正需要的最小配置。

    规则：
        1. TRAIN 返回 TrainLaunchConfig。
        2. VALIDATE 返回 ValidateLaunchConfig。
        3. VISUALIZE 返回 VisualizeLaunchConfig。

    Args:
        **kwargs: OwlApp.launch() 收到的原始参数。

    Returns:
        按当前运行模式规范化后的启动配置。
    """
    raw_kwargs = cast(RawLaunchKwargs, kwargs)
    validate_raw_launch_kwargs(raw_kwargs)

    mode = raw_kwargs["mode"]

    if mode == ExecMode.TRAIN:
        return _normalize_train_config(raw_kwargs)

    if mode == ExecMode.VALIDATE:
        return _normalize_validate_config(raw_kwargs)

    if mode == ExecMode.VISUALIZE:
        return _normalize_visualize_config(raw_kwargs)

    raise ValueError(f"不支持的运行模式：{mode}")


def _base_config(
    raw_kwargs: RawLaunchKwargs,
    *,
    work_dir: pathlib.Path,
    max_epochs: int,
) -> dict[str, Any]:
    """提取所有运行模式共享的基础配置字段。

    Args:
        raw_kwargs: OwlApp.launch() 的原始参数。
        work_dir: 当前任务工作目录。
        max_epochs: 当前任务实际运行轮次。

    Returns:
        包含 BaseLaunchConfig 字段的普通字典。
    """
    return {
        "mode": raw_kwargs["mode"],
        "work_dir": work_dir,
        "max_epochs": max_epochs,
        "checkpoint_path": raw_kwargs["checkpoint_path"],
        "finetune": raw_kwargs["finetune"],
        "device": raw_kwargs["device"],
        "model_name": raw_kwargs["model_name"],
        "model_cfg": dict(raw_kwargs.get("model_cfg") or {}),
        "owl_val_loaders": raw_kwargs.get("owl_val_loaders"),
    }


def _normalize_train_config(raw_kwargs: RawLaunchKwargs) -> TrainLaunchConfig:
    """规范化 TRAIN 模式配置。

    TRAIN 模式只返回训练真正需要的字段：
        - base config；
        - train loader；
        - criterion；
        - optimizer；
        - scheduler；
        - evaluator；
        - checkpoint autosave；
        - monitor。
    """
    work_dir = default_work_dir()

    config = _base_config(
        raw_kwargs,
        work_dir=work_dir,
        max_epochs=raw_kwargs["max_epochs"],
    )

    config["ckpt_autosave"] = raw_kwargs["ckpt_autosave"]
    config["monitor"] = raw_kwargs["monitor"]
    config["owl_train_loader"] = raw_kwargs["owl_train_loader"]

    config["criterion_name"] = raw_kwargs["criterion_name"]
    config["criterion_cfg"] = dict(raw_kwargs.get("criterion_cfg") or {})

    _apply_optimizer_config(config, raw_kwargs)
    _apply_scheduler_config(config, raw_kwargs)
    _apply_evaluator_config(config, raw_kwargs)

    return cast(TrainLaunchConfig, config)


def _normalize_validate_config(raw_kwargs: RawLaunchKwargs) -> ValidateLaunchConfig:
    """规范化 VALIDATE 模式配置。

    VALIDATE 模式只返回验证真正需要的字段：
        - base config；
        - evaluator。

    VALIDATE 模式下 max_epochs 固定为 1。
    """
    work_dir = default_work_dir()

    config = _base_config(
        raw_kwargs,
        work_dir=work_dir,
        max_epochs=1,
    )

    _apply_evaluator_config(config, raw_kwargs)

    return cast(ValidateLaunchConfig, config)


def _normalize_visualize_config(raw_kwargs: RawLaunchKwargs) -> VisualizeLaunchConfig:
    """规范化 VISUALIZE 模式配置。

    VISUALIZE 模式只返回可视化真正需要的字段：
        - base config；
        - visualizer。

    VISUALIZE 模式下 max_epochs 固定为 1。
    """
    work_dir = default_work_dir()

    config = _base_config(
        raw_kwargs,
        work_dir=work_dir,
        max_epochs=1,
    )

    _apply_visualizer_config(
        config,
        raw_kwargs,
        work_dir=work_dir,
    )

    return cast(VisualizeLaunchConfig, config)


def _apply_optimizer_config(
    config: dict[str, Any],
    raw_kwargs: RawLaunchKwargs,
) -> None:
    """向 TRAIN 配置注入 optimizer 配置。

    如果用户没有提供 optimizer，则使用框架默认 optimizer。
    """
    optimizer_name = raw_kwargs.get("optimizer_name")

    if optimizer_name:
        config["optimizer_name"] = optimizer_name
        config["optimizer_cfg"] = dict(raw_kwargs.get("optimizer_cfg") or {})
        return

    default_name, default_cfg = default_optimizer()
    config["optimizer_name"] = default_name
    config["optimizer_cfg"] = default_cfg


def _apply_scheduler_config(
    config: dict[str, Any],
    raw_kwargs: RawLaunchKwargs,
) -> None:
    """向 TRAIN 配置注入 scheduler 配置。

    如果用户没有提供 scheduler，则使用框架默认 scheduler。
    """
    scheduler_name = raw_kwargs.get("scheduler_name")

    if scheduler_name:
        config["scheduler_name"] = scheduler_name
        config["scheduler_cfg"] = dict(raw_kwargs.get("scheduler_cfg") or {})
        return

    default_name, default_cfg = default_scheduler()
    config["scheduler_name"] = default_name
    config["scheduler_cfg"] = default_cfg


def _apply_evaluator_config(
    config: dict[str, Any],
    raw_kwargs: RawLaunchKwargs,
) -> None:
    """向 TRAIN / VALIDATE 配置注入 evaluator 配置。

    如果用户没有提供 evaluator，则使用框架默认 evaluator。
    """
    evaluator_name = raw_kwargs.get("evaluator_name")

    if evaluator_name:
        config["evaluator_name"] = evaluator_name
        config["evaluator_cfg"] = dict(raw_kwargs.get("evaluator_cfg") or {})
        return

    default_name, default_cfg = default_evaluator()
    config["evaluator_name"] = default_name
    config["evaluator_cfg"] = default_cfg


def _apply_visualizer_config(
    config: dict[str, Any],
    raw_kwargs: RawLaunchKwargs,
    *,
    work_dir: pathlib.Path,
) -> None:
    """向 VISUALIZE 配置注入 visualizer 配置。

    如果用户没有提供 visualizer，则使用框架默认 visualizer。
    """
    visualizer_name = raw_kwargs.get("visualizer_name")

    if visualizer_name:
        config["visualizer_name"] = visualizer_name
        visualizer_cfg = dict(raw_kwargs.get("visualizer_cfg") or {})
    else:
        default_name, default_cfg = default_visualizer()
        config["visualizer_name"] = default_name
        visualizer_cfg = default_cfg

    config["visualizer_cfg"] = _normalize_visualizer_cfg(
        visualizer_cfg,
        work_dir=work_dir,
    )


def _normalize_visualizer_cfg(
    visualizer_cfg: dict[str, Any],
    *,
    work_dir: pathlib.Path,
) -> dict[str, Any]:
    """规范化 visualizer_cfg。

    处理规则：
        1. 如果没有 save_dir，则默认保存到 work_dir / "visual"。
        2. 如果 save_dir 是相对路径，则锚定到 work_dir 下。
        3. 如果 save_dir 是绝对路径，则原样保留。
        4. 如果没有 threshold，则补为 None。
    """
    cfg = dict(visualizer_cfg)

    save_dir = cfg.get("save_dir")
    if not save_dir:
        cfg["save_dir"] = str(default_visualizer_save_dir(work_dir))
    else:
        save_path = pathlib.Path(save_dir)
        if not save_path.is_absolute():
            cfg["save_dir"] = str(work_dir.joinpath(save_path))

    if "threshold" not in cfg:
        cfg["threshold"] = None

    return cfg