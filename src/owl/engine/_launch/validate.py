from __future__ import annotations

from typing import Any, cast

from .types import RawLaunchKwargs
from ..state import ExecMode


_COMPONENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("model_name", "model_cfg"),
    ("criterion_name", "criterion_cfg"),
    ("optimizer_name", "optimizer_cfg"),
    ("scheduler_name", "scheduler_cfg"),
    ("evaluator_name", "evaluator_cfg"),
    ("visualizer_name", "visualizer_cfg"),
)


def validate_raw_launch_kwargs(kwargs: RawLaunchKwargs) -> None:
    """检查 OwlApp.launch() 原始输入参数。

    该函数只做校验，不修改 kwargs。

    Args:
        kwargs: OwlApp.launch() 收到的原始参数字典。

    Raises:
        TypeError: 参数类型不符合约定时抛出。
        ValueError: 参数组合不符合运行模式要求时抛出。
    """
    validate_mode(kwargs)
    validate_common_kwargs(kwargs)
    validate_component_pairs(kwargs)
    validate_mode_requirements(kwargs)


def validate_mode(kwargs: RawLaunchKwargs) -> None:
    """检查运行模式是否合法。"""
    mode = kwargs.get("mode")

    if not isinstance(mode, ExecMode):
        raise TypeError(
            f"类型错误：'mode' 必须是 ExecMode 类型，当前为 {type(mode).__name__}"
        )


def validate_common_kwargs(kwargs: RawLaunchKwargs) -> None:
    """检查各运行模式通用参数。"""
    max_epochs = kwargs.get("max_epochs")
    if not isinstance(max_epochs, int):
        raise TypeError(
            f"类型错误：'max_epochs' 必须是 int 类型，当前为 {type(max_epochs).__name__}"
        )

    if max_epochs <= 0:
        raise ValueError("参数错误：'max_epochs' 必须大于 0。")

    ckpt_autosave = kwargs.get("ckpt_autosave")
    if not isinstance(ckpt_autosave, bool):
        raise TypeError(
            f"类型错误：'ckpt_autosave' 必须是 bool 类型，当前为 {type(ckpt_autosave).__name__}"
        )

    finetune = kwargs.get("finetune")
    if not isinstance(finetune, bool):
        raise TypeError(
            f"类型错误：'finetune' 必须是 bool 类型，当前为 {type(finetune).__name__}"
        )

    monitor = kwargs.get("monitor")
    if not isinstance(monitor, bool):
        raise TypeError(
            f"类型错误：'monitor' 必须是 bool 类型，当前为 {type(monitor).__name__}"
        )

    checkpoint_path = kwargs.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        raise TypeError(
            f"类型错误：'checkpoint_path' 必须是 str 类型，当前为 {type(checkpoint_path).__name__}"
        )


def validate_component_pairs(kwargs: RawLaunchKwargs) -> None:
    """检查组件 name/cfg 是否成对提供。

    规则：
        1. 如果提供了组件 name，则 cfg 不能为 None。
        2. 如果提供了 cfg，则必须提供组件 name。
        3. name 必须是 str。
        4. cfg 必须是 dict。

    注意：
        RawLaunchKwargs 是 TypedDict。TypedDict 对动态 key 的类型检查比较严格，
        因此这里通过 cast 将其视为普通 dict[str, Any] 进行统一遍历检查。
    """
    data = cast(dict[str, Any], kwargs)

    for name_key, cfg_key in _COMPONENT_PAIRS:
        name_val = data.get(name_key)
        cfg_val = data.get(cfg_key)

        if (name_val and cfg_val is None) or (not name_val and cfg_val is not None):
            raise ValueError(
                f"参数不匹配：'{name_key}' 和 '{cfg_key}' 必须成对提供，"
                f"或全不提供以使用默认值。当前状态: {name_key}={name_val}, {cfg_key}={cfg_val}"
            )

        if name_val and not isinstance(name_val, str):
            raise TypeError(
                f"类型错误：'{name_key}' 必须是 str 类型，当前为 {type(name_val).__name__}"
            )

        if cfg_val is not None and not isinstance(cfg_val, dict):
            raise TypeError(
                f"类型错误：'{cfg_key}' 必须是 dict 类型，当前为 {type(cfg_val).__name__}"
            )


def validate_mode_requirements(kwargs: RawLaunchKwargs) -> None:
    """检查不同运行模式下的必需参数。"""
    mode = kwargs["mode"]

    if mode == ExecMode.TRAIN:
        validate_train_requirements(kwargs)
        return

    if mode == ExecMode.VALIDATE:
        validate_validate_requirements(kwargs)
        return

    if mode == ExecMode.VISUALIZE:
        validate_visualize_requirements(kwargs)
        return

    raise ValueError(f"不支持的运行模式：{mode}")


def validate_train_requirements(kwargs: RawLaunchKwargs) -> None:
    """检查 TRAIN 模式必需参数。"""
    if kwargs.get("owl_train_loader") is None:
        raise ValueError("参数错误：TRAIN 模式必须提供 'owl_train_loader'。")

    criterion_name = kwargs.get("criterion_name")
    if not criterion_name or not criterion_name.strip():
        raise ValueError(
            "参数错误：TRAIN 模式下必须显式指定 'criterion_name'，框架不提供默认损失函数。"
        )


def validate_validate_requirements(kwargs: RawLaunchKwargs) -> None:
    """检查 VALIDATE 模式必需参数。"""
    if kwargs.get("owl_val_loaders") is None:
        raise ValueError("参数错误：VALIDATE 模式下 'owl_val_loaders' 不能为空。")


def validate_visualize_requirements(kwargs: RawLaunchKwargs) -> None:
    """检查 VISUALIZE 模式必需参数。"""
    if kwargs.get("owl_val_loaders") is None:
        raise ValueError("参数错误：VISUALIZE 模式下 'owl_val_loaders' 不能为空。")