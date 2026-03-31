from torch.optim.lr_scheduler import LRScheduler
from torch import optim
from collections.abc import Callable
from typing import Any

# 定义全局字典
SchedulerFunc = Callable[..., LRScheduler]
_SCHEDULER_REGISTRY: dict[str, SchedulerFunc] = {}


def register_scheduler(name: str | None = None):
    """装饰器：将函数注册到调度器工厂中。

    Args:
        name (str, optional): 注册名称。默认使用函数名。
    """

    def decorator(func: SchedulerFunc):
        key = name if name else func.__name__
        if key in _SCHEDULER_REGISTRY:
            raise ValueError(f"调度器 '{key}' 已经被注册过了！")
        _SCHEDULER_REGISTRY[key] = func
        return func

    return decorator


def build_scheduler(
        config: dict[str, Any],
        optimizer: optim.Optimizer,
        epochs: int,
        batches: int
) -> LRScheduler:
    """根据配置构建学习率调度器。

    适配的 YAML 结构::

        schedule:
            type: poly
            param:
                power: 0.9

    此函数会自动注入 'epochs' 和 'batches' 参数给目标函数。

    Args:
        config (dict[str, Any]): 配置字典。
        optimizer (optim.Optimizer): 已创建的优化器实例。
        epochs (int): 训练总轮数 (来自 Engine)。
        batches (int): 每轮的 Batch 数 (来自 Engine)。

    Returns:
        LRScheduler: PyTorch 学习率调度器实例。
    """
    # 提取类型
    sch_type = config.get("type")
    if not sch_type:
        raise ValueError("构建调度器失败：配置缺少 'type' 字段。")

    # 检查注册
    if sch_type not in _SCHEDULER_REGISTRY:
        valid_list = list(_SCHEDULER_REGISTRY.keys())
        raise ValueError(f"未知的调度器类型: '{sch_type}'。支持列表: {valid_list}")

    # 提取参数
    sch_params = config.get("param", {}) or {}

    # 获取构建函数
    build_func = _SCHEDULER_REGISTRY[sch_type]

    # 调用函数
    # 将上下文 (epochs, batches) 显式传入
    # 如果具体的 scheduler 不需要这两个参数，它需要在定义时使用 **kwargs 忽略它们
    try:
        return build_func(
            optimizer=optimizer,
            epochs=epochs,
            batches=batches,
            **sch_params
        )
    except TypeError as e:
        raise ValueError(f"构建调度器 '{sch_type}' 失败，参数不匹配: {e}")

@register_scheduler(name="poly")
def poly(optimizer: optim.Optimizer, power: float, epochs: int, batches: int) -> LRScheduler:
    """初始化多项式衰减学习率调度器 (PolynomialLR)。

    该函数会根据传入的 epochs 和 batches 自动计算总迭代次数 (total_iters)。

    Args:
        optimizer (optim.Optimizer): 需要调整学习率的优化器实例。
        power (float): 多项式衰减的幂指数。
        epochs (int): 训练的总轮数 (Epochs)。
        batches (int): 每个 Epoch 中的 Batch 数量。

    Returns:
        LRScheduler: 多项式学习率调度器实例。
        """
    total_iters = epochs * batches
    poly_scheduler = optim.lr_scheduler.PolynomialLR(optimizer=optimizer,
                                                     total_iters=total_iters,
                                                     power=power)
    return poly_scheduler
