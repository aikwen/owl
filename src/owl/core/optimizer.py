from torch import  nn, optim
from collections.abc import Callable
from typing import Any

# 用于存储 "名称 -> 函数" 的映射字典
OptimizerFunc = Callable[..., optim.Optimizer]
_OPTIMIZER_REGISTRY: dict[str, OptimizerFunc] = {}


def register_optimizer(name: str = None):
    """装饰器：将函数注册到优化器工厂中。

    Args:
        name (str, optional): 注册的名称。如果为空，则默认使用函数名。
    """

    def decorator(func: Callable):
        key = name if name else func.__name__
        if key in _OPTIMIZER_REGISTRY:
            raise ValueError(f"优化器 '{key}' 已经被注册过了！")

        _OPTIMIZER_REGISTRY[key] = func
        return func

    return decorator


def build_optimizer(config: dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """根据嵌套配置字典构建优化器实例。

    适配的 YAML 结构::

        optimizer:
            type: adamw
            param:
                lr: 0.001
                weight_decay: 0.01

    Args:
        config (dict[str, Any]): 包含 'type' 和可选 'param' 的配置字典。
        model (nn.Module): 需要优化的模型。

    Returns:
        optim.Optimizer: 构建好的 PyTorch 优化器实例。
    """
    # 提取类型名称
    opt_type = config.get("type")

    if not opt_type:
        raise ValueError("构建优化器失败：配置字典中缺少 'type' 字段。")

    # 检查注册表
    if opt_type not in _OPTIMIZER_REGISTRY:
        # 打印出所有可用的优化器
        valid_opts = list(_OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"未知的优化器类型: '{opt_type}'。支持列表: {valid_opts}")

    # 提取参数字典 (param)
    # 如果 YAML 里没有写 param 字段，默认使用空字典 {}
    # 如果 YAML 里面写了 param 字段，但是没有内容防止 yaml 解析为 None
    opt_params = config.get("param", {}) or {}

    # 获取构建函数
    build_func = _OPTIMIZER_REGISTRY[opt_type]

    # 调用 (将 opt_params 字典解包传入)
    # 相当于调用: adamw(model, lr=0.001, weight_decay=0.01)
    try:
        return build_func(model, **opt_params)
    except TypeError as e:
        # 捕获参数不匹配的错误（比如用户 YAML 里写了 decay_wight 但函数只要 weight_decay）
        raise ValueError(f"构建优化器 '{opt_type}' 失败，参数不匹配: {e}")


@register_optimizer(name="adamw")
def adamw(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """ AdamW 优化器。

        Args:
            model (nn.Module): 包含需要优化参数的神经网络模型。
            lr (float): 优化器的学习率。
            weight_decay (float): 权重衰减系数（L2 惩罚）。

        Returns:
            optim.Optimizer: AdamW 优化器实例。
        """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


