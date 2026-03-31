from abc import ABC, abstractmethod
from torch import nn
import torch
from .schemas import CriterionBundle
from ..utils import reflect


class OwlCriterion(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: CriterionBundle) -> torch.Tensor:
        """计算损失值。

        Args:
            batch (CriterionBundle): 包含 model_output, gt 等信息的上下文对象。
        """
        ...


def build_criterion(class_path: str) -> OwlCriterion:
    """通过反射构建损失函数实例。

    仅支持字符串路径配置， Loss 类必须支持无参构造。

    Example:
        criterion: "my_project.loss.MyCustomLoss"

    Args:
        class_path (str): 类的完整点分路径。

    Returns:
        OwlCriterion: 实例化后的损失函数对象。
    """
    if not isinstance(class_path, str):
        raise TypeError(f"Criterion 配置错误：必须是字符串路径，实际收到 {type(class_path)}。")

    # 反射加载类 (并强制检查是否继承自 OwlCriterion)
    cls = reflect.load_class(class_path, base_class=OwlCriterion)

    # 无参实例化
    try:
        instance = cls()
    except TypeError as e:
        # 捕获用户定义的类如果包含必填参数导致的错误
        raise ValueError(
            f"实例化 Loss 类 '{class_path}' 失败。\n"
            f"Loss 类必须支持无参构造 (__init__ 不应有必填参数)。\n"
            f"底层错误: {e}"
        )

    return instance
