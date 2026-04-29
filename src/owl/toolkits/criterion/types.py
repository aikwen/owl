from typing import Any, TypedDict, TypeAlias
import torch

class CriterionOutput(TypedDict, total=False):
    """Owl 损失函数输出的标准约定格式。

    Attributes:
        loss:
            最终用于 backward 的综合损失。必须是零维标量 Tensor。

        extra:
            损失函数的附加信息，
            ``extra["metrics"]`` 如果存在，会被 Owl 训练监控器自动提取，
            用于日志、状态服务、可视化等旁路观测功能。
            ``extra["metrics"]`` 应只包含轻量级、可序列化或可安全转换的值，
            例如 ``float/int/str/bool/None``，或零维 ``Tensor``。
            不建议放入大 Tensor、中间特征或需要保留计算图的对象。

    """
    loss: torch.Tensor
    extra: dict[str, Any]

CriterionReturn: TypeAlias = torch.Tensor | CriterionOutput