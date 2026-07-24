from typing import Any, TypedDict, TypeAlias

import torch


MetricValue: TypeAlias = str | int | float | bool | None
Metrics: TypeAlias = dict[str, MetricValue]


class CriterionOutput(TypedDict, total=False):
    """Owl 损失函数输出的标准约定格式。

    Attributes:
        loss:
            最终用于 backward 的综合损失。
            必须是零维标量 Tensor。

        metrics:
            损失函数希望暴露给日志、训练监控器、状态服务或可视化组件的轻量级指标。

            metrics 必须是 dict[str, MetricValue]，其中 key 为指标名称，
            value 只能是 str / int / float / bool / None 等基础类型。

            不允许放入 Tensor、ndarray、中间特征、计算图相关对象，
            或其他复杂 Python 对象。

            这些指标属于旁路观测信息，不应该参与反向传播。

            示例：
                {
                    "bce_loss": 0.21,
                    "dice_loss": 0.34,
                    "edge_loss": 0.08,
                }

        extra:
            损失函数的其他附加信息。

            extra 不会被训练监控器自动提取，适合放置仅供特定组件使用的扩展数据。
            如果某些信息希望被日志或 monitor 读取，应放入 metrics，而不是 extra。
    """

    loss: torch.Tensor
    metrics: Metrics
    extra: dict[str, Any]


CriterionReturn: TypeAlias = torch.Tensor | CriterionOutput