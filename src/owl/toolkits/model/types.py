from typing import Any, TypedDict, TypeAlias

import torch


MetricValue: TypeAlias = str | int | float | bool | None
Metrics: TypeAlias = dict[str, MetricValue]


class ModelOutput(TypedDict, total=False):
    """Owl 模型输出的标准约定格式。

    模型 forward 应输出未经过激活函数处理的原始张量，例如 logits。

    原始 logits 可以稳定地用于损失函数计算，例如 BCEWithLogitsLoss、
    CrossEntropyLoss 等。评估器和可视化器如果需要概率值，应在组件内部
    根据任务类型显式调用 sigmoid 或 softmax。

    Attributes:
        logits:
            模型的原始预测输出，通常是未经过 Sigmoid / Softmax 的 logits。
            logits 是下游损失函数、评估器和可视化器默认读取的主输出。

        metrics:
            模型希望暴露给日志、训练监控器、状态服务或可视化组件的轻量级指标。

            metrics 必须是 dict[str, MetricValue]，其中 key 为指标名称，
            value 只能是 str / int / float / bool / None 等基础类型。

            不允许放入 Tensor、ndarray、中间特征、计算图相关对象，
            或其他复杂 Python 对象。

            这些指标属于旁路观测信息，不应该参与反向传播。

            示例：
                {
                    "router_rgb_weight": 0.62,
                    "router_srm_weight": 0.38,
                }

        extra:
            模型的其他附加输出，例如辅助监督 logits、中间特征、可视化信息等。

            extra 不会被训练监控器自动提取，适合放置仅供特定组件使用的扩展数据。
            如果某些信息希望被日志或 monitor 读取，应放入 metrics，而不是 extra。
    """

    logits: torch.Tensor
    metrics: Metrics
    extra: dict[str, Any]


ModelReturn: TypeAlias = ModelOutput