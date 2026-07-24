from abc import ABC, abstractmethod

import torch.nn as nn

import owl.toolkits.data.datasets.image_mask_types
from .types import ModelReturn
from ..data import types

__all__ = ["OwlModel"]


class OwlModel(nn.Module, ABC):
    """Owl 模型基类。"""

    @abstractmethod
    def forward(
        self,
        batch_data: owl.toolkits.data.datasets.image_mask_types.ImageMaskBatch,
        current_epoch: int = 0,
        current_step: int = 0,
        **kwargs,
    ) -> ModelReturn:
        """执行模型的前向传播逻辑。

        Args:
            batch_data (owl.toolkits.data.datasets.image_mask_types.ImageMaskBatch): 包含当前批次数据的字典，通常包含篡改图像
                (tp_tensor) 及其对应的文件名等元信息。
            current_epoch (int, optional): 当前所处的训练轮次 (Epoch)。默认为 0。
            current_step (int, optional): 当前所处的全局 batch (Step)。默认为 0。
            **kwargs: 保留字典，用于接收未来框架可能会下发的其他扩展上下文参数。

        Returns:
            ModelReturn:
                模型 forward 方法的标准返回值，推荐返回 ModelOutput 字典。

                ModelOutput 至少应包含 logits 字段：

                1. ``logits``:
                    模型的原始、未经过激活函数处理的预测输出。

                    模型 forward 仅需输出未经过激活函数（如 Sigmoid/Softmax）
                    处理的原始张量。因为原始 Logits 可以无损转换为概率值，
                    而概率值转换为 Logits 会存在极大的数值精度丢失。

                2. ``metrics``:
                    可选字段。模型希望暴露给日志、训练监控器、状态服务或可视化组件
                    的轻量级指标。

                    metrics 的 key 必须是 str，value 只能是 str / int / float /
                    bool / None 等基础类型。

                    不允许放入 Tensor、ndarray、中间特征、计算图相关对象，
                    或其他复杂 Python 对象。

                    这些指标属于旁路观测信息，不应该参与反向传播。

                3. ``extra``:
                    可选字段。模型的其他附加输出，例如辅助监督 logits、
                    中间特征、可视化信息等。

                    extra 不会被训练监控器自动提取。
                    如果某些信息希望被日志或 monitor 读取，应放入 metrics，
                    而不是 extra。

                示例：
                    >>> return {
                    ...     "logits": logits,
                    ...     "metrics": {
                    ...         "router_weight": 0.62,
                    ...     },
                    ...     "extra": {
                    ...         "aux_logits": aux_logits,
                    ...     },
                    ... }

        """
        pass