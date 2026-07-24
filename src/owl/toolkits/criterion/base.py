from abc import ABC, abstractmethod

import torch.nn as nn

import owl.toolkits.data.datasets.image_mask_types
from ..data import types
from ..model.types import ModelOutput
from .types import CriterionReturn

__all__ = ["OwlCriterion"]


class OwlCriterion(nn.Module, ABC):
    """Owl 损失函数协议基类。"""

    @abstractmethod
    def forward(
        self,
        model_outputs: ModelOutput,
        batch_data: owl.toolkits.data.datasets.image_mask_types.ImageMaskBatch,
        current_epoch: int = 0,
        current_step: int = 0,
        **kwargs,
    ) -> CriterionReturn:
        """计算模型预测结果与真实标签之间的损失值。

        Args:
            model_outputs: 模型 forward 方法的返回值。
            batch_data: 包含当前批次数据的字典。
            current_epoch: 当前所处的训练轮次。
            current_step: 当前所处的全局 batch 数。
            **kwargs: 保留字典，用于接收扩展上下文参数。

        Returns:
            CriterionReturn:
                允许返回两种格式：

                1. torch.Tensor:
                    兼容写法，表示最终综合损失。
                    框架会直接对该 Tensor 调用 backward。

                2. CriterionOutput:
                    推荐写法，必须包含 loss 字段。

                    可选 metrics 字段，用于暴露日志、训练监控器、
                    状态服务或可视化组件需要读取的轻量级指标。
                    metrics 的 key 必须是 str，value 只能是
                    str / int / float / bool / None 等基础类型。

                    可选 extra 字段，用于放置损失函数的其他附加信息。
                    extra 不会被训练监控器自动提取。
        """
        pass