from abc import ABC, abstractmethod

import torch.nn as nn

from ..data import types
from ..model.types import ModelOutput
from .types import CriterionReturn

__all__ = ["OwlCriterion"]

class OwlCriterion(nn.Module, ABC):
    """Owl 损失函数协议基类。
    """

    @abstractmethod
    def forward(self, model_outputs: ModelOutput, batch_data: types.DataSetBatch, current_epoch: int = 0, current_step: int = 0,
                **kwargs) -> CriterionReturn:
        """计算模型预测结果与真实标签之间的损失值。

        Args:
            model_outputs (ModelOutput): 模型 forward 方法的返回值。
            batch_data (types.DataSetBatch): 包含当前批次数据的字典。
            current_epoch (int, optional): 当前所处的训练轮次 (Epoch)。
            current_step (int, optional): 当前所处的全局batch数 (Step)。默认为 0。
            **kwargs: 保留字典，用于接收扩展上下文参数。

        Returns:
            CriterionReturn:
                允许返回两种格式：

                1. ``torch.Tensor``:
                    旧版兼容写法，表示最终综合损失。
                    框架会直接对该 Tensor 调用 backward。

                2. ``CriterionOutput``:
                    新版推荐写法，必须包含 loss 字段；
                    可选 extra 字段。
                    其中 loss 用于 backward，metrics 用于日志、状态服务、可视化。
                """
        pass