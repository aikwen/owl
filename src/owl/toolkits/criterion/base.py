from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn

from ..data import types


class OwlCriterion(nn.Module, ABC):
    """Owl 损失函数协议基类。
    """

    @abstractmethod
    def forward(self, model_outputs: Any, batch_data: types.DataSetBatch, current_epoch: int = 0, current_step: int = 0,
                **kwargs) -> torch.Tensor:
        """计算模型预测结果与真实标签之间的损失值。

        Args:
            model_outputs (Any): 模型 forward 方法的返回值。
            batch_data (types.DataSetBatch): 包含当前批次数据的字典。
            current_epoch (int, optional): 当前所处的训练轮次 (Epoch)。
            current_step (int, optional): 当前所处的全局batch数 (Step)。默认为 0。
            **kwargs: 保留字典，用于接收扩展上下文参数。

        Returns:
            torch.Tensor: 一个零维度的标量张量 (Scalar Tensor)，代表当前批次的最终综合损失值。会直接调用其 `.backward()` 方法来计算梯度。
        """
        pass