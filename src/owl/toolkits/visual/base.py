# src/owl/toolkits/visual/base.py
from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class OwlVisualizer(nn.Module, ABC):
    """可视化基类。

    将模型输出转换为最终的 [0, 1] 二值化图像张量。
    """

    @abstractmethod
    def forward(self, model_outputs: Any) -> torch.Tensor:
        """执行可视化处理逻辑。

        Returns:
            torch.Tensor: 形状为 [BatchSize, 1, H, W] 的张量。
                         必须是已经过阈值处理的二值化图（0 或 1）。
        """
        pass