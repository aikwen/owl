from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class OwlCriterion(nn.Module, ABC):
    """Owl 损失函数协议基类。"""

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        """
        pass