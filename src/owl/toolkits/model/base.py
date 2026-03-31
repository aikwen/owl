from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class OwlModel(nn.Module, ABC):
    """Owl 引擎模型协议基类。

    所有用户自定义模型必须继承此类。
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        pass