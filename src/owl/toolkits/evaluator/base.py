from abc import ABC, abstractmethod
from typing import Any, Dict
from ..data.types import DataSetBatch

class OwlEvaluator(ABC):
    """Owl 评估器基类"""

    @abstractmethod
    def reset(self):
        """在每个 DataLoader 开始前调用，清空上一轮的缓存"""
        pass

    @abstractmethod
    def update(self, outputs: Any, batch_data: DataSetBatch):
        """在每个 Batch 结束后调用，收集预测值和真实标签"""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """在 DataLoader 结束后调用，计算并返回最终指标字典"""
        pass