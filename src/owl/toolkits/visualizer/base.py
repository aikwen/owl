import pathlib
from abc import ABC, abstractmethod
import torch
from ..data.types import DataSetBatch
from ..model.types import ModelOutput


class OwlVisualizer(ABC):
    """可视化基类

    负责将模型输出的 Logits 转换为概率图或二值化掩码。
    具体的渲染和落盘逻辑由子类的 __call__ 方法实现。
    """

    def __init__(self, save_dir: str , threshold: float | None = 0.5):
        """
        Args:
            save_dir (str): 可视化结果保存的基础目录。
            threshold (float | None): 二值化阈值。
                - 传 0.5: 输出 {0, 1} 的硬掩码图。
                - 传 None: 输出 [0, 1] 的软概率灰度图。
        """
        self.save_dir = pathlib.Path(save_dir)
        self.threshold = threshold

    def _process_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits -> Probs -> Mask"""
        # logits 映射到 [0, 1] 概率空间
        probs = torch.sigmoid(logits)

        # 阈值截断 (如果指定了 threshold)
        if self.threshold is not None:
            probs = (probs >= self.threshold).float()

        return probs

    @abstractmethod
    def __call__(self, batch_data: DataSetBatch, outputs: ModelOutput, dataset_name: str):
        """执行可视化与保存操作

        Args:
            batch_data (DataSetBatch): 当前批次的数据，包含 tp_names 等元数据。
            outputs (ModelOutput): 模型的标准输出包装器。
            dataset_name (str): 当前数据集名称，用于隔离保存目录。
        """
        pass