from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, Dict, Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

class OwlMetrics(ABC):
    @abstractmethod
    def __call__(self, dataloader: DataLoader, model: nn.Module) -> Dict[str, float]:
        """
        计算某个数据集的指标
        无需设置 model.eval()
        :param dataloader:
        :param model:
        :return: 比如 {"F1": 0.7, "auc": 0.9}
        """
        ...

class OwlFactory(ABC):
    @abstractmethod
    def create_model(self) -> nn.Module:
        """
        返回模型实例
        :return:
        """
        ...

    @abstractmethod
    def create_optimizer(self, model:nn.Module) -> optim.Optimizer:
        """
        根据 model 创建并返回优化器实例。
        :param model:
        :return:
        """
        ...

    @abstractmethod
    def create_scheduler(self, optimizer: optim.Optimizer, epochs:int, batches:int) -> Any | None:
        """
        根据 optimizer等参数 创建并返回学习率调度器。
        :param optimizer: 优化器
        :param epochs: 总的 epoch 数量
        :param batches: 每个 epoch 有多少个 batch
        :return:
        """
        ...

    @abstractmethod
    def create_train_dataloader(self) -> DataLoader:
        """
        返回训练数据集
        :return:
        """
        ...

    @abstractmethod
    def create_val_dataloader(self) -> Tuple[Optional[Dict[str, DataLoader]],
                                       Optional[OwlMetrics]]:
        """
        返回验证数据集和指标计算类
        :return:
        """
        return None, None

    @abstractmethod
    def create_criterion(self) -> nn.Module:
        """

        :return:
        """
        ...



class TrainMode(Enum):
    TRAIN = "train"       # 从头训练 (Scratch)
    RESUME = "resume"     # 断点续训 (Resume Full State)
    FINETUNE = "finetune" # 迁移学习 (Transfer Learning, Only Weights)


@dataclass
class BaseAugConfig:
    """
    p 概率值, 0-1之间
    """
    p: float

    def __post_init__(self):
        if self.p < 0 or self.p > 1:
            raise ValueError(f"{self.p} 取值必须是 [0,1] 之间")

@dataclass
class RotateConfig(BaseAugConfig):
    """
    随机旋转
    """
    pass

@dataclass
class VFlipConfig(BaseAugConfig):
    """
    垂直反转
    """
    pass

@dataclass
class HFlipConfig(BaseAugConfig):
    """
    水平反转
    """
    pass

@dataclass
class ResizeConfig(BaseAugConfig):
    """
    改变尺寸
    """
    height: int
    width: int

@dataclass
class JpegConfig(BaseAugConfig):
    """
    jpeg 压缩
    随机在 [quality_low, quality_high] 区间之间选择一个质量因子进行压缩
    """
    quality_low: int
    quality_high: int

    def __post_init__(self):
        super().__post_init__()
        if self.quality_low > self.quality_high:
            raise ValueError(f"quality_low 应该小于 quality_high")

        if self.quality_high > 100 or self.quality_low < 0:
            raise ValueError(f"quality_high 和 quality_low 范围是[0, 100]")

@dataclass
class GblurConfig(BaseAugConfig):
    """
    高斯模糊
    随机在 [kernel_low, kernel_high] 区间之间选择一个 kernal size
    """
    kernel_low: int
    kernel_high: int

    def __post_init__(self):
        super().__post_init__()
        if self.kernel_low > self.kernel_high:
            raise ValueError(
                f"GblurConfig Error: kernel_low ({self.kernel_low}) 不能大于 kernel_high ({self.kernel_high})")

        if self.kernel_low % 2 != 1 or self.kernel_high % 2 != 1:
            raise ValueError("kernel_low 和 kernel_high 必须是奇数")

@dataclass
class GNoiseConfig(BaseAugConfig):
    """
    高斯噪声
    """
    std_low: float
    std_high: float
    def __post_init__(self):
        super().__post_init__()

        if not (0.0 <= self.std_low <= 1.0):
            raise ValueError(f"GNoise Error: std_low ({self.std_low}) 必须在 [0, 1] 之间")

        if not (0.0 <= self.std_high <= 1.0):
            raise ValueError(f"GNoise Error: std_high ({self.std_high}) 必须在 [0, 1] 之间")

        if self.std_low > self.std_high:
            raise ValueError(f"GNoise Error: std_low ({self.std_low}) 不能大于 std_high ({self.std_high})")

@dataclass
class ScaleConfig(BaseAugConfig):
    """
    随机在 (1-scale1, 1+scale2) 之间旋转一个值进行缩放
    """
    scale1: float
    scale2: float
