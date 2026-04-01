from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

from ..data import types

class OwlModel(nn.Module, ABC):
    """Owl 模型基类。
    """

    @abstractmethod
    def forward(self, batch_data: types.DataSetBatch, current_epoch: int = 0, current_step: int = 0, **kwargs) -> Any:
        """执行模型的前向传播逻辑。

        Args:
            batch_data (types.DataSetBatch): 包含当前批次数据的字典，通常包含篡改图像
                (tp_tensors) 及其对应的文件名等元信息。
            current_epoch (int, optional): 当前所处的训练轮次 (Epoch)。 默认为 0。
            current_step (int, optional): 当前所处的全局batch (Step)。默认为 0。
            **kwargs: 保留字典，用于接收未来框架可能会下发的其他扩展上下文参数。

        Returns:
            Any: 模型的预测结果。建议返回 Dict[str, torch.Tensor] 格式
            （例如: {"mask": mask_pred, "cls": cls_pred}），以优雅地支持多头网络输出，
            但为保证兼容性，也允许返回单一的 torch.Tensor 或 Tuple。
        """
        pass