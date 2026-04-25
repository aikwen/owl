from abc import ABC, abstractmethod
from .types import ModelOutput
import torch.nn as nn

from ..data import types

class OwlModel(nn.Module, ABC):
    """Owl 模型基类。
    """

    @abstractmethod
    def forward(self, batch_data: types.DataSetBatch, current_epoch: int = 0, current_step: int = 0, **kwargs) -> ModelOutput:
        """执行模型的前向传播逻辑。

        Args:
            batch_data (types.DataSetBatch): 包含当前批次数据的字典，通常包含篡改图像
                (tp_tensor) 及其对应的文件名等元信息。
            current_epoch (int, optional): 当前所处的训练轮次 (Epoch)。 默认为 0。
            current_step (int, optional): 当前所处的全局batch (Step)。默认为 0。
            **kwargs: 保留字典，用于接收未来框架可能会下发的其他扩展上下文参数。

        Returns:
            ModelOutput: 模型的输出，实际上是一个字典，必须包含 logits key，作为 Evaluator 自动提取结果；
            比如::
                >>> class Model(OwlModel):
                >>>     def __init__(self, ...):
                >>>         super().__init__()
                >>>         ...
                >>>     def forward(self, batch_data: types.DataSetBatch, current_epoch: int = 0, current_step: int = 0,**kwargs) -> ModelOutput:
                >>>         ...
                >>>         return {
                >>>             "logits": torch.tensor([1,2,3]),
                                "other": ...,
                >>>         }

        """
        pass