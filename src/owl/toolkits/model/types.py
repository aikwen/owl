from typing import TypedDict

import torch


class ModelOutput(TypedDict, total=False):
    """Owl 模型输出的标准约定格式。

    必须包含:
        - prediction(torch.Tensor): 模型的最终预测输出。

    """
    prediction: torch.Tensor