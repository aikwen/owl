from torch import nn, optim
from . import OPTIMIZERS

@OPTIMIZERS.register(name="adamw")
def adamw(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """封装后的 AdamW 工厂函数"""
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)