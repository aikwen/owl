from torch import nn, optim
from ..common import registry

OPTIMIZERS = registry.Registry[optim.Optimizer]("optimizer")

# 自动装载
from . import adamw

__all__ = ["OPTIMIZERS"]

