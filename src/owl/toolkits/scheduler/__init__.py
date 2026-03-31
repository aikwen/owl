from torch.optim.lr_scheduler import LRScheduler
from ..common import registry

SCHEDULERS = registry.Registry[LRScheduler]("scheduler")

from . import poly

__all__ = ["SCHEDULERS"]