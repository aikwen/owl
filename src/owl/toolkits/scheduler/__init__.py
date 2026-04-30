from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

from ..._internal.lazy import attach_lazy_modules
from ..common.registry import Registry

if TYPE_CHECKING:
    from . import poly

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "poly": ".poly",
    },
)

SCHEDULERS = Registry[LRScheduler]("scheduler")

# 导入默认学习率调度器实现，触发注册器注册
from . import poly

__all__.append("SCHEDULERS")