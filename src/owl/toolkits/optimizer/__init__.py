from typing import TYPE_CHECKING

from torch import optim

from ..._internal.lazy import attach_lazy_modules
from ..._internal.registry import Registry

if TYPE_CHECKING:
    from . import adamw

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "adamw": ".adamw",
    },
)

OPTIMIZERS = Registry[optim.Optimizer]("optimizer")

# 导入默认优化器实现，触发注册器注册
from . import adamw

__all__.append("OPTIMIZERS")
