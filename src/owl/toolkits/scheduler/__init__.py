from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

from ..._internal.lazy import attach_lazy_modules
from ..._internal.registry import Registry

if TYPE_CHECKING:
    from . import poly

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "poly": ".poly",
    },
)

# 学习率调度器按 batch/iteration 粒度 step。
# batch-level scheduler 可以兼容 epoch-level scheduler：
# 只需要在同一个 epoch 内返回相同 lr factor 即可。
# 反过来，epoch-level scheduler 无法表达 warmup/poly/cosine 等细粒度策略。
BATCH_SCHEDULERS = Registry[LRScheduler]("batch_scheduler")
SCHEDULERS = BATCH_SCHEDULERS

# 导入默认学习率调度器实现，触发注册器注册
from . import poly

__all__.append("SCHEDULERS")