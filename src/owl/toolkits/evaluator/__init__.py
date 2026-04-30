from typing import TYPE_CHECKING

from ..._internal.lazy import attach_lazy_modules
from ..._internal.registry import Registry

if TYPE_CHECKING:
    from . import base
    from . import default

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "base": ".base",
        "default": ".default",
    },
)

EVALUATORS = Registry("evaluator")

# 触发默认评估器注册
from .default import DefaultEvaluator

__all__.extend([
    "EVALUATORS",
    "DefaultEvaluator",
])