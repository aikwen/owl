from typing import TYPE_CHECKING

from .base import OwlModel
from ..._internal.lazy import attach_lazy_modules
from ..common.registry import Registry

if TYPE_CHECKING:
    from . import base

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "base": ".base",
    },
)

MODELS = Registry[OwlModel]("model")

__all__.extend([
    "OwlModel",
    "MODELS",
])