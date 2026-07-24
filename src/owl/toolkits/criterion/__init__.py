from typing import TYPE_CHECKING

from .base import OwlCriterion
from ..._internal.lazy import attach_lazy_modules
from ..._internal.registry import Registry

if TYPE_CHECKING:
    from . import base
    from . import types

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "base": ".base",
        "types": ".types",
    },
)

CRITERIA = Registry[OwlCriterion]("criterion")

__all__.extend([
    "OwlCriterion",
    "CRITERIA",
])