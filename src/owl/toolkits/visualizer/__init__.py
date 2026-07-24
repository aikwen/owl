from typing import TYPE_CHECKING

from .base import OwlVisualizer
from ..._internal.lazy import attach_lazy_modules
from ..._internal.registry import Registry

if TYPE_CHECKING:
    from . import base

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "base": ".base",
    },
)

VISUALIZERS = Registry[OwlVisualizer]("visualizer")

__all__.extend([
    "OwlVisualizer",
    "VISUALIZERS",
])