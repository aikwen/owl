from typing import TYPE_CHECKING

from .._internal.lazy import attach_lazy_modules

if TYPE_CHECKING:
    from . import common
    from . import criterion
    from . import data
    from . import evaluator
    from . import model
    from . import optimizer
    from . import scheduler
    from . import visualizer

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "common": ".common",
        "criterion": ".criterion",
        "data": ".data",
        "evaluator": ".evaluator",
        "model": ".model",
        "optimizer": ".optimizer",
        "scheduler": ".scheduler",
        "visualizer": ".visualizer",
    },
)