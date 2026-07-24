from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import fs
    from . import image
    from . import seed
    from . import metrics
    from . import ckpt

from ..._internal.lazy import attach_lazy_modules

__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "fs": ".fs",
        "image": ".image",
        "seed": ".seed",
        "metrics": ".metrics",
        "ckpt": ".ckpt",
    },
)