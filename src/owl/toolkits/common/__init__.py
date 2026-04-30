from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import fs
    from . import image
    from . import registry
    from . import seed
    from . import metrics
    from . import ckpt
    from . import fmt
    from . import logger

from ..._internal.lazy import attach_lazy_modules

attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "fs": ".fs",
        "image": ".image",
        "registry": ".registry",
        "seed": ".seed",
        "metrics": ".metrics",
        "ckpt": ".ckpt",
        "fmt": ".fmt",
        "logger": ".logger",
    },
)