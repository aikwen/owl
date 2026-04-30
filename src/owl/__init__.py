from typing import TYPE_CHECKING

from ._internal.lazy import attach_lazy_modules
from .toolkits.common.seed import seed_everything


try:
    from importlib.metadata import version

    __version__ = version("owl-imdl")
except ImportError:
    __version__ = "unknown"


seed = seed_everything


# IDE 提示
if TYPE_CHECKING:
    from . import engine
    from . import toolkits


__all__ = attach_lazy_modules(
    target_globals=globals(),
    package=__package__,
    delayed_modules={
        "engine": ".engine",
        "toolkits": ".toolkits",
    },
)

__all__.extend([
    "__version__",
    "seed",
    "seed_everything",
])