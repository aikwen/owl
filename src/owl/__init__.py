import importlib
from typing import TYPE_CHECKING

try:
    from importlib.metadata import version
    __version__ = version("owl-imdl")
except ImportError:
    __version__ = "unknown"

# IDE 提示
if TYPE_CHECKING:
    from . import core
    from . import utils


_delayed_imports = {
    "core": ".core",
    "utils": ".utils"
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module
    raise AttributeError(f"module 'owl' has no attribute '{name}'")


__all__ = [
    "core",
    "utils",
    "__version__",
]