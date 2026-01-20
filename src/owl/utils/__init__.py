import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import io
    from . import img_op
    from . import validator
    from . import console
    from . import metrics

_delayed_imports = {
    "file_io": ".file_io",
    "img_aug": ".img_aug",
    "img_op": ".img_op",
    "types": ".types",
    "validator": ".validator",
    "console": ".console",
    "metrics": ".metrics",
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module
    raise AttributeError(f"module 'utils' has no attribute '{name}'")

__all__ = [
    "io",
    "img_op",
    "validator",
    "metrics",
    "validator",
    "console",
]