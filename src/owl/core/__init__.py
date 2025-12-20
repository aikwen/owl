import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dataset
    from . import app
    from . import config

_delayed_imports = {
    "dataset": ".dataset",
    "app": ".app",
    "config": ".config",
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module

    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = ["dataset",
           "app",
           "config"]