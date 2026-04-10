import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import app
    from . import engine
    from . import pipeline
    from . import state

_delayed_imports = {
    "app": ".app",
    "engine": ".engine",
    "pipline": ".pipeline",
    "state": ".state",
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module

    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = [
    "app",
    "engine",
    "pipeline",
    "state",
]