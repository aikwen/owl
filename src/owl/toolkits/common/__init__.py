import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import fs
    from . import image
    from . import registry

_delayed_imports = {
    "fs": ".fs",
    "image": ".image",
    "registry": ".registry",

}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module

    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = ["fs",
           "image",
           "registry",
           ]