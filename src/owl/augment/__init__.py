import importlib
from typing import TYPE_CHECKING

# IDE 提示
if TYPE_CHECKING:
    from . import transforms
    from . import types


_delayed_imports = {
    "transforms": ".transforms",
    "types": ".types",
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module
    raise AttributeError(f"module 'augment' has no attribute '{name}'")


__all__ = [
    "transforms",
    "types",
]