from .base import OwlModel
from ..common import registry

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from . import base

_delayed_imports = {
    "base": ".base",
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module

    raise AttributeError(f"module 'core' has no attribute '{name}'")

MODELS = registry.Registry[OwlModel]("model")

__all__ = ["MODELS", "base"]