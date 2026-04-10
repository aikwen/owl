import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (common, criterion, data, evaluator, model,
                   optimizer, scheduler, visualizer)

_delayed_imports = {
    "common": ".common",
    "criterion": ".criterion",
    "data": ".data",
    "evaluator": ".evaluator",
    "model": ".model",
    "optimizer": ".optimizer",
    "scheduler": ".scheduler",
    "visualizer": ".visualizer",
}

def __getattr__(name:str):
    if name in _delayed_imports:
        module =  importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存
        globals()[name] = module
        return module

    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = [
    "common",
    "criterion",
    "data",
    "evaluator",
    "model",
    "optimizer",
    "scheduler",
    "visualizer",
]