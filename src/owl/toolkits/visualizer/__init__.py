import importlib
from typing import TYPE_CHECKING
from ..common import registry

if TYPE_CHECKING:
    from .base import OwlVisualizer

_delayed_imports = {
    "base": ".base",
}

def __getattr__(name: str):
    if name in _delayed_imports:
        module = importlib.import_module(_delayed_imports[name], package=__package__)
        # 缓存到 globals 中，下次访问不再触发 __getattr__
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 定义可视化组件注册表
from .base import OwlVisualizer
VISUALIZERS = registry.Registry[OwlVisualizer]("visualizer")

__all__ = ["VISUALIZERS", "base"]