from __future__ import annotations

import importlib
from types import ModuleType
from typing import Mapping


def attach_lazy_modules(
    target_globals: dict,
    package: str | None,
    delayed_modules: Mapping[str, str],
) -> list[str]:
    """
    给当前包挂载子模块懒加载能力。

    当用户第一次访问包下的某个子模块时，才真正导入该模块。
    例如访问 ``common.fs`` 时，才导入 ``owl.toolkits.common.fs``。
    Args:
        target_globals (dict): 调用方模块的 globals()。
        package (str | None): 调用方模块的 __package__。
        delayed_modules (Mapping[str, str]): 子模块懒加载映射表。

    Returns:
        list[str]: 建议写入 __all__ 的模块名列表。
    """

    if package is None:
        package = target_globals.get("__name__", "")

    def __getattr__(name: str) -> ModuleType:
        if name in delayed_modules:
            module = importlib.import_module(delayed_modules[name], package=package)

            # 缓存，避免下次再次触发 __getattr__
            target_globals[name] = module

            return module

        current_module_name = target_globals.get("__name__", package)
        raise AttributeError(f"module {current_module_name!r} has no attribute {name!r}")

    target_globals["__getattr__"] = __getattr__
    return list(delayed_modules.keys())