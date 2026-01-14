import importlib
from collections.abc import Callable
from typing import Any, Type, TypeVar

T = TypeVar("T")

def _load_attr(path: str) -> Any:
    """动态加载模块属性，处理通用的导入逻辑。"""
    if "." not in path:
        raise ValueError(f"路径 '{path}' 格式错误。必须包含模块路径。")

    module_path, attr_name = path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ModuleNotFoundError(f"无法加载模块 '{module_path}': {e}")

    try:
        return getattr(module, attr_name)
    except AttributeError:
        raise AttributeError(f"模块 '{module_path}' 中找不到属性 '{attr_name}'")


def load_class(class_path: str, base_class: Type[T] = object) -> Type[T]:
    """动态加载类并校验继承关系。"""
    cls = _load_attr(class_path)

    # 类型检查：必须是类，且是 base_class 的子类
    if not isinstance(cls, type):
        raise TypeError(f"'{class_path}' 指向的不是一个类，而是 {type(cls)}。")

    if not issubclass(cls, base_class):
        raise TypeError(f"类 '{cls.__name__}' 不是 '{base_class.__name__}' 的子类。")

    return cls


def load_func(func_path: str) -> Callable[..., Any]:
    """动态加载函数并校验是否可调用。"""
    func = _load_attr(func_path)

    # 这里的类型检查：必须可调用
    if not callable(func):
        raise TypeError(f"'{func_path}' 指向的对象不可调用 (Not Callable)。")

    return func