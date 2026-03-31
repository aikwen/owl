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


def load_validate_func(func_path: str) -> Callable:
    """验证函数专用加载器 (严格模式)。

    要求目标函数的签名必须严格匹配：
    (dataloader: DataLoader, model: torch.nn.Module, device: Any)

    Raises:
        TypeError: 如果参数数量、名称或类型注解不匹配。
    """
    import inspect
    import torch
    from torch.utils.data import DataLoader
    # 加载函数
    func = load_func(func_path)

    # 获取用户函数的签名
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # 定义严格的标准 (名称, 类型)
    expected_schema = [
        ("dataloader", DataLoader),
        ("model", torch.nn.Module),
        ("device", Any)
    ]

    # --- 检查参数数量 ---
    if len(params) != len(expected_schema):
        raise TypeError(
            f"验证函数 '{func_path}' 参数数量错误。\n"
            f"  期望: {len(expected_schema)} 个\n"
            f"  实际: {len(params)} 个"
        )

    # --- 检查参数名称 和 类型注解 ---
    for i, (u_param, (exp_name, exp_type)) in enumerate(zip(params, expected_schema)):

        # A. 检查名称 (Name Check)
        if u_param.name != exp_name:
            raise TypeError(
                f"验证函数 '{func_path}' 第 {i + 1} 个参数名称错误。\n"
                f"  期望名称: '{exp_name}'\n"
                f"  实际名称: '{u_param.name}'\n"
                f"  -> 请修改为 def func({exp_name}, ...):"
            )

        # 检查类型 (Type Check)
        # 注意：inspect._empty 表示用户没写类型注解

        if u_param.annotation != inspect._empty:
            # 简单的类型比较
            if exp_type is not Any and u_param.annotation != exp_type:
                # 尝试比较字符串表示 (防止 import 路径不同导致的类对象不一致)
                # 例如: 'torch.nn.modules.module.Module'
                if str(u_param.annotation) != str(exp_type):
                    raise TypeError(
                        f"验证函数 '{func_path}' 参数 '{u_param.name}' 类型注解错误。\n"
                        f"  期望类型: {exp_type}\n"
                        f"  实际类型: {u_param.annotation}"
                    )
    return func