from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def check_required_params(
    func: Callable[..., Any],
    required_params: tuple[str, ...],
) -> None:
    """检查函数签名是否包含指定的最小参数名集合。

    该函数只检查参数名，不检查类型注解。

    规则:
        1. 如果函数显式声明了 required_params 中的所有参数，则通过。
        2. 如果函数包含 **kwargs，则认为可以接收任意参数，也通过。
        3. 否则抛出 ValueError。

    Args:
        func: 被检查的函数或可调用对象。
        required_params: 必须包含的参数名集合。

    Raises:
        ValueError: 当函数签名缺少 required_params 中的参数时抛出。
    """
    sig = inspect.signature(func)

    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )

    if has_var_kwargs:
        return

    missing_params = [
        name for name in required_params
        if name not in sig.parameters
    ]

    if missing_params:
        raise ValueError(
            f"函数签名缺少必要参数: {missing_params}. "
            f"required={list(required_params)}, "
            f"signature={func.__name__}{sig}"
        )


def _has_var_kwargs(sig: inspect.Signature) -> bool:
    """判断函数签名中是否包含 **kwargs。"""
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )


def _required_param_names_from_base(
    base_sig: inspect.Signature,
) -> tuple[str, ...]:
    """从基类方法签名中提取需要子类显式兼容的参数名。

    这里只提取普通参数名，不包含 self，也不包含 *args / **kwargs。
    """
    names: list[str] = []

    for name, param in base_sig.parameters.items():
        if name == "self":
            continue

        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        names.append(name)

    return tuple(names)


def check_method_contract(
    child_cls: type,
    base_cls: type,
    method_name: str,
    *,
    require_var_kwargs: bool = True,
    check_return_annotation: bool = False,
) -> None:
    """检查子类方法是否满足基类方法的最小签名契约。

    该函数只检查参数名，不检查类型注解。

    规则:
        1. 子类必须具有 method_name 方法。
        2. 子类方法必须显式包含基类方法中的必要参数名。
        3. 默认要求子类方法保留 **kwargs，以兼容 Owl 后续扩展。
        4. 可选检查子类方法是否声明返回值注解。

    注意:
        **kwargs 只用于接收未来扩展参数，不用于替代当前核心参数。
        例如基类 forward 要求 batch_data/current_epoch/current_step，
        那么子类也应该显式声明这些参数。

    Args:
        child_cls: 用户实现的子类，例如 MyModel。
        base_cls: Owl 基类，例如 OwlModel。
        method_name: 需要检查的方法名，例如 "forward"。
        require_var_kwargs: 是否要求子类方法包含 **kwargs。
        check_return_annotation: 是否要求子类方法声明返回值注解。

    Raises:
        TypeError: 当子类方法签名不满足基类契约时抛出。
    """
    if not hasattr(base_cls, method_name):
        raise TypeError(
            f"基类 {base_cls.__name__} 不存在方法: {method_name}"
        )

    if not hasattr(child_cls, method_name):
        raise TypeError(
            f"子类 {child_cls.__name__} 不存在方法: {method_name}"
        )

    base_method = getattr(base_cls, method_name)
    child_method = getattr(child_cls, method_name)

    base_sig = inspect.signature(base_method)
    child_sig = inspect.signature(child_method)

    base_required_params = _required_param_names_from_base(base_sig)
    child_has_var_kwargs = _has_var_kwargs(child_sig)

    missing_params = [
        name for name in base_required_params
        if name not in child_sig.parameters
    ]

    if missing_params:
        raise TypeError(
            f"{child_cls.__name__}.{method_name} 签名不满足 "
            f"{base_cls.__name__}.{method_name} 契约。"
            f"缺少参数: {missing_params}. "
            f"required={list(base_required_params)}, "
            f"signature={child_cls.__name__}.{method_name}{child_sig}"
        )

    if require_var_kwargs and not child_has_var_kwargs:
        raise TypeError(
            f"{child_cls.__name__}.{method_name} 缺少 **kwargs。"
            f"Owl 组件方法建议保留 **kwargs，以兼容未来框架注入的扩展上下文参数。"
            f"signature={child_cls.__name__}.{method_name}{child_sig}"
        )

    if check_return_annotation:
        if child_sig.return_annotation is inspect.Signature.empty:
            raise TypeError(
                f"{child_cls.__name__}.{method_name} 缺少返回值类型注解。"
                f"建议与 {base_cls.__name__}.{method_name} 保持一致。"
            )