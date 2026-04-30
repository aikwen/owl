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