"""Owl monitor 公开客户端 API。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from .._monitor.client import (
    health_from_address,
    stream_from_address,
    window_from_address,
)


def health(address: str) -> dict[str, Any]:
    """读取 monitor server 状态。

    Args:
        address: monitor 地址或端口号。
            例如 "127.0.0.1:64448" 或 "64448"。

    Returns:
        monitor server 状态字典。

    Example:
        >>> from owl.monitor.client import health
        >>> status = health("64448")
        >>> print(status)
        {
            "ok": True,
            "seq": 12,
            "buffer_size": 25,
            "cached": 12,
        }
    """
    return health_from_address(_normalize_address(address))


def health_with_retry(
    address: str,
    *,
    retry: int = 5,
    interval: float = 1.0,
) -> dict[str, Any]:
    """读取 monitor server 状态，失败后按固定间隔重试。

    Args:
        address: monitor 地址或端口号。
            例如 "127.0.0.1:64448" 或 "64448"。
        retry: 连接失败后的重试次数。
            0 表示只尝试一次。
        interval: 每次重试之间的等待时间，单位秒。

    Returns:
        monitor server 状态字典。

    Raises:
        Exception: 最后一次连接仍然失败时，抛出底层异常。

    Example:
        >>> from owl.monitor.client import health_with_retry
        >>> status = health_with_retry("64448", retry=10, interval=1.0)
        >>> print(status["ok"], status["cached"], status["buffer_size"])
        True 12 25
    """
    retry = max(0, retry)
    interval = max(0.0, interval)

    last_error: Exception | None = None

    for attempt in range(retry + 1):
        try:
            return health(address)
        except Exception as exc:
            last_error = exc

            if attempt >= retry:
                raise

            time.sleep(interval)

    if last_error is not None:
        raise last_error

    raise RuntimeError("failed to connect monitor server")

def window(address: str) -> list[dict[str, Any]]:
    """读取 monitor ring 当前缓存窗口。

    Args:
        address: monitor 地址或端口号。
            例如 "127.0.0.1:64448" 或 "64448"。

    Returns:
        当前缓存窗口中的快照列表，按 seq 升序排列。

    Example:
        >>> from owl.monitor.client import window
        >>> snapshots = window("64448")
        >>> for snapshot in snapshots:
        ...     print(snapshot["epoch"], snapshot["step"], snapshot["loss_metrics"])
    """
    return window_from_address(_normalize_address(address))


def stream(
    address: str,
    *,
    last_seq: int = 0,
) -> Iterator[dict[str, Any]]:
    """持续读取 monitor server 推送的训练快照。

    Args:
        address: monitor 地址或端口号。
            例如 "127.0.0.1:64448" 或 "64448"。
        last_seq: 已经读取到的最新快照序号。
            只会返回 seq 大于 last_seq 的快照。

    Yields:
        训练监控快照字典。

    Example:
        >>> from owl.monitor.client import stream, window
        >>> snapshots = window("64448")
        >>> last_seq = snapshots[-1]["seq"] if snapshots else 0
        >>> for snapshot in stream("64448", last_seq=last_seq):
        ...     step = snapshot["step"]
        ...     loss = snapshot["loss_metrics"].get("loss")
        ...     print(step, loss)

    Notes:
        stream 适合用户自定义实时消费逻辑，例如写入 CSV / JSONL，
        或将 loss 转成 float 后接入 matplotlib 等绘图工具。
    """
    yield from stream_from_address(
        _normalize_address(address),
        last_seq=last_seq,
    )



def _normalize_address(value: str) -> str:
    """把端口号或完整地址统一转换为 gRPC 地址。"""
    value = value.strip()

    if not value:
        raise ValueError("address cannot be empty")

    if ":" in value:
        return value

    if not value.isdigit():
        raise ValueError(
            'address must be a port number or "host:port", '
            f"got {value!r}"
        )

    return f"127.0.0.1:{value}"