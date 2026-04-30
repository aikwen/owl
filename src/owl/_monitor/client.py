from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from typing import Any

import httpx


DEFAULT_UNIX_SOCKET_NAME = ".owl-monitor.sock"


def stream_from_work_dir(
    work_dir: str | pathlib.Path,
    *,
    socket_name: str = DEFAULT_UNIX_SOCKET_NAME,
) -> Iterator[dict[str, Any]]:
    """从训练运行目录读取监控流。

    Args:
        work_dir: 当前训练运行目录。
        socket_name: Unix Domain Socket 文件名。

    Yields:
        训练监控快照字典。
    """
    socket_path = pathlib.Path(work_dir) / socket_name
    yield from stream_from_uds(socket_path)


def stream_from_uds(socket_path: str | pathlib.Path) -> Iterator[dict[str, Any]]:
    """通过 Unix Domain Socket 读取监控流。

    Args:
        socket_path: Unix Domain Socket 路径。

    Yields:
        训练监控快照字典。

    Raises:
        FileNotFoundError: socket 文件不存在。
        RuntimeError: 监控服务连接失败或返回异常数据。
    """
    socket_path = pathlib.Path(socket_path)

    if not socket_path.exists():
        raise FileNotFoundError(f"monitor socket not found: {socket_path}")

    transport = httpx.HTTPTransport(uds=str(socket_path))

    try:
        with httpx.Client(transport=transport, timeout=None) as client:
            with client.stream("GET", "http://owl-monitor/stream") as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"monitor stream failed: HTTP {response.status_code}"
                    )

                for line in response.iter_lines():
                    if not line:
                        continue

                    yield _loads_line(line)

    except httpx.ConnectError as exc:
        raise RuntimeError(f"failed to connect monitor socket: {socket_path}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"failed to read monitor stream: {exc}") from exc


def stream_from_http(base_url: str) -> Iterator[dict[str, Any]]:
    """通过 HTTP 读取监控流。

    该函数暂时保留给 Windows 使用。

    Args:
        base_url: 监控服务 HTTP 地址。

    Yields:
        训练监控快照字典。
    """
    base_url = base_url.rstrip("/")

    try:
        with httpx.Client(timeout=None) as client:
            with client.stream("GET", f"{base_url}/stream") as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"monitor stream failed: HTTP {response.status_code}"
                    )

                for line in response.iter_lines():
                    if not line:
                        continue

                    yield _loads_line(line)

    except httpx.HTTPError as exc:
        raise RuntimeError(f"failed to read monitor stream: {exc}") from exc


def health_from_uds(socket_path: str | pathlib.Path) -> dict[str, Any]:
    """通过 Unix Domain Socket 读取监控服务状态。

    Args:
        socket_path: Unix Domain Socket 路径。

    Returns:
        监控服务状态。
    """
    socket_path = pathlib.Path(socket_path)

    if not socket_path.exists():
        raise FileNotFoundError(f"monitor socket not found: {socket_path}")

    transport = httpx.HTTPTransport(uds=str(socket_path))

    try:
        with httpx.Client(transport=transport, timeout=5.0) as client:
            response = client.get("http://owl-monitor/health")
            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as exc:
        raise RuntimeError(f"failed to read monitor health: {exc}") from exc


def _loads_line(line: str) -> dict[str, Any]:
    """解析一行 NDJSON。

    Args:
        line: 单行 JSON 字符串。

    Returns:
        解析后的字典。

    Raises:
        RuntimeError: 当前行不是合法 JSON object。
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid monitor stream line: {line!r}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"monitor stream line is not an object: {line!r}")

    return data