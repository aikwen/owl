
from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass

import uvicorn

from .app import create_monitor_app
from .config import MonitorConfig, MonitorTransport
from .state import MonitorState


@dataclass(slots=True)
class MonitorServerHandle:
    """监控服务句柄。

    Attributes:
        transport: 实际使用的传输方式。
        address: 监控服务地址。
        thread: 运行 uvicorn 的后台线程。
    """

    transport: MonitorTransport
    address: str
    thread: threading.Thread


def start_monitor_server(
    *,
    state: MonitorState,
    config: MonitorConfig,
    work_dir: str | pathlib.Path,
) -> MonitorServerHandle:
    """启动训练监控服务。

    Args:
        state: 训练监控运行状态。
        config: 训练监控配置。
        work_dir: 当前训练运行目录。

    Returns:
        监控服务句柄。
    """
    work_dir = pathlib.Path(work_dir)
    transport = config.normalized_transport()

    app = create_monitor_app(state)

    if transport == MonitorTransport.UNIX:
        return _start_unix_server(
            app=app,
            config=config,
            work_dir=work_dir,
        )

    if transport == MonitorTransport.HTTP:
        return _start_http_server(
            app=app,
            config=config,
        )

    raise ValueError(f"unsupported monitor transport: {transport}")


def _start_unix_server(
    *,
    app,
    config: MonitorConfig,
    work_dir: pathlib.Path,
) -> MonitorServerHandle:
    """通过 Unix Domain Socket 启动监控服务。

    Args:
        app: FastAPI 应用实例。
        config: 训练监控配置。
        work_dir: 当前训练运行目录。

    Returns:
        监控服务句柄。
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    socket_path = work_dir / config.unix_socket_name

    # 清理旧 sock 文件
    if socket_path.exists():
        socket_path.unlink()

    uvicorn_config = uvicorn.Config(
        app=app,
        uds=str(socket_path),
        access_log=False,
        log_level="warning",
    )
    server = uvicorn.Server(uvicorn_config)

    thread = threading.Thread(
        target=server.run,
        name="owl-monitor-server",
        daemon=True,
    )
    thread.start()

    return MonitorServerHandle(
        transport=MonitorTransport.UNIX,
        address=str(socket_path),
        thread=thread,
    )


def _start_http_server(
    *,
    app,
    config: MonitorConfig,
) -> MonitorServerHandle:
    """通过 HTTP 启动监控服务。

    该分支暂时保留给 Windows 使用。

    Args:
        app: FastAPI 应用实例。
        config: 训练监控配置。

    Returns:
        监控服务句柄。
    """
    uvicorn_config = uvicorn.Config(
        app=app,
        host=config.http_host,
        port=config.http_port,
        access_log=False,
        log_level="warning",
    )
    server = uvicorn.Server(uvicorn_config)

    thread = threading.Thread(
        target=server.run,
        name="owl-monitor-server",
        daemon=True,
    )
    thread.start()

    address = f"http://{config.http_host}:{config.http_port}"

    return MonitorServerHandle(
        transport=MonitorTransport.HTTP,
        address=address,
        thread=thread,
    )